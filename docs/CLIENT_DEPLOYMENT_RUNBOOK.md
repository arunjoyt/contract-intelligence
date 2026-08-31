# Client Deployment Runbook

**Audience:** internal — engineers deploying for a new client. **Topology:** Option B
(standalone app + ERPNext OAuth2), single VM.

`docs/DEPLOYMENT.md` § "AWS Deployment (Option B, single EC2 instance)" is the authoritative
command-level reference. This runbook is the ordered checklist with the client-specific decisions
and the known traps pulled to the front. Work top to bottom; each phase depends on the previous
one. Copy this into the client's project folder and tick as you go.

The [client onboarding doc](CLIENT_ONBOARDING.md) is the customer-facing view of the same
sequence.

---

## Environment strategy

**Do not build a dev / test / UAT / prod ladder for this.** It buys almost nothing here:

- The app is a known-good artifact — repo CI tests the pipeline with mocks. You are deploying a
  built thing, not developing against the client's site.
- Ingest is **read-only** against ERPNext (`GET` requests; webhooks are outbound *from* ERPNext).
  Pointing the RAG stack at prod ERPNext carries no write risk to ERPNext. The only things you
  write to are Qdrant and Langfuse, which are yours.
- Ingest is **idempotent and cheap**. Deterministic point IDs (`uuid5(docname:chunk_index)`) mean
  a re-run overwrites, never duplicates. Embedding a few hundred contracts with
  `text-embedding-3-small` costs cents and takes minutes. "Prod ingest" is not a one-shot cutover.
- A UAT env with **non-prod data proves nothing** — the two client-specific unknowns (does
  parsing/chunking handle their data shape; is retrieval good on their corpus) both need real
  prod data.

What *is* worth doing:

| | Purpose |
|---|---|
| **Sample smoke ingest** (step 4a below) | Catch data-shape problems on 15–30 real docs before the full run |
| **One staging RAG stack** | Second compose project / own infra, pointed at **prod ERPNext, read-only**. Use for Phase 5 validation, retrieval tuning, dry-running upgrades. Promote config to the prod stack once green. |
| **Prod RAG stack** | The client-facing deployment |

**If the client's change process mandates UAT** (governance sign-off, or a staged OpenAI data
approval — "pilot with 20 contracts, then the full corpus"): the step 4a sample ingest *is* the
pilot. Run it against a UAT ERPNext site if that is where the approved data lives, verify, then
repoint at prod. Document that UAT results only transfer if UAT data mirrors prod.

---

## Phase 0 — Pre-flight (collect before you start)

Do not provision anything until all of these are in hand. A missing ERPNext reachability or
API-permission item is the usual cause of a stalled deploy.

- [ ] **ERPNext base URL**, reachable over HTTPS from the public internet
  - verify from the target box later: `curl -I $ERPNEXT_URL`
- [ ] **ERPNext API key + secret** for a service user with read access to `Contract`,
  `Terms and Conditions`, and `User` (the last is needed for OAuth role resolution)
- [ ] **OpenAI API key** — or a model-provider swap plan per `docs/MODEL_PROVIDER_SWAP.md`
- [ ] **Client subdomain + DNS control** — you need to create two A records
- [ ] **Naming decision**: the two hostnames, e.g. `contracts.client.com` (frontend) and
  `api-contracts.client.com` (API)
- [ ] **Target server**: 2 vCPU / 8 GB / 30 GB, Ubuntu 22.04 LTS or Amazon Linux 2023
- [ ] **Document scope confirmed** — standard is `Contract` + `Terms and Conditions` +
  contract-attached PDFs. Anything wider is a code change, not config.
- [ ] **Data-handling sign-off** from the client (contract text is processed by OpenAI)

---

## Phase 1 — Provision infrastructure

*~1 day*

- [ ] Launch the VM (`t3.large` minimum — Qdrant, Langfuse + Postgres, FastAPI, Streamlit,
  nginx, and the reranker model all share one box)
- [ ] Security group — inbound `22` (your IP only), `80`, `443` (`0.0.0.0/0`); outbound all
- [ ] Allocate a static / Elastic IP and associate it (so DNS survives reboots)
- [ ] Two `A` records → the IP: the frontend host and the API host
- [ ] Wait for DNS propagation *before* running certbot — it validates via HTTP-01 on port 80
- [ ] Bootstrap the box:

```bash
# Docker + compose plugin + certbot
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER   # log out/in after
sudo apt install -y docker-compose-plugin certbot
```

- [ ] Issue certificates:

```bash
sudo certbot certonly --standalone \
  -d contracts.client.com \
  -d api-contracts.client.com
```

> ⚠️ **Never expose** ports `6333` (Qdrant), `3000` (Langfuse), `5432` (Postgres). They stay on
> the internal `rag_internal` Docker network. `docker-compose.yml` already binds 6333/3000 to
> `127.0.0.1` only — do not add public mappings. Operator access is via SSH tunnel (see Phase 6).

---

## Phase 2 — ERPNext-side configuration

*~½ day*

### OAuth client

- [ ] ERPNext desk → **Integrations → OAuth Client** → new
- [ ] Redirect URI: `https://api-contracts.client.com/auth/callback` (the API host, exact match
  — a trailing-slash mismatch breaks the callback)
- [ ] Record the generated `client_id` and `client_secret` → `.env`

### Webhook records × 4

For incremental re-indexing. Create via desk or REST API. Every record: **Request URL**
`https://api-contracts.client.com/webhook/erpnext`, method **POST**, structure **JSON**,
**Enable Security** checked, **Webhook Secret** = `WEBHOOK_SECRET`, and this exact JSON body:

```json
{"doctype": "{{ doc.doctype }}", "docname": "{{ doc.name }}"}
```

| Name | Doctype | Event |
|---|---|---|
| `contract-on-submit` | Contract | `on_submit` |
| `contract-on-update` | Contract | `on_update` |
| `contract-on-cancel` | Contract | `on_cancel` |
| `terms-on-update` | Terms and Conditions | `on_update` |

> ⚠️ **Traps.** **Enable Security** must be checked or Frappe never sends
> `X-Frappe-Webhook-Signature` and `webhook_handler._verify_signature()` rejects every call with
> 401. — `on_submit` is *not valid* for Terms and Conditions (not submittable). — The
> scripted-setup snippet is in `docs/DEPLOYMENT.md` § "Via REST API".

> **Role enforcement in Option B** is gated by the `ALLOWED_ROLES` env var plus the live role
> fetch in `api/auth/oauth2.py:fetch_user_roles()` — *not* the ERPNext Role Permissions Manager
> (that only matters for the unimplemented Option A desk page). No ERPNext permission changes are
> needed here.

---

## Phase 3 — Deploy the application

*~½ day*

- [ ] Clone the repo onto the box; `cp .env.example .env`
- [ ] Fill `.env` — full field reference in the [Environment reference](#environment-reference)
  section
- [ ] Generate secrets:

```bash
# 32-byte hex
openssl rand -hex 32   # WEBHOOK_SECRET, ADMIN_SECRET, JWT_SECRET,
                       # LANGFUSE_NEXTAUTH_SECRET, LANGFUSE_SALT, LANGFUSE_ADMIN_PASSWORD
# 16-byte hex
openssl rand -hex 16   # POSTGRES_USER, POSTGRES_PASSWORD
# LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY — any string, self-seed on first boot
```

- [ ] Set `FRONTEND_DOMAIN` / `API_DOMAIN` (bare hostnames, no scheme) — these drive nginx's
  config template at container start; `nginx.conf` is never hand-edited
- [ ] Set `OAUTH_REDIRECT_URI`, `FRONTEND_URL`, `PUBLIC_API_URL` to the public HTTPS URLs
- [ ] Confirm ERPNext is reachable from the box: `curl -I $ERPNEXT_URL`
- [ ] Launch:

```bash
docker compose up -d
docker compose ps   # all 6 services healthy: app, frontend, qdrant, langfuse, postgres, nginx
curl https://api-contracts.client.com/health
```

> ⚠️ **Fresh box only.** `POSTGRES_USER` / `POSTGRES_PASSWORD` are applied by Postgres *only on
> first init of an empty data dir*. On a box with an existing `pg_data` volume they must match
> the original values or the `langfuse` service fails to connect. Rotating them means recreating
> the volume.

> ⚠️ **nginx `default_server` trap.** The stock `nginx:alpine` image ships
> `/etc/nginx/conf.d/default.conf` ("Welcome to nginx", `server_name localhost`) which sorts
> first and catches any request whose Host header isn't ours (bots on the raw IP). Our template's
> port-80 block is marked `default_server` to win — verify a raw-IP request doesn't hit the stock
> page after launch.

- [ ] **Correct Langfuse's stale model prices** — run once now, before any traces accumulate:

  ```bash
  python scripts/langfuse_fix_model_prices.py   # reads LANGFUSE_* from .env
  ```

  Self-hosted Langfuse 2.x's bundled table prices `gpt-4o` at its mid-2024 launch rate
  ($5/$15 per 1M), so every `totalCost` in the UI/API is ~1.9× high (token counts are fine).
  The script adds a project-level price override at the current list rate ($2.50/$10 per 1M),
  matching `evaluation/results.json`'s `costs` block and `docs/BENCHMARKS.md`. Idempotent —
  re-run any time (e.g. after a Langfuse upgrade) with `--dry-run` to check first. See
  [#137](https://github.com/arunjoyt/contract-intelligence/issues/137).

---

## Phase 4 — Ingest

### 4a · Sample smoke ingest

*~1 hour · do this before the full run*

Ingest 15–30 representative documents and inspect the output **before** committing to the full
run. This is where client-specific data-shape problems surface while they are still cheap to fix.
It is also the vehicle for a staged client pilot (see *Environment strategy*).

**Pick the sample deliberately** — not just the first N. Cover: the largest contracts, contracts
with attached PDFs, contracts using any custom fields or a customized Contract doctype, a few
Terms and Conditions docs, and anything non-English.

**Run it into a throwaway collection** so nothing partial is left behind — set
`QDRANT_COLLECTION=contract_smoke` in the environment for this run, then switch back to
`contract` for the real ingest. (The full ingest is idempotent and would overwrite anyway, but a
separate collection keeps the smoke test self-contained.)

There is **no partial-ingest flag** — the sample runner is a trimmed copy of `_run_full_ingest`
in `api/main.py`, reusing the same helpers so it exercises the real code path:

```python
# scripts/sample_ingest.py  (sketch — reuses the production helpers verbatim)
import asyncio
from ingestion.erpnext_client import ERPNextClient
from ingestion.embedder import Embedder
from ingestion.webhook_handler import (
    prepare_doc_for_indexing, gather_chunks_for_doc, resolve_supplier_group,
)
from retrieval.vector_store import VectorStore

# ("Contract", "CON-2026-00012"), ("Terms and Conditions", "Standard-Terms"), ...
SAMPLE: list[tuple[str, str]] = [ ... ]

async def main() -> None:
    embedder, store = Embedder(), VectorStore()
    store.ensure_collection()            # honours QDRANT_COLLECTION — set it to contract_smoke
    async with ERPNextClient() as client:
        for doctype, name in SAMPLE:
            doc = await client.get_doc(doctype, name)
            supplier_group = await resolve_supplier_group(doctype, doc, client)
            text, metadata, force_single = prepare_doc_for_indexing(doctype, doc, supplier_group)
            chunks = await gather_chunks_for_doc(doctype, doc, text, force_single, client)
            if not chunks:
                print(f"!! {doctype} {name}: 0 chunks — empty body / unreadable PDF?")
                continue
            vectors = embedder.embed_texts([c["text"] for c in chunks])
            store.upsert_chunks([{**c, **metadata, "vector": v}
                                 for c, v in zip(chunks, vectors, strict=True)])
            print(f"ok {doctype} {name}: {len(chunks)} chunks")

asyncio.run(main())
```

**Then inspect the points** (Qdrant scroll snippets — `docs/DEPLOYMENT.md` §11, swap in
`contract_smoke`):

- [ ] **Chunk text is clean** — HTML stripped, no nav/boilerplate/markup residue, no mojibake
- [ ] **PDFs extracted** — attached-PDF chunks have real text, not empty (scanned-image PDFs
  `pypdf` can't read show as `0 chunks` or near-empty)
- [ ] **Metadata populated** — `supplier`, `start_date`, `end_date`, `status`, `company` are not
  all `None`. If they are, the client has a customized Contract doctype and
  `prepare_doc_for_indexing` needs field-name updates
- [ ] **`status` mapping sane** — cancelled contracts show `Cancelled`, disabled Terms show
  `Disabled`
- [ ] **Chunk counts reasonable** — a 40-page contract as 1 chunk means the body didn't parse;
  as 400 tiny chunks means the splitter hit something pathological
- [ ] **Ask 3–4 questions against the smoke collection** and confirm citations resolve to the
  right docnames

Fix any parser/metadata issues, re-run the sample, then drop `contract_smoke`
(`DELETE /collections/contract_smoke`) and proceed.

---

### 4b · Full ingest

*hours (scales with volume)*

Confirm `QDRANT_COLLECTION` is back to `contract`. Webhooks only cover documents changed *after*
setup — existing ERPNext data needs one manual full re-index.

```bash
curl -X POST https://api-contracts.client.com/ingest/full \
  -H "X-Admin-Secret: $ADMIN_SECRET"
```

- [ ] Runs as a background task — watch `docker compose logs app -f`
- [ ] Poll the Qdrant point count (climbs during ingest). `curl` is *not* in the `app` image —
  use `urllib` via exec:

```bash
docker compose exec app python3 -c \
 "import urllib.request,json; print(json.load(urllib.request.urlopen(
  'http://qdrant:6333/collections/contract'))['result']['points_count'])"
```

- [ ] Confirm both doctypes indexed — count by `source_doctype` (`Contract`, then
  `Terms and Conditions`); snippet in `docs/DEPLOYMENT.md` §11

---

## Phase 5 — Validate & tune

*~1–2 days*

### End-to-end smoke tests

- [ ] Browser → frontend URL → **Login with ERPNext** → sign in as a `Purchase Manager` → ask a
  question → confirm answer + `[docname]` citations
- [ ] Negative: user outside `ALLOWED_ROLES` → `403`
- [ ] `POST /query` with no `Authorization` header → `401`
- [ ] Direct TCP to `<ip>:6333` → connection refused
- [ ] Edit a contract in ERPNext → confirm re-index in `app` logs / rising point count
- [ ] Set `JWT_EXPIRY_HOURS=0` temporarily → next request bounces to login → revert

### Guided review with the client expert

**This is the default acceptance step** — the repo's own `CLAUDE.md` notes the LLM judge is too
noisy to gate on at any dataset size. The formal RAGAS benchmark below is an optional, separately
scoped add-on.

- [ ] Copy `docs/templates/guided_review.md` → `evaluation/client/<client>_guided_review.md`
  (gitignored — it names real documents) and fill in the header
- [ ] The client expert brings **10–15 real questions** their team actually asks — mix single-clause
  lookups, cross-document ("which of our contracts…"), and 1–2 you expect it to *refuse*
- [ ] Run them live, score each row against the worksheet rubric (answer verdict + citation verdict)
- [ ] **Metadata-filter vocabulary check** — does the client's ERPNext use doctype names or `status`
  values other than the defaults (`Contract` / `Terms and Conditions`; `Cancelled` / `Active` /
  `Unsigned`)? Check a real `status` value in Qdrant against their wording. If so, set
  `METADATA_FILTER_DOCTYPE_KEYWORDS` / `METADATA_FILTER_STATUS_KEYWORDS` in `.env` (JSON — see the
  Environment reference) and re-test the affected questions. No source edit, no re-ingest.
- [ ] When a group of questions fails the same way, nudge **one knob**, in the reach order from
  `docs/PIPELINE_TUNING.md` § Per-client tuning (`RETRIEVAL_TOP_K` → prompt vocabulary →
  `RERANK_TOP_N`), and re-run the failing questions
- [ ] Record the decision + any `.env` changes in the worksheet — it is the validation summary for
  the client's deployment record

### Optional: client-specific RAGAS benchmark

> ⚠️ **Not a drop-in.** `evaluation/evaluate.py` scores against `evaluation/test_dataset.json` —
> 92 entries hand-authored against *our demo fixtures* (names real suppliers, real docnames). It
> is meaningless against a client corpus. A real benchmark needs a client-specific dataset.

> 🔒 **All client eval artifacts live under `evaluation/client/` — a gitignored directory (#118).**
> The client's questions, contract text, and scores never enter this repo's history. See
> `evaluation/client/README.md`.

> 📍 **Which collection.** Every command below scores/reads a Qdrant collection. Run them
> against the **staging RAG stack's** collection (see [Environment strategy](#environment-strategy)),
> or — if there is no staging stack — read-only against the prod `<client>` collection *before*
> the frontend is opened to users. Per-client validation never rebuilds the collection (the only
> per-client knobs are `RETRIEVAL_TOP_K` / prompt / `RERANK_TOP_N`, all query-time), so a
> read-only pass is safe; there is **no separate eval collection to stand up**. `<coll>` below is
> that collection name.

- [ ] **Draft candidates with `scripts/generate_eval_set.py`** (this is the #127 per-client tool):
  `python scripts/generate_eval_set.py --collection <coll> --per-class 12 --output evaluation/client/<client>_candidates.review.json`
  → a human-review file grounded in the client's own indexed chunks. Review every entry with the
  client expert, verify each answer against the real document, set `split`, drop the `_`-prefixed
  helper keys, and save as `evaluation/client/<client>_dataset.json`. Per-entry schema:
  - `case_class`, `capability` — labels for slicing
  - `question`, `split` (`"dev"` / `"test"`)
  - `ground_truth_contexts` — the framed chunk text(s) that should be retrieved;
    **empty list = grounded-refusal case**
  - `ground_truth_answer` — reference answer with `[docname]` citations
- [ ] For anything the generator missed, pull candidate chunk text from Qdrant with the
  scroll/filter snippets in `docs/DEPLOYMENT.md` §11
- [ ] Sync the client dataset into *their own* Langfuse project so runs are browsable there:
  `LANGFUSE_PUBLIC_KEY=… LANGFUSE_SECRET_KEY=… LANGFUSE_HOST=… python evaluation/push_dataset.py --dataset evaluation/client/<client>_dataset.json`
- [ ] Run:
  `python evaluation/evaluate.py --collection <coll> --dataset evaluation/client/<client>_dataset.json --output evaluation/client/<client>_results.json`
- [ ] **Do not** compare against `evaluation/results.baseline.json` — different corpus
- [ ] Read the per-`case_class` scores against the reference numbers in `docs/PIPELINE_TUNING.md`
  § Per-client tuning by eye — the LLM judge is too noisy for a hard gate (`CLAUDE.md`). When a
  slice sits clearly low, nudge **one** knob in the reach order there
  (`RETRIEVAL_TOP_K` → prompt → `RERANK_TOP_N`), record the `.env` override in the guided-review
  worksheet, and re-run the failing slice. A full Step A–E sweep per client is almost never
  warranted (#127).

---

## Phase 6 — Handover

*~½ day*

### Backups

> The backup **scripts** are committed (`scripts/backup_all.sh` / `restore_all.sh`); **nothing
> schedules them**. No cron entry, systemd timer, or off-box copy / retention logic ships — you
> wire that up here. Strategy + automation is being revisited in
> [#116](https://github.com/arunjoyt/contract-intelligence/issues/116); **the client's ERPNext
> is out of our backup scope** and stays with the client.

- [ ] Copy `scripts/backup_all.local.sh.example` → `scripts/backup_all.local.sh` (gitignored) and
  fill in `BACKUP_ROOT` (+ `BENCH_*` / `SITE_NAME` only if ERPNext is self-hosted on this box)
- [ ] Run `./scripts/backup_all.sh` once by hand and confirm it produces files. It wraps Qdrant
  snapshot + Langfuse `pg_dump` + (bench only if present) ERPNext backup; each step
  skips-with-warning rather than failing the run
- [ ] **If the client runs ERPNext elsewhere** (the usual case): the bench step skips — you cover
  Qdrant + Langfuse only; the client owns ERPNext backups
- [ ] Add the schedule yourself, e.g.:
  `0 2 * * * cd /path/to/contract-intelligence && ./scripts/backup_all.sh >> /var/log/ci-backup.log 2>&1`
- [ ] Decide retention + **off-box copy** — `backup_all.sh` only writes to `$BACKUP_ROOT` on the
  local disk. A daily EBS snapshot of the instance (covers `qdrant_data` / `pg_data`) is the
  simplest single-box answer; otherwise `aws s3 sync $BACKUP_ROOT s3://…` after the backup run
- [ ] Note in the handover doc: `.env` is **not** in any backup by design — it lives in the
  client's secrets manager

### Ops docs to leave

- [ ] Cert renewal cron:
  `0 3 * * * certbot renew --quiet && docker compose exec nginx nginx -s reload`
- [ ] Deploy procedure: `git pull && docker compose up -d --build`
  - `.env` value change → `docker compose up -d --force-recreate <service>`
  - code change → `--build` (source is baked into the image; `--force-recreate` alone keeps stale
    code)
- [ ] Operator access to Langfuse / Qdrant — SSH tunnel only:
  - `ssh -L 3000:localhost:3000 user@<ip>` → `http://localhost:3000`
    (login `admin@localhost.local` / `LANGFUSE_ADMIN_PASSWORD`)
  - `ssh -L 6333:localhost:6333 user@<ip>` → `http://localhost:6333/dashboard`
- [ ] After any Langfuse version bump: re-run `python scripts/langfuse_fix_model_prices.py
  --dry-run` — a newer bundled price table may make the `gpt-4o` override redundant (or need
  a new one). See Phase 3.

### Credentials

> ⚠️ **`.env` transfer.** Move `.env` into the client's secrets manager (1Password / Bitwarden /
> Vault) as one secure note, or re-derive each value at deploy time. **Never** a plain file copy
> in a repo, shared drive, or unencrypted archive. It holds `OPENAI_API_KEY`, ERPNext
> key/secret, `JWT_SECRET`, `ADMIN_SECRET`, `WEBHOOK_SECRET`, OAuth client secret — nothing in
> the stack starts without it.

---

## Environment reference

`.env` — copied from `.env.example`.

**Source key:** `generate` = `openssl rand` · `erpnext` = from the client's ERPNext ·
`client` = from the client / your decision

| Variable | Source | Notes |
|---|---|---|
| `ERPNEXT_URL` | client | HTTPS base URL; must be reachable from the box |
| `ERPNEXT_API_KEY` | erpnext | service user, read on Contract / Terms / User |
| `ERPNEXT_API_SECRET` | erpnext | |
| `ERPNEXT_OAUTH_CLIENT_ID` | erpnext | Integrations → OAuth Client |
| `ERPNEXT_OAUTH_CLIENT_SECRET` | erpnext | |
| `OAUTH_REDIRECT_URI` | client | `https://<api>/auth/callback` — exact match to OAuth client |
| `OPENAI_API_KEY` | client | or model-provider swap |
| `WEBHOOK_SECRET` | generate | hex 32; must match every Webhook record |
| `ADMIN_SECRET` | generate | hex 32; gates `POST /ingest/full` |
| `JWT_SECRET` | generate | hex 32 |
| `JWT_EXPIRY_HOURS` | — | default `8` |
| `ALLOWED_ROLES` | — | default: Purchase Manager, Purchase User, Accounts User, System Manager |
| `LANGFUSE_NEXTAUTH_SECRET` | generate | hex 32 |
| `LANGFUSE_SALT` | generate | hex 32 |
| `LANGFUSE_ADMIN_PASSWORD` | generate | hex 32; UI login |
| `LANGFUSE_PUBLIC_KEY` / `_SECRET_KEY` | generate | any string; self-seed on first boot |
| `POSTGRES_USER` / `_PASSWORD` | generate | hex 16; **fresh box only** — else match existing volume |
| `FRONTEND_DOMAIN` / `API_DOMAIN` | client | bare hostnames, no scheme; drive nginx template |
| `FRONTEND_URL` / `PUBLIC_API_URL` | client | full `https://` URLs |
| `QUERY_REWRITE_STRATEGY` | — | default `hyde` |
| `OPENAI_MODEL` / `EMBEDDING_MODEL` | — | defaults `gpt-4o` / `text-embedding-3-small`. Changing embedding dim → recreate collection + full re-ingest |
| `METADATA_FILTER_DOCTYPE_KEYWORDS` | client | JSON `{"<doctype>": ["kw", …]}`. Only if the client's doctype names differ. Default: `{"Contract": ["contract"], "Terms and Conditions": ["terms and conditions"]}` |
| `METADATA_FILTER_STATUS_KEYWORDS` | client | JSON `{"kw": "<status value>"}` — a *map*, so e.g. `{"expired": "Inactive"}`. Default: `{"cancelled": "Cancelled", "active": "Active", "unsigned": "Unsigned"}`. Set from Phase 5. |

---

## Known limitations — disclose to the client team

- **Role changes lag by up to 8h.** Roles are baked into the JWT at login; a revoked role still
  works until the token expires.
- **Metadata filters are keyword-only.** Doctype and status only — no date-range parsing, no LLM
  extraction. The keyword vocabulary is per-client config (`METADATA_FILTER_*` env vars), not the
  hardcoded default, but it stays a plain substring match.
- **RAGAS is not in CI and not a gate.** Quality is validated manually.
- **Single box, no HA.** Host reboot recovers automatically (`restart: unless-stopped`); a host
  failure is a restore-from-backup event.

---

## Troubleshooting

| Symptom | First check |
|---|---|
| `/health` fails after launch | `docker compose logs app` — usually a malformed `.env` value or unreachable `ERPNEXT_URL` |
| Login redirects in a loop / errors on callback | `OAUTH_REDIRECT_URI` must exactly match the ERPNext OAuth client (trailing slash, scheme, host); `PUBLIC_API_URL` set to the public API URL |
| Every webhook returns `401 Invalid webhook signature` | "Enable Security" unchecked on the Webhook record, or `webhook_secret` ≠ `WEBHOOK_SECRET` |
| Full ingest indexes 0 documents | API key/secret permissions on `Contract`; `ERPNEXT_URL` reachable from the box; check `app` logs for the fetch error |
| `langfuse` service crash-loops on boot | Postgres credential mismatch against an existing `pg_data` volume — see Phase 3 |
| Raw-IP request shows "Welcome to nginx" | stock `default.conf` winning — see Phase 3 nginx trap |
| Answers empty or citations wrong | confirm reranker warmed at startup (`app` logs); then `docs/PIPELINE_TUNING.md` |
