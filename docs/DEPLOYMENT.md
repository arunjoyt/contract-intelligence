# Deployment Strategy

## Overview

The Contract Intelligence system is deployed alongside ERPNext, with access restricted to specific roles (`Purchase Manager`, `Purchase User`, `Accounts User`, `System Manager`). Two deployment options are available — choose based on your ERPNext hosting type.

| | Option A: Frappe App | Option B: Standalone + OAuth2 |
|---|---|---|
| **ERPNext hosting** | Self-hosted bench, or Frappe Cloud (paid plan) | Any — cloud, SaaS, self-hosted |
| **User experience** | Native ERPNext desk — no separate login | Separate URL, "Login with ERPNext" button |
| **Role enforcement** | `frappe.get_roles()` — live on every request | JWT with 8-hour expiry window |
| **Custom app required** | Yes | No |
| **Best for** | Teams who live in the ERPNext desk all day | ERPNext hosted where bench access is unavailable |

---

## Option A: Frappe Custom App (Native ERPNext Desk Integration)

### How It Works

A lightweight Frappe app (`contract_intelligence`) is installed on the ERPNext site. It adds a **Workspace shortcut and Desk Page** visible only to the allowed roles — controlled via Frappe's standard Role Permissions Manager. The page contains a chat widget that calls a **whitelisted Python method** server-side, which enforces the role check and proxies the query to the FastAPI backend over the private network.

The FastAPI RAG backend still runs as a separate Docker Compose service. The Frappe app is only the UI and auth layer.

### ERPNext Setup

1. `bench get-app https://github.com/your-org/contract_intelligence`
2. `bench --site erp.example.com install-app contract_intelligence`
3. `bench --site erp.example.com migrate`
4. In ERPNext desk: **Role Permissions Manager** → set Page `Contract Intelligence` visible to `Purchase Manager`, `Purchase User`, `Accounts User`, `System Manager`

> On **Frappe Cloud**: custom app installation is available on Business / Unlimited plans. On ERPNext SaaS (erpnext.com), use Option B instead.

### Auth Flow

1. User is already logged into ERPNext desk — no separate login required
2. User opens the "Contract Intelligence" Workspace item (hidden from all other roles)
3. The Desk Page calls `frappe.call('contract_intelligence.api.query', {question: ...})`
4. The whitelisted method runs inside the bench process; `frappe.session.user` is already resolved
5. Method checks allowed roles: `if not any(r in frappe.get_roles() for r in ALLOWED_ROLES): raise frappe.PermissionError`
6. If authorized, method calls `http://127.0.0.1:8000/query` with a short-lived HMAC token (`HMAC-SHA256(username + timestamp, INTERNAL_TOKEN_SECRET)`, 30s TTL)
7. FastAPI validates the HMAC and timestamp window, then runs the RAG pipeline
8. Result is returned through the whitelisted method to the chat widget

### New Code Modules

**Frappe app (separate repository):**
```
apps/contract_intelligence/
├── contract_intelligence/
│   ├── hooks.py                    # app metadata, workspace registration
│   ├── api.py                      # @frappe.whitelist() def query(question, filters)
│   │                               #   → role check → HMAC token → POST 127.0.0.1:8000/query
│   └── public/js/
│       └── contract_chat.js        # chat widget embedded in the Desk Page
└── setup.py
```

**Inside `contract-intelligence` project:**
```
api/
└── auth/
    └── internal_token.py           # validate_hmac_token() — FastAPI dependency for /query
```

### New Environment Variables

```bash
# Shared between bench site_config.json and Docker Compose .env
INTERNAL_TOKEN_SECRET=<random-256-bit-hex>
INTERNAL_TOKEN_TTL_SECONDS=30
```

### Infrastructure Topology

```
Internet (443)
    └── ERPNext nginx (bench-managed)
          └── erp.example.com → ERPNext desk
                  └── [loopback: 127.0.0.1:8000]
                        └── FastAPI (Docker, loopback-bound)
                              [Docker network: rag_internal]
                              ├── Qdrant    (private — no public port)
                              └── Langfuse  (private — no public port)
                                    └── Postgres (private)
```

**`docker-compose.yml` for Option A:**
```yaml
services:
  app:
    ports:
      - "127.0.0.1:8000:8000"   # loopback only — bench can reach it; internet cannot
```

ERPNext webhook URL (internal — no TLS needed):
```
http://127.0.0.1:8000/webhook/erpnext
```

### Verification

1. `docker compose up -d` — FastAPI starts, bound to `127.0.0.1:8000`
2. Log into ERPNext desk as `Purchase Manager` — "Contract Intelligence" Workspace item is visible
3. Log in as a user with no allowed role — Workspace item is hidden; direct `frappe.call` raises `PermissionError`
4. Submit a query from the desk page — answer and sources returned
5. `curl http://server-ip:8000/query` from outside — connection refused (loopback binding)

---

## Option B: Standalone App with ERPNext OAuth2 + JWT Sessions

### How It Works

The app is deployed as a separate service. Users log in with their **normal ERPNext username and password** via OAuth2 Authorization Code flow with PKCE — the RAG app never handles the password. After login, FastAPI mints a JWT for the session.

Works with any ERPNext hosting. No custom app installation required.

### One-Time ERPNext Setup

OAuth2 is built into Frappe — no custom app needed:

1. Go to ERPNext desk → **Integrations → OAuth Client**
2. Create a new OAuth2 client
3. Set **Redirect URI** to `https://api.contract-intelligence.example.com/auth/callback` (the API domain —
   `/auth/callback` is a FastAPI route, not served by the Streamlit frontend)
4. Note the generated `client_id` and `client_secret`

### Auth Flow

1. User navigates to `https://contract-intelligence.example.com` — sees a **"Login with ERPNext"** button
2. Clicking redirects to:
   ```
   {ERPNEXT_URL}/api/method/frappe.integrations.oauth2.authorize
     ?client_id=...&redirect_uri=...&response_type=code&code_challenge=...&scope=openid+all
   ```
3. User logs in on the ERPNext site — the RAG app never sees the password
4. ERPNext redirects back to `https://api.contract-intelligence.example.com/auth/callback?code=...`
5. FastAPI exchanges the code for an access token via `POST {ERPNEXT_URL}/api/method/frappe.integrations.oauth2.get_token`
6. FastAPI resolves the user's roles in three steps (`fetch_user_roles()` in `api/auth/oauth2.py`):
   a. verifies identity with the OAuth access token via `GET .../oauth2.openid_profile` (email from
      the profile, not the OIDC `sub`, which is a pairwise hash unusable for the lookup below)
   b. resolves email → ERPNext user `docname` via `GET {ERPNEXT_URL}/api/resource/User` — this and
      the next call use the **server's own** `ERPNEXT_API_KEY`/`ERPNEXT_API_SECRET`, not the OAuth
      token, because the OAuth Bearer token can't read the `User` doctype via the resource API
   c. fetches `GET {ERPNEXT_URL}/api/resource/User/{docname}?fields=["name","roles"]` with the same
      server credentials
7. If no allowed roles → `403 Forbidden`
8. If authorized, FastAPI mints a signed **JWT** (`sub=username`, `roles=[...]`, `exp=now+8h`) and returns it
9. Streamlit stores the JWT in `st.session_state`; every `POST /query` carries `Authorization: Bearer <jwt>`
10. FastAPI validates the JWT on each request — no ERPNext round-trip on queries

Role changes in ERPNext take effect at next login (8-hour JWT expiry).

**Duplicate callback tolerance:** ERPNext's own OAuth confirmation page (`oauth_confirmation.html`)
binds "Allow" with a plain click handler calling `window.location.replace(success_url)` — a
double-click (or a trackpad registering one tap as two) fires it twice, sending two requests for the
same `code`/`state` before the first navigation completes. `/auth/callback` caches each completed
login's redirect for 60s so a duplicate request within that window replays the same successful
redirect instead of failing on the already-consumed `state`.

### Allowed Roles

| ERPNext Role | Access |
|---|---|
| `Purchase Manager` | Yes |
| `Purchase User` | Yes |
| `Accounts User` | Yes |
| `System Manager` | Yes (admin/testing) |
| All other roles | No |

### New Code Modules

```
api/
├── auth/
│   ├── oauth2.py           # build_authorize_url(), exchange_code_for_token(), fetch_user_roles()
│   ├── pkce.py             # generate_code_verifier(), generate_code_challenge()
│   ├── jwt_handler.py      # mint_token(), decode_token()
│   └── dependencies.py     # FastAPI Depends(): get_current_user, require_allowed_role
├── routers/
│   └── auth.py             # GET /auth/login (redirect), GET /auth/callback (token exchange + role check)

frontend/
├── auth_ui.py              # "Login with ERPNext" button, logout, OAuth redirect handling
└── app.py                  # gate main chat UI behind st.session_state.jwt
```

Key additions to `api/main.py`: mount `auth_router`; apply `require_allowed_role` dependency to `POST /query`.

### New Environment Variables

```bash
# OAuth2 client — created in ERPNext desk → Integrations → OAuth Client
ERPNEXT_OAUTH_CLIENT_ID=<from ERPNext>
ERPNEXT_OAUTH_CLIENT_SECRET=<from ERPNext>
OAUTH_REDIRECT_URI=https://api.contract-intelligence.example.com/auth/callback

# JWT session
JWT_SECRET=<random 256-bit hex>
JWT_EXPIRY_HOURS=8

# Role gate
ALLOWED_ROLES=Purchase Manager,Purchase User,Accounts User,System Manager
```

Also required (not new — already used by ingestion, but Option B's role-fetch depends on them too):
`ERPNEXT_API_KEY`/`ERPNEXT_API_SECRET`. The OAuth access token only proves identity; steps 6b/6c of
the Auth Flow above use these server-side credentials to read the `User` doctype's roles, since the
OAuth Bearer token doesn't have resource-API read access to it.

### Infrastructure Topology

```
Internet (443/80)
    └── Nginx
          ├── contract-intelligence.example.com     → Streamlit  (internal: 8501)
          └── api.contract-intelligence.example.com → FastAPI    (internal: 8000)
                  [Docker network: rag_internal — no external routing]
                  ├── Qdrant    (6333, private)
                  └── Langfuse  (3000, private)
                        └── Postgres (5432, private)

ERPNext cloud instance
    └── Called by FastAPI during /auth/callback (token exchange + role fetch)
    └── Sends webhooks to https://api.contract-intelligence.example.com/webhook/erpnext
```

**`docker-compose.yml` for Option B** (see the actual file at repo root — this is a summary, not a
verbatim copy):
```yaml
networks:
  rag_internal:
    driver: bridge
    # No public ports on the network itself — nginx is the sole ingress for
    # app traffic. All services get restart: unless-stopped (survives
    # instance stop/reboot).

services:
  app:                    # FastAPI — no ports: mapping
    networks: [rag_internal]

  frontend:               # Streamlit — no ports: mapping
    networks: [rag_internal]

  qdrant:
    ports: ["127.0.0.1:6333:6333"]  # loopback-only — SSH tunnel, see §11
    networks: [rag_internal]

  langfuse:
    ports: ["127.0.0.1:3000:3000"]  # loopback-only — SSH tunnel, see §12
    networks: [rag_internal]

  postgres:
    networks: [rag_internal]

  nginx:
    ports: ["80:80", "443:443"]
    networks: [rag_internal]  # sole bridge to public internet traffic
```

**Nginx configuration notes:**
- Domains are set once via `FRONTEND_DOMAIN`/`API_DOMAIN` in `.env` — substituted into nginx's
  config automatically at container start (`nginx/templates/contract-intelligence.conf.template`), no
  manual editing of `nginx.conf` required
- The port-80 redirect block is marked `default_server` — without it, the base `nginx:alpine`
  image's own leftover `/etc/nginx/conf.d/default.conf` (a stock "Welcome to nginx" page,
  `server_name localhost`) sorts first alphabetically and silently catches any request whose Host
  header isn't one of ours (e.g. bots hitting the raw IP)
- `proxy_read_timeout 120s` on the FastAPI location — LLM calls take 20–30s
- `proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for`
- TLS via certbot / Let's Encrypt
- Streamlit requires WebSocket: `proxy_http_version 1.1; proxy_set_header Upgrade $http_upgrade; proxy_set_header Connection "upgrade";`

**ERPNext webhook URL:**
```
https://api.contract-intelligence.example.com/webhook/erpnext
```

### Verification

1. `docker compose up -d` — all services start; nginx on 443
2. `https://contract-intelligence.example.com` — "Login with ERPNext" button visible
3. Click → redirected to ERPNext login → log in as `Purchase Manager` → redirected back to chat UI
4. Repeat with a user outside allowed roles → `403 Access Denied`
5. Submit a contract query → answer with source citations returned
6. Set `JWT_EXPIRY_HOURS=0` temporarily → next request redirects back to login
7. Trigger ERPNext webhook → document re-indexed; updated answer returned
8. `POST /query` with no `Authorization` header → `401 Unauthorized`
9. Direct TCP to `server-ip:6333` (Qdrant) → connection refused (no public port)

---

## ERPNext Webhook Setup (both options)

This is a **one-time ERPNext configuration** required for incremental re-indexing. It applies regardless of which auth option you chose. Webhooks fire when documents are submitted, cancelled, or updated; the handler at `POST /webhook/erpnext` verifies the HMAC signature, fetches the full document from ERPNext, and re-indexes it in Qdrant.

### Required webhook records

Create the following Webhook records in ERPNext desk (or via the REST API). For each one:
- **Request URL**: `http://127.0.0.1:8000/webhook/erpnext` (Option A / local) or `https://api.contract-intelligence.example.com/webhook/erpnext` (Option B / prod)
- **Request Method**: POST
- **Request Structure**: JSON
- **Enable Security**: must be checked. Without it, Frappe never sends the
  `X-Frappe-Webhook-Signature` header, and `_verify_signature()` in
  `ingestion/webhook_handler.py` will reject every request with `401 Invalid webhook signature`.
- **Webhook Secret**: must match `WEBHOOK_SECRET` in your `.env`
- **JSON Request Body** (`webhook_json` — this field only appears once Request Structure is set
  to `JSON`; the separate "Webhook Data" table is for `Form URL-Encoded` and doesn't apply here).
  Enter this Jinja template exactly:

  ```json
  {"doctype": "{{ doc.doctype }}", "docname": "{{ doc.name }}"}
  ```

  The handler reads `docname` directly from the JSON body, so this maps straight through with
  no extra field-mapping step needed.

| Webhook Name              | Doctype             | Event       | Why                                               |
|---------------------------|---------------------|-------------|----------------------------------------------------|
| `contract-on-submit`      | Contract            | `on_submit` | Index contracts (and attached PDFs) when submitted |
| `contract-on-update`      | Contract            | `on_update` | Re-index contracts saved/modified                  |
| `contract-on-cancel`      | Contract            | `on_cancel` | Re-index with status=Cancelled when cancelled      |
| `terms-on-update`         | Terms and Conditions| `on_update` | Not submittable; update only                       |

> **Note:** `on_submit` is not valid for Terms and Conditions — it is not a submittable doctype in ERPNext.

> **Note:** `on_update_after_submit` is **not configured** for any doctype here. Investigation (against
> Purchase Order, since removed from scope — see [Future Enhancements](#future-enhancements) below)
> confirmed that Frappe 15 does not reliably fire this event via REST API or desk UI saves. This
> platform limitation was only verified against Purchase Order; if edits to an already-submitted
> Contract need to be re-indexed automatically, re-verify against Contract before assuming the same
> gap applies.

### Via REST API (scripted setup)

```python
import urllib.request, json

BASE = "http://127.0.0.1:8005"   # your ERPNext URL
AUTH = "token <api_key>:<api_secret>"
SECRET = "<your WEBHOOK_SECRET>"
URL = "http://127.0.0.1:8000/webhook/erpnext"

WEBHOOKS = [
    ("contract-on-submit", "Contract",            "on_submit"),
    ("contract-on-update", "Contract",            "on_update"),
    ("contract-on-cancel", "Contract",            "on_cancel"),
    ("terms-on-update",    "Terms and Conditions","on_update"),
]

WEBHOOK_JSON = '{"doctype": "{{ doc.doctype }}", "docname": "{{ doc.name }}"}'

for wname, doctype, event in WEBHOOKS:
    payload = json.dumps({
        "doctype": "Webhook", "name": wname,
        "webhook_doctype": doctype, "webhook_docevent": event,
        "request_url": URL, "request_method": "POST",
        "request_structure": "JSON", "enabled": 1,
        "enable_security": 1, "webhook_secret": SECRET,
        "webhook_json": WEBHOOK_JSON,
    }).encode()
    req = urllib.request.Request(f"{BASE}/api/resource/Webhook", data=payload,
        headers={"Authorization": AUTH, "Content-Type": "application/json"}, method="POST")
    resp = urllib.request.urlopen(req)
    print(json.loads(resp.read())["data"]["name"])
```

### How modifications are handled

The webhook handler (`ingestion/webhook_handler.py`) is event-agnostic: for any event it calls `vector_store.delete_by_docname(docname)` then re-indexes the full document. This means:

- **New PO submitted** → `on_submit` fires → indexed fresh, including any attached PDFs
- **PO amended** → amendment creates a new PO (new `docname`) → `on_submit` fires again on the amendment
- **PO cancelled** → `on_cancel` fires → re-indexed with status=Cancelled
- **Invoice submitted/cancelled** → indexed/re-indexed the same way as POs (no attachment handling)
- **Contract updated** → `on_update` fires → re-indexed with latest content, including any attached PDFs
- **Terms and Conditions updated** → `on_update` fires → re-indexed
- **Scorecard updated** → `on_update` fires → re-indexed

> **Known gap:** edits to a submitted PO (e.g. changing delivery date or remarks) do not trigger re-indexing. Frappe 15 does not reliably fire `on_update_after_submit` via API or desk saves. The Qdrant vector for a live PO will reflect the state at submission time until the next `on_cancel` or a manual full re-index (`POST /ingest/full`). See [Future Enhancements](#future-enhancements).

---

## ERPNext Bench Backup & Restore

Standard `bench` commands for snapshotting and resetting an ERPNext site's data (e.g. dev/demo data
used for testing this app, or before/after a risky change). Applies to any bench-managed site,
local or remote — run from the bench directory (`cd <bench-dir>` first).

**Backup** (`--with-files` captures attached files, e.g. Contract PDFs, not just the database):

```bash
bench --site <site-name> backup --with-files
```

Writes to `sites/<site-name>/private/backups/`, timestamped, e.g.:
```
<timestamp>-<site-name>-database.sql.gz
<timestamp>-<site-name>-private-files.tar
<timestamp>-<site-name>-files.tar               # public files, despite the plain name
<timestamp>-<site-name>-site_config_backup.json
```

**Restore** — ⚠️ overwrites the site's database and files entirely:

```bash
bench --site <site-name> restore \
  sites/<site-name>/private/backups/<timestamp>-<site-name>-database.sql.gz \
  --with-private-files sites/<site-name>/private/backups/<timestamp>-<site-name>-private-files.tar \
  --with-public-files sites/<site-name>/private/backups/<timestamp>-<site-name>-files.tar
```

**Copying to a different machine** (e.g. local backup → VPS) — `scp` the four backup files to the
target bench's `sites/<site-name>/private/backups/` directory first, then run the restore command
there:

```bash
scp sites/<site-name>/private/backups/<timestamp>-<site-name>-database.sql.gz \
    sites/<site-name>/private/backups/<timestamp>-<site-name>-private-files.tar \
    sites/<site-name>/private/backups/<timestamp>-<site-name>-files.tar \
    sites/<site-name>/private/backups/<timestamp>-<site-name>-site_config_backup.json \
    <remote-user>@<remote-host>:<remote-bench-dir>/sites/<remote-site-name>/private/backups/
```

Then, on the remote host, run the restore command above with `<remote-site-name>`.

---

## Qdrant Backup & Restore

The `contract` collection is Qdrant's own copy of every chunk + embedding. It's regenerable via
`POST /ingest/full` against ERPNext, but that re-embeds everything through OpenAI again (time + cost),
so snapshotting it is worthwhile the same way ERPNext's bench backup is. Qdrant's snapshot API does
this — run against the container's exposed port (`localhost:6333` for local dev; loopback-only on a VPS,
so run these from the host or over an SSH tunnel).

> **Note:** snapshots are written to `/qdrant/snapshots` inside the container, which is **not** one of
> the named volumes in `docker-compose.yml` (only `/qdrant/storage` is, via `qdrant_data`). A snapshot
> survives container restarts but is lost if the container is removed/recreated — copy it out with
> `docker cp` right after creating it, same as below.

**Backup:**

```bash
# Create the snapshot (server-side, inside the qdrant container)
curl -X POST http://localhost:6333/collections/contract/snapshots

# Response includes "name": "<snapshot-file>.snapshot" — copy it (and its checksum) out of the container
docker cp <qdrant-container>:/qdrant/snapshots/contract/<snapshot-file>.snapshot ./
docker cp <qdrant-container>:/qdrant/snapshots/contract/<snapshot-file>.snapshot.checksum ./

# Verify integrity before trusting the copy
shasum -a 256 <snapshot-file>.snapshot   # compare against the .checksum file's contents
```

**Restore** — ⚠️ overwrites the `contract` collection entirely:

```bash
curl -X PUT http://localhost:6333/collections/contract/snapshots/upload \
  -F "snapshot=@<snapshot-file>.snapshot"
```

**Copying to a different machine** (e.g. local backup → VPS): `scp` the `.snapshot` (and `.checksum`)
file to the target host, then run the restore `curl` above against that host's Qdrant port. The API
side's `hybrid_search`'s in-memory BM25 index is rebuilt automatically from Qdrant at API startup
(see `api/main.py`'s lifespan hook), so no separate BM25 backup is needed.

---

## Langfuse Postgres Backup & Restore

The `postgres` service (`pg_data` volume) is Langfuse's own database — traces, observations, scores,
project config. Unlike ERPNext and Qdrant, this data is **not regenerable**: it's a running history of
past queries, not something a re-ingest or re-query rebuilds. Losing it only affects observability
(no functional impact on `/query` itself), so it's the lowest-priority of the three, but still worth a
periodic `pg_dump` if you care about historical traces surviving a migration.

**Backup** (custom format — compressed, and the only format `pg_restore` accepts for selective/parallel
restore):

```bash
docker exec <postgres-container> pg_dump -U langfuse -d langfuse -F c -f /tmp/langfuse_dump.pgdump
docker cp <postgres-container>:/tmp/langfuse_dump.pgdump ./langfuse_<timestamp>.pgdump
docker exec <postgres-container> rm /tmp/langfuse_dump.pgdump
```

**Verify** the dump is readable before trusting it (lists the archive's table of contents without
restoring anything):

```bash
docker cp ./langfuse_<timestamp>.pgdump <postgres-container>:/tmp/verify.pgdump
docker exec <postgres-container> pg_restore --list /tmp/verify.pgdump
docker exec <postgres-container> rm /tmp/verify.pgdump
```

**Restore** — ⚠️ run against an empty/fresh `langfuse` database, or expect conflicts on existing objects:

```bash
docker cp ./langfuse_<timestamp>.pgdump <postgres-container>:/tmp/restore.pgdump
docker exec <postgres-container> pg_restore -U langfuse -d langfuse --clean --if-exists /tmp/restore.pgdump
docker exec <postgres-container> rm /tmp/restore.pgdump
```

**Copying to a different machine**: `scp` the `.pgdump` file to the target host, then run the restore
steps above against that host's `postgres` container. Credentials (`langfuse`/`langfuse` by default —
see the security issue about hardcoding these, #65) come from `docker-compose.yml`'s `POSTGRES_USER`/
`POSTGRES_PASSWORD`, not `.env`.

---

## `.env` — the one thing that blocks every restore above

`.env` is gitignored by design (see `ingestion/erpnext_client.py`, `pipeline/query_pipeline.py`, etc. —
all secrets are read from environment variables, never committed). It holds `OPENAI_API_KEY`,
`ERPNEXT_API_KEY`/`ERPNEXT_API_SECRET`, `JWT_SECRET`, `ADMIN_SECRET`, `WEBHOOK_SECRET`, and the OAuth
client id/secret. Without it, neither an ERPNext restore nor a Qdrant restore is usable on a new
box — nothing in the stack can start.

- **Do not** back it up as a plain file copy sitting in a repo, shared drive, or unencrypted archive.
- Store it in a password manager or secrets vault (1Password, Bitwarden, `pass`, etc.) as a single
  secure note, or re-derive each value at deploy time from its source of truth (ERPNext's own API key
  admin page, a freshly generated random secret for `JWT_SECRET`/`ADMIN_SECRET`/`WEBHOOK_SECRET`, the
  OpenAI/ERPNext OAuth app dashboards) rather than copying the file around.
- If you do keep an encrypted copy, treat rotating any one of these secrets as invalidating that copy —
  update the vault entry at rotation time (see `docs/DEPLOYMENT.md`'s "`.env` changes vs. code changes"
  note under "Ongoing ops" for how a rotated value gets picked up by the running containers).

---

## What was built — Option B (implemented, PR #32)

Option B was chosen and merged. The table below maps the delivered modules to `IMPLEMENTATION_PLAN.md` steps:

| Step | Module | What was added |
|---|---|---|
| **Step 12** (FastAPI) | `api/auth/` + `api/routers/auth.py` | OAuth2 authorize URL, PKCE, token exchange, role fetch, JWT mint/decode, `require_allowed_role` dependency wired onto `POST /query` |
| **Step 13** (Streamlit) | `frontend/auth_ui.py` | Login page, OAuth2 redirect, JWT storage in `st.session_state`; main chat UI gated behind session JWT |
| **Step 15** (Docker Compose) | `docker-compose.yml` | Services on `rag_internal` network; nginx as sole public gateway |

Option A (Frappe custom app) was not implemented — use the Option A section above if you later need native ERPNext desk integration.

---

## AWS Deployment (Option B, single EC2 instance)

This section covers running the existing `docker-compose.yml` topology (nginx as sole public ingress,
everything else on the internal `rag_internal` network) on a single AWS EC2 instance. It assumes ERPNext
runs on a separate/existing server (self-hosted bench or Frappe Cloud) reachable over the internet — not
on the same instance or VPC.

For a more "cloud-native" setup (ECS/Fargate, RDS, ALB), the services would need to be split apart
individually; that is a larger lift than a single `docker compose up -d` and is not covered here.

### 1. Provision the instance

- **AMI**: Ubuntu 22.04 LTS or Amazon Linux 2023.
- **Size**: `t3.large` (2 vCPU / 8GB) minimum — Qdrant, Langfuse+Postgres, FastAPI, Streamlit, nginx, and
  the reranker's cross-encoder model all run on one box.
- **Storage**: 30GB gp3 (Qdrant vectors + Postgres + Docker images).
- **Security group**:
  - Inbound: 22 (SSH, restrict to your IP), 80, 443 (0.0.0.0/0)
  - Outbound: all (needed to reach ERPNext, OpenAI, and for OAuth/webhook round-trips)
  - Do **not** expose 6333 (Qdrant), 3000 (Langfuse), or 5432 (Postgres) — they must stay on the internal
    `rag_internal` Docker network, same as the existing compose file enforces.
- Allocate an **Elastic IP** and associate it with the instance so DNS survives reboots/replacements.

### 2. DNS (Route53)

Create two `A` records pointing at the Elastic IP:
```
contract-intelligence.<yourdomain>      → Elastic IP
api.contract-intelligence.<yourdomain>  → Elastic IP
```
Wait for propagation before running certbot — it validates ownership via HTTP-01 on port 80.

### 3. Bootstrap the box

```bash
ssh ubuntu@<elastic-ip>
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER   # log out/in after this
sudo apt install -y docker-compose-plugin certbot
```

### 4. Get the code + secrets onto the box

```bash
git clone <your-repo-url> contract-intelligence && cd contract-intelligence
cp .env.example .env
```

Fill in `.env` per the production checklist in the README — generate secrets with `openssl rand -hex 32`:
`WEBHOOK_SECRET`, `ADMIN_SECRET`, `JWT_SECRET`, `LANGFUSE_NEXTAUTH_SECRET`, `LANGFUSE_SALT`,
`LANGFUSE_ADMIN_PASSWORD`, and `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` (any string — they self-seed on
first Langfuse boot). Plus:

```bash
ERPNEXT_URL=https://<your-existing-erpnext-host>
ERPNEXT_API_KEY=...
ERPNEXT_API_SECRET=...
OPENAI_API_KEY=...
OAUTH_REDIRECT_URI=https://api.contract-intelligence.<yourdomain>/auth/callback
FRONTEND_URL=https://contract-intelligence.<yourdomain>
PUBLIC_API_URL=https://api.contract-intelligence.<yourdomain>
FRONTEND_DOMAIN=contract-intelligence.<yourdomain>
API_DOMAIN=api.contract-intelligence.<yourdomain>
```

`FRONTEND_DOMAIN`/`API_DOMAIN` (bare hostnames, no scheme) drive nginx's config — they're
substituted into `nginx/templates/contract-intelligence.conf.template` automatically on every container
start, so `nginx/nginx.conf` never needs manual editing and survives `git pull` cleanly.

Since ERPNext is a separate/existing server, confirm it's reachable from the EC2 box
(`curl -I $ERPNEXT_URL` from the instance) before proceeding — if it's currently only on a private
network or localhost, that's the one networking gap to close first.

### 5. TLS certs

```bash
sudo certbot certonly --standalone \
  -d contract-intelligence.<yourdomain> \
  -d api.contract-intelligence.<yourdomain>
```

### 6. ERPNext-side config (on the existing ERPNext server)

- **Integrations → OAuth Client**: update Redirect URI to
  `https://api.contract-intelligence.<yourdomain>/auth/callback`.
- **Webhook records**: update each `request_url` to
  `https://api.contract-intelligence.<yourdomain>/webhook/erpnext` (see the webhook table earlier in this
  document).

### 7. Launch

```bash
docker compose up -d
docker compose ps   # confirm all 6 services healthy
```

### 8. Trigger the initial full ingest

Webhooks only cover documents created/changed *after* they're wired up — existing ERPNext data
needs one manual full re-index:

```bash
curl -X POST https://api.contract-intelligence.<yourdomain>/ingest/full \
  -H "X-Admin-Secret: <your ADMIN_SECRET from .env>"
```
Runs as a background task; watch progress with `docker compose logs app -f` on the instance.

### 9. Verify

```bash
curl https://api.contract-intelligence.<yourdomain>/health
curl -I https://<elastic-ip>:6333   # should fail/timeout — not publicly reachable
```
Then in a browser: `https://contract-intelligence.<yourdomain>` → **Login with ERPNext** → sign in as
`Purchase Manager` → run a query and confirm citations come back. Trigger a real ERPNext webhook (e.g.
submit a PO) and confirm it re-indexes.

### 10. Ongoing ops

- Cert renewal cron (see header comment in `nginx/nginx.conf`):
  `0 3 * * * certbot renew --quiet && docker compose exec nginx nginx -s reload`
- Back up the `pg_data` and `qdrant_data` named volumes periodically — an EBS snapshot of the instance is
  the simplest option for a single-box deployment.
- Deploys: `git pull && docker compose up -d --build`.
- **`.env` changes vs. code changes** — `env_file: .env` only loads at container start, so a value
  change (e.g. rotating `OPENAI_API_KEY` or `WEBHOOK_SECRET`) needs
  `docker compose up -d --force-recreate <service>`. A **code** change needs a rebuild
  (`docker compose up -d --build <service>`) since `app`/`frontend` bake the source into the image
  via `COPY . .` in the Dockerfile — `--force-recreate` alone reuses the old image and silently keeps
  running stale code.
- Every service in `docker-compose.yml` has `restart: unless-stopped`, so the whole stack comes back
  up automatically after a host reboot or Docker daemon restart without manual intervention.

### 11. Inspecting Qdrant

Qdrant ships a built-in web dashboard (`/dashboard`) on its API port, and `docker-compose.yml` binds that
port to `127.0.0.1` on the host — reachable via SSH tunnel, same as Langfuse (§12), never from the public
internet. `curl` also isn't installed in the `app` image (`python:3.11-slim`), so the API examples below
use Python's own `urllib` from inside the container instead. Replace `contract` with your
`QDRANT_COLLECTION` value if it differs.

#### Browsing the Qdrant dashboard (SSH tunnel)

`docker-compose.yml` binds Qdrant's port 6333 to `127.0.0.1` on the host (not `0.0.0.0`) — it's still not
publicly reachable (no security group change needed, same as the "don't expose 3000/6333/5432" rule
above), but a tunnel from your laptop can reach it:

```bash
ssh -L 6333:localhost:6333 ubuntu@<elastic-ip>
```

Then browse `http://localhost:6333/dashboard` on your laptop — collections, point counts, and payload
schema are browsable with no login (Qdrant has no built-in auth unless `QDRANT_API_KEY` is set).

#### API access (no login needed)

**Collection info (point count, vector config, status):**
```bash
docker compose exec app python3 -c "import urllib.request,json; print(json.dumps(json.load(urllib.request.urlopen('http://qdrant:6333/collections/contract'))['result'], indent=2))"
```

**Just the point count** (handy to poll during/after a full ingest — see if it's climbing):
```bash
docker compose exec app python3 -c "import urllib.request,json; print(json.load(urllib.request.urlopen('http://qdrant:6333/collections/contract'))['result']['points_count'])"
```

**List all collections:**
```bash
docker compose exec app python3 -c "import urllib.request,json; print(json.dumps(json.load(urllib.request.urlopen('http://qdrant:6333/collections'))['result'], indent=2))"
```

**Sample a few points** (inspect payload shape — `source_doctype`, `docname`, `supplier`, etc.):
```bash
docker compose exec app python3 -c "import urllib.request,json; req=urllib.request.Request('http://qdrant:6333/collections/contract/points/scroll', data=json.dumps({'limit':5,'with_payload':True,'with_vector':False}).encode(), headers={'Content-Type':'application/json'}); print(json.dumps(json.load(urllib.request.urlopen(req))['result'], indent=2))"
```

**Check a specific document was indexed** (filter by `docname`):
```bash
docker compose exec app python3 -c "import urllib.request,json; req=urllib.request.Request('http://qdrant:6333/collections/contract/points/scroll', data=json.dumps({'filter':{'must':[{'key':'docname','match':{'value':'CON-2024-00042'}}]},'limit':10,'with_payload':True,'with_vector':False}).encode(), headers={'Content-Type':'application/json'}); print(json.dumps(json.load(urllib.request.urlopen(req))['result'], indent=2))"
```
Replace `CON-2024-00042` with the docname you're checking for.

**Count points by doctype** (e.g. confirm all Contracts got indexed):
```bash
docker compose exec app python3 -c "import urllib.request,json; req=urllib.request.Request('http://qdrant:6333/collections/contract/points/count', data=json.dumps({'filter':{'must':[{'key':'source_doctype','match':{'value':'Contract'}}]}}).encode(), headers={'Content-Type':'application/json'}); print(json.load(urllib.request.urlopen(req))['result'])"
```
Swap `'Contract'` for `'Terms and Conditions'`.

**Delete all points for a docname** (manual cleanup — e.g. force a clean re-index of one document;
mirrors what the webhook handler does automatically on `on_submit`/`on_update`):
```bash
docker compose exec app python3 -c "import urllib.request,json; req=urllib.request.Request('http://qdrant:6333/collections/contract/points/delete', data=json.dumps({'filter':{'must':[{'key':'docname','match':{'value':'CON-2024-00042'}}]}}).encode(), headers={'Content-Type':'application/json'}, method='POST'); print(json.load(urllib.request.urlopen(req))['result'])"
```
This is destructive for that docname's vectors — only use it if you intend to re-trigger indexing for
it afterward (e.g. re-save/re-submit the doc in ERPNext, or re-run a full ingest).

### 12. Inspecting Langfuse

Langfuse (`langfuse/langfuse:2`) stores trace data directly in the `postgres` service (not ClickHouse),
so it can be queried with plain SQL via `psql`, already present in the `postgres:16-alpine` image. Every
`/query` request creates one row in `traces`; each pipeline stage (`rewrite`, `filter_extraction`, hybrid
search, `rerank`, `generate`) is a child row in `observations` linked by `trace_id` — see
`pipeline/query_pipeline.py` for exactly what gets traced.

#### Browsing the Langfuse UI (SSH tunnel)

`docker-compose.yml` binds Langfuse's port 3000 to `127.0.0.1` on the host (not `0.0.0.0`) — it's still
not publicly reachable (no security group change needed, same as the "don't expose 3000/6333/5432"
rule above), but a tunnel from your laptop can reach it:

```bash
ssh -L 3000:localhost:3000 ubuntu@<elastic-ip>
```

Then browse `http://localhost:3000` on your laptop. Login: `admin@localhost.local` /
`<LANGFUSE_ADMIN_PASSWORD from .env>`.

This works for one operator at a time without any new public exposure. For team-wide access without a
manual tunnel per session, see the mesh VPN entry under **Future Enhancements**.

#### SQL access (no login needed)

**Recent traces** (question asked + answer returned):
```bash
docker compose exec postgres psql -U langfuse -d langfuse -c "select id, timestamp, input->>'question' as question, output->>'answer' as answer from traces order by timestamp desc limit 10;"
```

**Full span waterfall for one trace** (swap in an `id` from the query above):
```bash
docker compose exec postgres psql -U langfuse -d langfuse -c "select name, type, level, start_time, end_time, model, total_tokens, calculated_total_cost from observations where trace_id = '<trace-id>' order by start_time;"
```

**Errored spans in the last 24h** (failed generations, retrieval errors, etc.):
```bash
docker compose exec postgres psql -U langfuse -d langfuse -c "select trace_id, name, status_message, start_time from observations where level = 'ERROR' and start_time > now() - interval '24 hours' order by start_time desc;"
```

**Token/cost summary for GPT-4o generations in the last 24h:**
```bash
docker compose exec postgres psql -U langfuse -d langfuse -c "select count(*) as generations, sum(total_tokens) as total_tokens, round(sum(calculated_total_cost)::numeric, 4) as total_cost_usd from observations where type = 'GENERATION' and start_time > now() - interval '24 hours';"
```

All four verified against a real seeded Langfuse instance before writing this up. UI access is now
available via SSH tunnel (above); a mesh VPN for tunnel-free team-wide access is documented as a future
enhancement below.

---

## Future Enhancements

### PO re-indexing on post-submit edits (`on_update_after_submit`)

**Status:** Not implemented — blocked by Frappe 15 platform limitation.

**Problem:** When a buyer edits a field on an already-submitted Purchase Order (e.g. adjusting the delivery date or adding remarks), the Qdrant vector is not updated. The indexed document reflects the state at submission time.

**Root cause:** Frappe 15 does not reliably fire the `on_update_after_submit` webhook event. Tested approaches that all failed to enqueue the webhook:
- `frappe.client.save` via REST API
- `frappe.desk.form.save.savedocs` via REST API
- Save from the ERPNext desk UI

**Workaround:** Run a full re-index (`POST /ingest/full`) after bulk PO edits, or wait for the next `on_cancel`/re-submission event.

**When to revisit:** If a future Frappe release resolves this, add the `po-on-update-submitted` webhook back (it is already handled by the event-agnostic webhook handler) and reinstate the E2E integration test in `tests/test_integration.py`.

### Mesh VPN (Tailscale) for team-wide Langfuse/Qdrant access

**Status:** Not implemented — SSH tunnel (see §12) covers the immediate need.

**Problem:** SSH tunneling works for a single operator but doesn't scale to a team — everyone needs SSH
key access to the box and has to re-run `ssh -L` every session.

**Approach (future):** Install Tailscale on the EC2 host and join it to a tailnet; each teammate installs
Tailscale on their own device and joins the same tailnet. Once joined, Langfuse (and Qdrant, if useful) can
be reached directly at the host's Tailscale IP without any port-forwarding step, while port 3000/6333
remain closed to the public internet — only devices authenticated to the tailnet can reach them.

**When to revisit:** When more than one or two people regularly need Langfuse/Qdrant access, or SSH-key
management for tunneling becomes a bottleneck.
