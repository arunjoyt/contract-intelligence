# Demo Data Plan

**Status:** in progress. Rewritten 2026-07-22 to make the seed script work against a genuinely fresh
ERPNext site (no pre-existing Contracts, Suppliers, or Purchase Orders assumed) rather than only
against today's `prag01.test` state.

## Context

`prag01.test`'s `Contract`/`Terms and Conditions` data has accumulated stale integration-test junk (18
placeholder `MA Inc.` contracts) and needs to be rebuilt into a clean, portfolio-ready dataset. The seed
tool must also work unmodified against a completely empty site — no records of any kind — so it's
portable across site resets and fresh environments, not a one-off cleanup tied to `prag01.test`'s
current junk. Goals:

- Support firing all of the README's example questions successfully as the first demo step.
- Decent data volume, multiple industries, PDFs, realistic prose — not templated/Lorem-ipsum.
- Cover the RAG architecture's different pieces (hybrid search, HyDE, reranking/cross-document
  synthesis, grounded refusal, PDF ingestion, status/date metadata filtering).
- End state gets backed up (`bench backup --with-files` + Qdrant snapshot) and restored into production
  for the actual demo — see `CLAUDE.local.md`'s existing backup/restore notes.

## Decisions

1. **Fully idempotent, fresh-site-capable script.** No hardcoded docnames, no assumption that any
   Contract/Supplier/Terms and Conditions/Purchase Order/Company record already exists. Running the
   script twice in a row, or against an empty site vs. a dirty one, produces the same end state.
2. **Build a reusable, committed seed script** (`scripts/seed_demo_data.py` + a hand-editable YAML
   fixture) rather than one-off manual creation.
3. **Target volume**: ~20-22 Suppliers, ~30-32 Contracts, 8-10 Terms and Conditions docs, 8-10 PDF
   attachments, spanning multiple industries (not just security/logistics/packaging).
4. **Cleanup is generic, not allowlist-based.** `--reset` deletes *every* existing `Contract` and
   `Terms and Conditions` document (docstatus-aware: submitted docs get cancelled then deleted), with no
   special-casing of specific docnames. This works whether the site has 18 junk contracts or zero.
   Suppliers are never deleted by the script (create-if-missing only) — they may be referenced by
   unrelated ERPNext data (Purchase Orders, etc.) outside this system's ingestion scope, and leaving
   them alone naturally preserves records like `MA Inc.`/`Sprint Inc` untouched.
5. **All "keeper" contracts (Zuckerman Security Ltd., Summit Traders Ltd., Alpha Supplies Ltd.) are
   regular fixture entries**, re-authored to match their current live content so the README questions
   keep working, and created fresh by the script like every other contract. Nothing is assumed
   pre-seeded. Zuckerman's currently-attached real PDF (liability cap = 12 months' fees, termination for
   cause = 15-day cure) gets its content captured into the fixture and rendered via the same reportlab
   path used for every other generated PDF.
6. **No Company bootstrapping.** Verified via `show columns from tabContract` on the live site: Contract
   has no `company` column. `ingestion/webhook_handler.py`'s `doc.get("company")` and the `company` key
   in `tests/test_integration.py`'s helper are harmless no-ops (Frappe silently ignores unknown keys on
   insert) — not something the seed script needs to satisfy.
7. **No Purchase Order linking.** The original plan's decision to link 2-3 contracts to real Purchase
   Orders via `document_type`/`document_name` is dropped — a fresh site has no Items/Warehouses either,
   and the ingestion pipeline never dereferences that Dynamic Link (it's stored as inert metadata), so a
   real PO added no functional test coverage. The fixture schema omits these fields entirely.
8. **No ERPNext Webhook record creation.** Stays a manual/documented step per `docs/DEPLOYMENT.md`'s
   existing "ERPNext Webhook Setup" section — out of scope for this script.

## README example questions the data must support (verbatim, final)

1. "What penalty applies if Zuckerman Security Ltd. exceeds two service level incidents in a quarter?" —
   covered by Zuckerman's `contract_terms` + PDF (5% invoice penalty).
2. "Which supplier bears the cost of replacing defective goods delivered under warranty?" — covered by
   Alpha Supplies Ltd.'s `contract_terms` (warranty/replacement-at-Alpha's-expense clause).
3. "Compare the payment terms across our contracts with Alpha Supplies Ltd. and Summit Traders Ltd." —
   covered (Net 30 vs Net 45).
4. "What recourse do we have if a security services vendor doesn't perform to the agreed standard?" —
   covered by Zuckerman's SLA/penalty/termination-for-cause clauses (HyDE abstraction test).
5. "Has our contract with Zuckerman Security Ltd. been signed yet?" — Zuckerman `status="Unsigned"`,
   `is_signed=0`.
6. "What's our contract value with a supplier called Globex Corp?" — **Globex Corp must never be
   created** (grounded-refusal negative test case).
7. "What is Zuckerman Security Ltd.'s liability cap under the signed contract PDF, and how quickly can
   either party terminate for cause?" — covered by the generated PDF (liability cap = 12 months' fees;
   termination for cause = 15-day cure period).
8. (bonus, from README's Langfuse-verification curl example) "What are the payment terms for our active
   contracts?" — needs at least one Contract with `status="Active"`.

## Script architecture

```
scripts/seed_demo_data.py          # entrypoint: argparse, orchestration
scripts/seed_demo_data/
    __init__.py
    erp_admin_client.py            # sync httpx wrapper with write ops (create/submit/cancel/delete/upload)
    cleanup.py                     # phase functions: contracts, terms and conditions, qdrant
    seed.py                        # phase functions: suppliers, contracts, T&Cs, PDFs
    fixtures.py                    # loads/validates the YAML fixture
    pdf_gen.py                     # reportlab rendering
scripts/seed_data/
    demo_data.yaml                 # hand-editable fixture: suppliers, contracts, T&Cs
    generated_pdfs/                # gitignored build artifacts (repo already has a blanket *.pdf rule)
```

CLI: `--reset` (cleanup only), `--seed` (idempotent upsert), `--reset --seed` (typical full run),
`--dry-run`, `--verify` (read-only post-checks). No flags → refuse and print usage (forces explicitness
around `--reset` being destructive).

**Credentials**: reuse `.env` via `python-dotenv`, same `ERPNEXT_URL`/`ERPNEXT_API_KEY`/
`ERPNEXT_API_SECRET`/`QDRANT_URL`/`QDRANT_COLLECTION`/`ADMIN_SECRET` pattern as `api/main.py` and
`ingestion/erpnext_client.py`. No hardcoded credentials despite them being visible in `CLAUDE.local.md`.

**Why a new `ERPAdminClient` instead of extending `ingestion.erpnext_client.ERPNextClient`**: that class
is deliberately read-only (`get_list`/`get_doc`/`get_attached_files`/`get_file_content` only) and is
imported by the production FastAPI app and webhook handler — bolting create/submit/cancel/delete/upload
onto it for a one-off demo tool would blur that boundary. Mirrors the existing sync-httpx pattern already
used in `tests/test_integration.py`'s `_erp_create_and_submit_contract`/`_erp_cancel_contract` helpers
(same auth header, same `frappe.client.submit`/`frappe.client.cancel` form-encoded calls) instead.

**Idempotency**:
- Terms and Conditions / Supplier: check-by-name (`name == title` / `name == supplier_name` in this
  environment), create only if missing.
- Contract: **no natural docname** to check ahead of time (`autoname = "CON-.YYYY.-.#####"`, server
  sequence) — idempotency uses a `(party_name, start_date, end_date)` natural-key filter query instead.
  Fixture authoring must keep that triple unique per entry; validated at fixture-load time.

## Cleanup phase (`--reset`)

1. List every Contract with `[name, docstatus]`. `docstatus==1` → cancel then delete; `docstatus in
   (0, 2)` → delete directly. No allowlist — everything goes.
2. List every Terms and Conditions doc and delete (not submittable, always direct delete).
3. Recreate the Qdrant collection via `VectorStore.reset_collection()` (new method — delete-if-exists
   then `ensure_collection()`).
4. Suppliers and any other doctype are never touched.

## PDF generation

- **reportlab** (already present transitively in `.venv`, added as a direct `requirements.txt`
  dependency under a new `# Demo data seeding` section, alongside `PyYAML` for the fixture loader).
- One function renders a fixture entry's structured clause content into a formatted PDF (headers,
  numbered clauses, signature block) — same content source as the `contract_terms` HTML, rendered twice.
- Output to `scripts/seed_data/generated_pdfs/` (already gitignored via the repo's blanket `*.pdf` rule).
- Attach via `POST /api/method/upload_file` (multipart: `file`, `doctype`, `docname`, `is_private=1`)
  after the Contract is created (need its assigned docname first). No extra ERPNext-side wiring needed —
  `ingestion/webhook_handler.py`'s `ATTACHMENT_DOCTYPES = frozenset({"Contract"})` + `.pdf`-suffix check
  picks it up automatically once attached.
- Idempotent: check `get_attached_files` for an existing matching filename before re-uploading.
- Attach to a **subset** (8-10 of ~30+) of contracts, not all — keeps PDF-sourced answers (README Q1, Q7)
  meaningfully distinct from HTML-`contract_terms`-sourced ones (Q2-Q5).

## Data authoring approach

**Structured YAML fixture** (`scripts/seed_data/demo_data.yaml`), not inline Python literals or JSON —
multi-line block scalars are far more readable for hand-tuned prose than escaped strings, and keeps
content diffs separate from script-logic diffs. Validated at load time (`fixtures.py`): non-null
`supplier_group` per supplier, `status` in `{Unsigned, Active, Inactive}`, unique
`(party_name, start_date, end_date)` per contract, `docstatus` in `{0, 1, 2}`.

Fixture shape:

```yaml
suppliers:
  - name: "..."
    supplier_group: "..."
terms_and_conditions:
  - title: "..."
    terms_html: |
      <p>...</p>
    disabled: false
contracts:
  - key: "human-readable-log-only-key"
    party_type: "Supplier"
    party_name: "..."
    status: "Active"
    docstatus: 1
    start_date: "..."
    end_date: "..."
    contract_terms_html: |
      <p>...</p>
    pdf_attachment: null  # or a key referencing generated PDF content
```

## Data content plan (industries / distribution)

- **Suppliers (~20-22)** across varied industries: Security Services, Raw Materials Supply, Packaging,
  SaaS/software licensing, IT hardware procurement, consulting/professional services, facilities &
  maintenance, marketing/advertising, warehousing & logistics, cleaning services, legal services,
  telecom/connectivity, insurance brokerage, recruitment/staffing, equipment leasing, printing &
  stationery, waste management, travel & events, catering/office supplies, construction/fit-out. All get
  a non-null `supplier_group`.
- **Contracts (~30-32)**: status distribution roughly Active ~14, Unsigned ~10, Inactive ~5, Cancelled
  (docstatus=2) ~3. Dates spread 2023 (expired) through 2026 (current) to 2027 (future/upcoming renewals).
  Prose per contract should read like a real clause (payment terms, SLAs, liability, termination,
  warranty/deliverables) — no templated repetition across contracts.
- **Terms and Conditions (8-10)**: e.g. Standard Purchase Terms - Net 30, Security Services -
  Confidentiality Clause, Warehousing & Logistics - Liability Terms, SaaS/Software License Terms,
  Consulting/Professional Services Terms, Marketing Services Terms, IT Hardware Warranty Terms,
  Facilities Maintenance SLA Terms, plus one `disabled=1` doc (e.g. a deprecated "Legacy Payment Terms -
  Net 60") to exercise the non-Active T&C status path.
- **Globex Corp**: never created, anywhere (decision from README Q6).

## Verification plan

1. `--verify`: Contract count/docstatus/status distribution, Supplier `supplier_group` non-null check
   for all seeded suppliers, Terms and Conditions count, File-attachment count with `.pdf` filenames.
2. `POST /ingest/full` with `X-Admin-Secret` (per README's documented curl example) — background task,
   poll rather than assume synchronous completion.
3. Qdrant point-count check (poll until stable — OpenAI embedding takes real wall-clock time); rough
   order-of-magnitude sanity check against expected chunk counts.
4. Payload spot-checks: `supplier_group` populated on sampled points, `status=="Cancelled"` for the
   docstatus=2 contract, PDF-derived chunks present for contracts with attachments (text-content sniff,
   since there's no `chunk_source` marker distinguishing PDF vs. HTML chunks in the payload schema).
5. Manually run all 8 README questions (7 table + 1 bonus) against `POST /query`, confirm
   `sources[].docname` values match expectations and Q6 (Globex Corp) returns a grounded refusal.

## Final step (once data "feels perfect")

Fresh `bench --site prag01.test backup --with-files`, fresh Qdrant snapshot via `scripts/backup_all.sh`,
and update the timestamp in `CLAUDE.local.md`'s "Demo data reset" section — per that file's own existing
instructions to do this "whenever the demo data is intentionally changed." This backup is what later gets
restored into production for the actual demo.

## Open items still needing attention during implementation

- Fixture-load-time validation should catch natural-key collisions before they silently cause the
  Contract idempotency check to skip a distinct entry.
- PDF vs. HTML chunk provenance isn't distinguishable in the current Qdrant payload schema — fine for
  this task, but worth knowing if deeper verification is ever needed.
- The 30+ contracts' prose is a genuine hand-authoring effort deferred to implementation time — this plan
  doesn't reduce that effort, just structures where it lives (the YAML fixture).
