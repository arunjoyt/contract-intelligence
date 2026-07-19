# Descope to Contract Intelligence — refactor plan

> **Status: planning only, not started.** This document records a plan to be executed later. No
> branch, commit, or code change has been made from it yet.

## Context

Two prior discussions led here. First: whether to reframe the project's positioning around
"Contract Intelligence" (contracts are a strong RAG fit; aggregation queries are not). Second:
whether the known aggregation limitation (issue #45, closed won't-fix in PR #59 — the fixed
top-20/rerank-top-5 pipeline silently drops records on enumeration/sum queries) should be fixed now
via a router to live ERPNext data, or deferred.

**Decision made:** descope the project to **Contract Intelligence only**. Drop Purchase Order,
Purchase Invoice, and Supplier Scorecard entirely — keep Contract + Terms and Conditions (+ their
PDF attachments) as the sole ingested doctypes. This sidesteps the aggregation problem by removing
its only real use case (PO/Invoice totals and counts) rather than fixing it; aggregation is deferred
to a future enhancement or separate project, not designed here. The goal is a narrower, "foolproof"
system for a portfolio demo: semantic search over contract/clause language, where RAG has no
structured-query weak spot left to explain around.

**Refactor vs. new project:** refactor the existing repo. It's small (~7,150 lines of Python
excluding `.venv`) and cleanly factored — every PO/Invoice/Scorecard reference is an isolated
dispatch-list entry or a small dedicated function (`po_to_text`, `invoice_to_text`,
`supplier_scorecard_to_text`, each ~10-15 lines), not logic tangled through the core pipeline. The
entire retrieval/reranking/query-orchestration/auth/deployment/CI/eval-harness stack — the bulk of
the actual engineering — is 100% doctype-agnostic and untouched by this change. Starting fresh would
mean re-building all of that for zero benefit.

## Scope

**In scope:** remove PO/Invoice/Scorecard from ingestion, webhook handling, full-ingest, query
pipeline filtering, HyDE prompt, frontend filters, tests, evaluation dataset, and docs.

**Explicitly out of scope (do not bundle in):**
- Aggregation fix/router — deferred per the earlier decision.
- Clause-level chunking, risk flagging, clause comparison, expiry alerting — these were floated in
  the earlier "Contract Intelligence" framing discussion but not requested yet; natural follow-up,
  not part of this change.
- Two pre-existing bugs found on lines this change touches anyway (a dead `"Supplier"` keyword in
  `_DOCTYPE_KEYWORDS` that zeroes out vector search on any question mentioning "supplier"; a missing
  `"Unsigned"` status keyword/dropdown option despite the system prompt already treating it as a real
  value) — **decision: keep this descope strictly mechanical.** File two separate GitHub issues for
  these instead of fixing them here.

## Staging: 3 PRs, not 1

The diff is large and heterogeneous — ~6 source modules with real logic changes, ~10 test files (one,
`test_webhook_handler.py`, needs ~200+ lines rewritten because generic behavior tests currently
piggyback on the PO fixture), a 1914-line integration test file needing surgical edits across 9
groups, a 15-entry eval dataset needing 9 entries replaced, and 4 docs files including operationally
real content (webhook registration tables/scripts in `DEPLOYMENT.md`, not just prose). Splitting keeps
each PR's `ruff check .` + `pytest tests/` gate green independently, keeps review sizes sane, and
matches this repo's existing convention of a dedicated docs-only cleanup PR (see `93a87f4` / PR #62).

1. **`phase-descope-1-ingestion-schema`** — ingestion, webhook handling, full-ingest dispatch, their
   unit tests.
2. **`phase-descope-2-pipeline-api-frontend`** — query pipeline keywords, HyDE prompt, frontend
   filters, their unit tests, `test_integration.py` cleanup.
3. **`phase-descope-3-eval-docs`** — eval dataset, all docs, `CLAUDE.md`.

Stack sequentially (each branches from the previous once merged), consistent with this repo's
one-PR-per-phase workflow. PR 1 and PR 3 have minimal file overlap and could go up in parallel if
preferred; PR 2 depends on PR 1.

**Step 0 (before branching):** file a tracking issue — "Descope to Contract Intelligence: drop
Purchase Order / Purchase Invoice / Supplier Scorecard ingestion" — referencing roadmap issue #17.
Use its number as `Refs #<N>` in all three PRs' commits, per `CLAUDE.md`'s workflow rules. Also file
two standalone follow-up issues for the "Supplier" keyword bug and missing "Unsigned" status keyword
(found during design, deliberately not fixed here).

---

## PR 1 — Ingestion & schema

**`ingestion/document_parser.py`** — delete `po_to_text`, `invoice_to_text`,
`supplier_scorecard_to_text`, and their private helpers `_criteria_summary`, `_item_names`, `_fmt`
(all become dead code once the three serializers go — `_fmt` has no other caller). Rewrite the module
docstring (lines 1-14), which currently frames "two jobs" (structured serialization + HTML/PDF
extraction) — only the extraction job remains.

**`ingestion/webhook_handler.py`**:
- `SUPPORTED_DOCTYPES` → `frozenset({"Contract", "Terms and Conditions"})`.
- `ATTACHMENT_DOCTYPES` → `frozenset({"Contract"})` (was `{"Purchase Order", "Contract"}`).
- Delete the `Purchase Order`/`Purchase Invoice`/`Supplier Scorecard` branches in
  `prepare_doc_for_indexing` (~lines 251, 291, 326), leaving Contract + Terms and Conditions.
- Delete the `Purchase Invoice` special case in `resolve_supplier_group` (~line 182); Terms and
  Conditions has no supplier field so the generic `_fetch_supplier_group` lookup naturally no-ops for
  it — no new branch needed.
- Drop now-unused imports (`po_to_text`, `invoice_to_text`, `supplier_scorecard_to_text`).
- Rewrite docstrings referencing PO/Invoice/Scorecard (module docstring, `ATTACHMENT_DOCTYPES`
  comment, `resolve_supplier_group`/`prepare_doc_for_indexing`/`gather_chunks_for_doc` docstrings).

**`ingestion/chunker.py`** — docstring-only rewrite (lines 1-10); no logic change (the
single-chunk/`force_single_chunk` mechanism is generic and stays, it just no longer has a PO/Invoice/
Scorecard caller).

**`api/main.py`** — `_INGEST_DOCTYPES` → `("Contract", "Terms and Conditions")`.

**Tests:**
- `tests/test_document_parser.py` — delete the `po_to_text`/`invoice_to_text`/
  `supplier_scorecard_to_text` tests and their imports.
- `tests/test_webhook_handler.py` (719 lines, the largest test surface here) — rewrite the
  **generic-behavior** tests (signature verification, delete-before-upsert ordering,
  rebuild-BM25-after-upsert, name-vs-docname fallback, upsert-failure-propagates,
  supplier-group-lookup-failure) to use Contract fixtures instead of PO fixtures — these test
  doctype-agnostic behavior and must not be deleted, just re-pointed. Delete the PO-specific,
  Scorecard-specific, and Invoice-specific test blocks outright. Fold the PO-attachment tests into the
  existing Contract-attachment tests (`ATTACHMENT_DOCTYPES` is Contract-only now). Replace the
  Scorecard fixture in `test_attachments_not_fetched_for_non_attachment_doctypes` with Terms and
  Conditions (still a valid non-attachment doctype).
- `tests/test_chunker.py` — cosmetic-only rename, optional.

Verify: `ruff check .` and `pytest tests/` green before pushing.

---

## PR 2 — Pipeline, API, frontend

**`pipeline/query_pipeline.py`** — `_DOCTYPE_KEYWORDS`: drop `"Purchase Order"`, `"Purchase Invoice"`,
`"Supplier Scorecard"`, and `"Supplier"` entries, keeping only `"Contract"` and
`"Terms and Conditions"`. (Note: dropping the `"Supplier"` entry here is *required* by the descope
itself — Supplier was never an ingested doctype — this is distinct from the separately-filed
dead-filter bug, which was about the entry existing at all; removing it as part of shrinking this
dict to two doctypes resolves it as a side effect, which is fine — the point of "descope only" was to
avoid *hunting for and fixing unrelated bugs*, not to avoid deleting a line that's being deleted
anyway.) `_STATUS_KEYWORDS`: drop `"paid"`, `"unpaid"` (Purchase-Invoice-only vocabulary) — do **not**
add `"unsigned"` (that's the separately-filed follow-up).

**`pipeline/query_rewriter.py`** — line 23, HyDE prompt: replace
`"(Purchase Order, Contract, Supplier Scorecard, or similar)"` with
`"(a Contract or Terms and Conditions document)"`.

**`frontend/app.py`** — `_DOCTYPES` sidebar list → `["Contract", "Terms and Conditions"]`. Leave the
status dropdown as-is (no "Unsigned" addition — follow-up issue instead).

**Tests:**
- `tests/test_query_pipeline.py` — delete the PO/Invoice/Scorecard-specific `_extract_filters`
  tests; rewrite the composite filter-merge tests (which currently assert on "purchase order"/
  "submitted" wording) to use contract/status wording that still resolves through the smaller
  keyword dicts.
- `tests/test_api.py`, `tests/test_vector_store.py`, `tests/test_query_rewriter.py`,
  `tests/test_erpnext_client.py` — no functional changes required (PO references here are inert mock
  strings against fully-mocked layers); cosmetic rename optional, low priority.
- `tests/test_integration.py` (1914 lines, gated by `RUN_INTEGRATION=1`, not run in CI, but **is**
  covered by `ruff check .` which has no test exclusion) — remove/rewrite PO/Invoice/Scorecard
  imports, fixtures, and webhook-group tests enough to keep lint clean and imports consistent. Runtime
  correctness of the rewritten live-ERPNext assertions can't be verified without a live (also
  descoped) ERPNext instance — mark this "static-only verified" in the PR description.

Verify: `ruff check .` and `pytest tests/` green before pushing.

---

## PR 3 — Evaluation & docs

**`evaluation/test_dataset.json`** — delete the 9 PO/Invoice/Scorecard/supplier-group entries; keep
the 6 Contract/T&C entries (payment terms, active contracts, warranty/return terms, delivery
timelines, contracts expiring in 3 months, penalty clauses). Consider adding a few new Contract/T&C
questions to restore dataset coverage — optional, not required by the descope itself.

**`evaluation/evaluate.py`** — no functional change needed (already doctype-agnostic).

**`docs/ARCHITECTURE.md`** — collapse the "two distinct indexing paths" framing (Structured vs.
Unstructured) to a single path now that Contract/T&C's HTML-strip-then-chunk approach is the only one
left; rewrite the PDF-attachments paragraph (Contract-only now); drop PO/Invoice/Scorecard rows from
the webhook events table; reword the `linked_doctype`/`linked_docname` example (currently illustrates
a Contract linking to a Purchase Order — the generic Dynamic Link mechanism from #53/PR#58 stays
functional for any doctype, only the example needs to change since PO no longer exists in the corpus).

**`docs/DEPLOYMENT.md`** — delete 5 of 8 rows from the webhook registration table and the matching
entries in the REST-API-scripted webhook setup snippet (this is real operational config, not prose).
Reword Qdrant-inspection curl examples that use `'Purchase Order'` as sample payload data. Leave the
`ALLOWED_ROLES` table/env default as-is (`Purchase Manager, Purchase User, Accounts User,
System Manager`) — it's an env var, not hardcoded, and Contract/T&C data is still
procurement-adjacent for these roles; not a required change.

**`docs/IMPLEMENTATION_PLAN.md`** — no edits (its header already frames it as a historical build
reference); add one short pointer note near the top linking to the descope tracking issue and
`docs/ARCHITECTURE.md`, consistent with how #45/#60 were handled.

**`README.md`** — delete the 3 PO/Invoice/Scorecard rows from the "Data Sources" table.

**`CLAUDE.md`** — rewrite the "Structured docs" bullet (collapse to describe only the unstructured
HTML/PDF path) and the webhook-supported-doctypes list.

Verify: `ruff check .` passes (docs/JSON-only PR, should be a no-op); if `OPENAI_API_KEY` is
available, a manual `python evaluation/evaluate.py` run confirms the trimmed dataset still scores
well — otherwise this rides on the next merge-to-`main` CI evaluation job.

---

## Verification (end-to-end, after all 3 PRs land)

1. `ruff check .` and `pytest tests/` pass with no network calls.
2. `POST /ingest/full` against the live ERPNext instance (`prag01.test`) only ingests Contract and
   Terms and Conditions records — confirm via Qdrant point count / `source_doctype` payload values
   (no `Purchase Order`/`Purchase Invoice`/`Supplier Scorecard` points remain after a fresh full
   re-ingest against a clean collection).
3. Frontend sidebar doctype filter only shows Contract/Terms and Conditions; a query mentioning
   "purchase order" no longer matches a doctype filter (falls through to unfiltered search, since that
   entry is gone) rather than silently zeroing results.
4. `python evaluation/evaluate.py` completes against the trimmed dataset and produces reasonable RAGAS
   scores (faithfulness, answer_relevancy, context_recall, context_precision) — compare against the
   last recorded `evaluation/results.json` to ensure no regression from dataset trimming alone.
5. Webhook simulation: `on_submit`/`on_update`/`on_cancel` for Contract and Terms and Conditions still
   re-index correctly; simulating the same for a Purchase Order (if still configured in a stale
   ERPNext webhook record) should be handled gracefully as an unsupported/ignored doctype, not an
   error — confirm `docs/DEPLOYMENT.md`'s webhook table has actually been updated so no stale PO
   webhook registration is left calling a descoped path in a real deployment.
