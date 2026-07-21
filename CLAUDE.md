# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Machine-specific setup (bench path, local site URL/credentials) lives in `CLAUDE.local.md`, which is
gitignored — see that file if you need to run bench commands or hit the local ERPNext site.

## Project State

Implementation is underway. **Do not assume any step is done or pending from memory — always check GitHub
first:**

```bash
gh issue list          # open issues = pending work; closed = done
gh pr list --state all # open PRs = in-progress phases; merged = shipped
git log --oneline -20  # recent commits for context
```

`docs/IMPLEMENTATION_PLAN.md` is the **ordered** source of truth for what needs to be built (Step 0 →
Step 16). "Each step depends on the ones before it. Do not reorder." Treat `docs/ARCHITECTURE.md` as the
source of truth for data flow and schema details; treat `docs/DEPLOYMENT.md` as the source of truth for
auth/deployment when those steps come up.

---

## Branch & PR Workflow

**Every phase of remaining work must follow this workflow without exception.**

### Rules

1. **Never commit directly to `main`** for any new feature or phase work. Create a dedicated branch first.
2. **Branch naming** — `phase-<N>-<short-slug>`, e.g. `phase-3-pipeline`.
3. **One PR per phase** — open the PR against `main` as soon as the branch is pushed; keep it open
   (do not merge) until the user explicitly approves a merge.
4. **Do not merge to `main`** — create the PR and leave it. The user will review and merge.
5. Always `git push -u origin <branch>` before creating the PR.

### Cross-referencing on GitHub

- **Commit → Issue**: include `Refs #<N>` or `Closes #<N>` in the commit message body when the commit
  addresses an open issue. `Closes` auto-closes on merge; `Refs` links without closing.
- **PR → Issue**: open the PR body with `Closes #<N>` (or `Refs #<N>` if partial) so the issue appears
  in the PR sidebar.
- **PR body → Commits**: when writing the PR description, reference key commit SHAs so reviewers can
  jump to relevant diffs.
- **Issue updates**: when posting a progress comment on an issue, include the branch name and PR URL so
  the issue thread tells the full story.

### PR body template (always use this)

```
## Summary
- <bullet: what this phase implements>
- <bullet: key design decision or tradeoff>

## Steps completed
- [ ] Step N — description
- [ ] Step N+1 — description

## Closes / Refs
Closes #<issue>

## Test plan
- [ ] pytest passes with no network (mocks)
- [ ] ruff check . passes
- [ ] <phase-specific manual test>

🤖 Generated with [Claude Code](https://claude.ai/code)
```

### Roadmap tracking — issue #17

**Issue #17** (`📍 Project Roadmap`) is the single source of truth for overall progress. Keep it in sync:

- **When a phase PR is created**: post a comment on #17 linking to the PR
  (`gh issue comment 17 --body "Phase N PR: #<pr-number>"`)
- **When a step-level issue is closed**: the checkbox in #17 auto-ticks if the issue number is listed
  there — no manual edit needed. But do post a comment on #17 noting what shipped and referencing the
  commit SHA.
- **When a commit lands on a phase branch**: if it completes a step, close the corresponding step issue
  (`gh issue close <N> --comment "Completed in <sha> on branch <branch>"`); #17's checkbox updates
  automatically.
- **When a phase is fully done**: post a summary comment on #17 (e.g. "Phase 3 complete — all checkboxes
  ticked, PR #X merged") and update the phase heading in #17's body to add ✅
  (`gh issue edit 17 --body "$(gh issue view 17 --json body -q .body | sed ...)"` — or edit via the
  GitHub UI if the sed approach is fragile).

### Workflow checklist (per phase)

1. Check GitHub state: `gh issue list` + `gh pr list --state all` + `gh issue view 17`
2. `git checkout -b phase-<N>-<slug>` from latest `main`
3. Implement the step(s) for this phase
4. Commit with `Refs #<step-issue>` or `Closes #<step-issue>` in each commit message body
5. `git push -u origin <branch>`
6. `gh pr create` using the PR body template above
7. Post a comment on issue #17 linking to the new PR
8. Leave the PR open — do not merge

---

The commands to use:

```bash
# Run the API
uvicorn api.main:app --reload

# Run the frontend
streamlit run frontend/app.py

# Start infra (Qdrant + Langfuse + Postgres)
docker compose up qdrant langfuse postgres -d

# Trigger a full ingest
curl -X POST http://localhost:8000/ingest/full -H "X-Admin-Secret: <ADMIN_SECRET>"

# Lint
ruff check .

# Tests (must run with no network — OpenAI/Qdrant are mocked)
pytest tests/

# Single test
pytest tests/test_chunker.py::test_name -v

# RAGAS evaluation
python evaluation/evaluate.py
```

## Architecture

A RAG system that answers natural-language contract questions grounded in ERPNext (Frappe) data.

```
ERPNext --REST API (full ingest) / Webhooks (incremental)--> Ingestion --> Qdrant --> Retrieval --> Pipeline --> FastAPI --> Streamlit
                                                                                                         |
                                                                                                      Langfuse (tracing)
```

**Layers** (see `docs/ARCHITECTURE.md` for full diagrams):

- **Ingestion** (`ingestion/`): `erpnext_client.py` (async httpx wrapper around Frappe REST API) →
  `document_parser.py` (HTML stripping via BeautifulSoup, PDF extraction via pypdf, struct→NL serialization)
  → `chunker.py` (`RecursiveCharacterTextSplitter`, `chunk_size=512`/`chunk_overlap=64`) → `embedder.py`
  (`EMBEDDING_MODEL`, default `text-embedding-3-small`, batched). `webhook_handler.py` is the incremental
  re-indexing entry point.
- **Retrieval** (`retrieval/`): `vector_store.py` (Qdrant wrapper) + `hybrid_search.py` (BM25 in-memory
  index fused with Qdrant vector search via Reciprocal Rank Fusion, `k=60`) + `reranker.py`
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`, lazy singleton, loaded once at startup).
- **Pipeline** (`pipeline/`): `query_rewriter.py` (HyDE or step-back rewriting, controlled by
  `QUERY_REWRITE_STRATEGY`) → `query_pipeline.py` (orchestrates rewrite → metadata filter extraction →
  hybrid search top-20 → rerank top-5 → `OPENAI_MODEL` generation, default `gpt-4o`, with required source
  citations). Every step is a Langfuse child span. `OPENAI_MODEL`/`EMBEDDING_MODEL` are centralized in
  `config.py` (see `docs/ARCHITECTURE.md` § Model Configuration).
- **API** (`api/main.py`): `POST /query`, `POST /webhook/erpnext`, `POST /ingest/full` (background task,
  `X-Admin-Secret`-gated), `GET /health`. Startup hook ensures the Qdrant collection exists, rebuilds the
  BM25 index from Qdrant, and warms the reranker.
- **Frontend** (`frontend/app.py`): Streamlit chat UI with supplier/doctype/date/status filters in the
  sidebar; calls the API over `httpx`.

### Indexing strategy

Both ingested doctypes (Contract, Terms and Conditions, + Contract's attached PDFs) are unstructured
text: HTML-stripped, then split via `RecursiveCharacterTextSplitter`; each chunk carries
`chunk_index`/`total_chunks` for reconstruction.

### Qdrant payload & idempotency

Point ID is deterministic: `uuid5(NAMESPACE_DNS, f"{docname}:{chunk_index}")`. This makes upserts idempotent
— re-ingesting a document overwrites its existing points instead of duplicating them. Payload includes
`source_doctype`, `docname`, `supplier`, `supplier_group`, `start_date`, `end_date`, `status`, `company`,
`chunk_index`, `total_chunks`.

### Incremental indexing (webhooks)

ERPNext fires webhooks on `on_submit`/`on_update`/`on_cancel` for `Contract` and `Terms and Conditions`.
The handler: verify `X-Frappe-Webhook-Signature`
(HMAC-SHA256) → fetch full doc via `ERPNextClient` → `delete_by_docname` → re-run parse → chunk → embed →
upsert → rebuild BM25 index. No full re-index is needed for routine updates.

### Query pipeline order

1. `QueryRewriter.rewrite()` — HyDE (default) embeds a hypothetical answer document instead of the raw
   query, improving recall for abstract questions; step-back rewrites the question at a higher abstraction
   level instead.
2. Metadata filter extraction (pure keyword heuristic — doctype/status only, no LLM call, no date-range
   parsing) from the original question.
3. `HybridSearch.search()` — BM25 + Qdrant vector search in parallel, fused via RRF, top-20.
4. `Reranker.rerank()` — cross-encoder scores all 20 `(query, chunk)` pairs, returns top-5.
5. GPT-4o generation with a system prompt that requires `[docname]`-style citations per claim and forbids
   answering outside the provided context.

### Security invariants

- ERPNext API key/secret stay server-side; the webhook endpoint is the only externally reachable path and
  it's HMAC-protected.
- `/ingest/full` requires `X-Admin-Secret`.
- Qdrant filter expressions are built programmatically — never interpolate raw user input into them.

## Deployment

Two access-control options are documented in `docs/DEPLOYMENT.md` — pick based on ERPNext hosting:

- **Option A (Frappe custom app)**: native ERPNext desk integration; a `@frappe.whitelist()` method checks
  `frappe.get_roles()` and calls the FastAPI backend (bound to `127.0.0.1:8000`, loopback-only) using a
  short-lived HMAC token. No separate login. Requires bench access (not available on ERPNext SaaS).
- **Option B (standalone + OAuth2/JWT)**: separate app, "Login with ERPNext" via OAuth2 Authorization Code
  + PKCE; FastAPI exchanges the code, fetches roles, mints an 8-hour JWT. Works with any ERPNext hosting,
  no custom app install. Role changes apply at next login only.

Allowed roles in both options: `Purchase Manager`, `Purchase User`, `Accounts User`, `System Manager`.

`docs/DEPLOYMENT.md`'s "What was built — Option B" table maps these auth additions onto
`IMPLEMENTATION_PLAN.md` Steps 12/13/15 — consult it before modifying those steps.

## CI

GitHub Actions (`.github/workflows/ci.yml`): `ruff check .` and `pytest tests/` on every
push (tests run with no network — OpenAI and Qdrant are mocked); RAGAS evaluation runs only on merge to
`main`, with `evaluation/results.json` uploaded as an artifact.
