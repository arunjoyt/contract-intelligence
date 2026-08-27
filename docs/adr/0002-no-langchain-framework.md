# ADR 0002: No LangChain (or LangGraph) as an application framework

## Status

Accepted

## Context

`langchain` entered the dependency list at Step 0 (`docs/IMPLEMENTATION_PLAN.md` line ~58) alongside
`langchain-openai` and `langchain-community`. In practice the pipeline was built against the raw
`openai` SDK and the Qdrant client directly, and the only LangChain symbol imported anywhere is
`RecursiveCharacterTextSplitter` in `ingestion/chunker.py`.

Two recurring questions prompted this ADR:

- Should we adopt LangChain's model wrappers to make the LLM provider swappable? (#90)
- Should the query pipeline move to a LangGraph `StateGraph`? (#114)

The answer to both is "no, not now" for the same underlying reason, recorded here so it does not
have to be re-argued each time.

## Decision

Do **not** adopt LangChain, LangChain Expression Language (LCEL), or LangGraph as an orchestration
or abstraction framework. Keep `RecursiveCharacterTextSplitter` (the one component genuinely in
use). Remove `langchain-openai` and `langchain-community` from `requirements.txt` — they are
declared but unused.

Where a LangChain capability is actually needed, hand-roll a minimal seam scoped to this codebase:

- **Provider abstraction** — a `ChatModel` / `EmbeddingModel` interface (two methods each) plus an
  OpenAI adapter wrapping the existing SDK calls. Roughly 40 lines, no new dependency. Specced in
  #90 and `docs/ARCHITECTURE.md` § Future Enhancement — provider-agnostic adapter. For
  OpenAI-compatible endpoints (OpenRouter, a LiteLLM proxy, vLLM, Together) an `OPENAI_BASE_URL`
  config value passed to the three `OpenAI(...)` constructors is enough on its own.
- **Stateful / branching orchestration** — revisit LangGraph if and when the pipeline needs
  runtime-dependent control flow (multi-turn memory for #34, self-corrective retrieval loops, an
  explicit "answer not in context" branch). Tracked in #114.

## Rationale

### What LangChain core provides, and what this project uses instead

| LangChain capability | Status here | Why not LangChain's version |
| --- | --- | --- |
| **Chat model wrappers** (`ChatOpenAI`, `init_chat_model`) | Direct `openai` SDK in `query_pipeline.py`, `query_rewriter.py` | Three `chat.completions.create` call sites. A provider swap is a ~40-line adapter (#90); a framework is disproportionate. |
| **Embedding wrappers** | Direct `openai` SDK in `ingestion/embedder.py` | One `embeddings.create` call site, batched at OpenAI's documented limit. Same reasoning. |
| **Prompt templates** (`ChatPromptTemplate`) | `str.format` on constants in `pipeline/constants.py` | Two static prompts with one substitution each (`context`). Templating library adds indirection, not capability. |
| **Output parsers** (Pydantic / JSON / retry) | `_parse_sources` regex over `[docname]` citations in `query_pipeline.py` | Output is free-text prose with inline citations, not structured JSON. A parser framework does not fit the shape. |
| **LCEL chains** (`prompt \| model \| parser`) | Plain Python method calls, each wrapped in a Langfuse span | LCEL's payoff is free `.batch()` / `.stream()` / `.ainvoke()`. This pipeline is a single synchronous request path; it needs none of them. LCEL would also obscure the per-step Langfuse spans. |
| **Document loaders** | Custom `ingestion/document_parser.py` (BeautifulSoup + pypdf) + `erpnext_client.py` | Source is the Frappe REST API with project-specific struct→NL serialization and attachment handling. No generic loader covers it. |
| **Text splitters** (`RecursiveCharacterTextSplitter`) | **In use** — `ingestion/chunker.py` | Genuinely useful, stable, self-contained. Kept. |
| **Vector store wrappers** (`QdrantVectorStore`, `.as_retriever()`) | Custom `retrieval/vector_store.py` | Deterministic `uuid5` point IDs for idempotent upserts, `delete_by_docname`, programmatic filter construction (security invariant). The wrapper would hide exactly the behaviour that matters. |
| **Retrievers** (`EnsembleRetriever`, `MultiQueryRetriever`, contextual compression) | Custom `retrieval/hybrid_search.py` (BM25 + vector, RRF `k=60`) + `retrieval/reranker.py` (cross-encoder) | Hand-built to control fusion weighting, filter semantics (vector leg filters, BM25 leg is corpus-wide — see ADR 0001), and the top-20 → top-5 rerank cut. |
| **Query transformation** (HyDE, step-back) | Custom `pipeline/query_rewriter.py`, selected by `QUERY_REWRITE_STRATEGY` | Two prompt calls behind a strategy switch. Nothing a framework does better. |
| **Memory** (conversation buffer / summary) | Not present | Multi-turn context is #34; when built it needs per-thread state, which is a LangGraph question (#114), not a LangChain-core one. |
| **Agents / tool calling** | Not applicable | This is a fixed retrieve-then-generate pipeline, not an agent. No dynamic tool selection. |
| **LLM response caching** | Not present | Not currently a need; if it becomes one, a small dict/Redis cache around the adapter is enough. |
| **Callbacks / tracing** (LangSmith) | Langfuse directly, hand-rolled child spans with a `summarize` hook | Observability is already built and tuned (spans trimmed to keep chunk text and embeddings out of trace storage). Routing it through LangChain callbacks would be a rewrite with regression risk and no gain. |

### Adjacent packages

- **LangGraph** — stateful graph orchestration (loops, checkpointing, conditional edges). Real
  value, but only once control flow stops being linear. See #114.
- **LangSmith** — hosted tracing / eval / prompt management. Overlaps with the existing Langfuse
  setup; no reason to run both.
- **LangServe** — serve a chain as a FastAPI route. `api/main.py` already exists.

### The general principle

Nearly every LangChain capability maps onto something this codebase has deliberately hand-built to
keep control of its internals — idempotent upserts, fusion weighting, rerank cuts, trimmed Langfuse
spans, programmatic (injection-safe) filter construction. LangChain's value is highest when moving
fast and treating those internals as a black box; this project's design center is the opposite.
Against that, the framework is net cost: a large transitive dependency graph, breaking changes
across 0.x minor releases (already pinned at `0.3.x`), and abstraction seams that hide the exact
behaviour the project cares about.

## Consequences

- `requirements.txt` drops `langchain-openai` and `langchain-community`; `langchain` stays for
  `RecursiveCharacterTextSplitter` only. (If a future LangChain-free splitter is adopted,
  `langchain` can go too.)
- Provider swaps are a deliberate, scoped code change (the #90 adapter), not a config toggle —
  acceptable because no provider swap is planned. `docs/MODEL_PROVIDER_SWAP.md` documents the
  manual path for the current code.
- New LLM-orchestration features (retrieval grading loops, multi-turn memory) trigger a fresh
  build-vs-LangGraph decision at that point (#114), rather than inheriting a framework chosen
  speculatively now.
- Contributors expecting a LangChain codebase will not find one; this ADR and
  `docs/ARCHITECTURE.md` § Model Configuration are the explanation.

## References

- ADR 0001 — retrieval datastore choice (same "fewer moving parts" reasoning)
- #90 — provider-agnostic ChatModel/EmbeddingModel seam
- #114 — decide: adopt LangGraph for query pipeline orchestration
- #34 — conversational context across turns
- `docs/ARCHITECTURE.md` § Model Configuration → Future Enhancement — provider-agnostic adapter
- `docs/MODEL_PROVIDER_SWAP.md` — manual provider-switch steps for the current code
