# ADR 0001: Qdrant + in-memory BM25 over Elasticsearch/OpenSearch

## Status

Accepted

## Context

The retrieval layer needs both semantic (vector) search and lexical (keyword) search, fused via
Reciprocal Rank Fusion (see `docs/ARCHITECTURE.md` § Hybrid search). Elasticsearch and OpenSearch are
the default reach for lexical search and offer mature vector/kNN plugins as well, so they were the
natural alternative to evaluate against Qdrant + a lightweight BM25 index.

Relevant characteristics of this project:

- The indexed corpus is ERPNext `Contract` and `Terms and Conditions` documents plus attached PDFs —
  dozens to low-thousands of documents (demo baseline: 31 Contracts, 9 T&C docs), not log-scale or
  web-scale text.
- The primary retrieval signal is embedding similarity: `QueryRewriter` (HyDE/step-back) plus
  `HybridSearch` exist specifically to improve semantic recall for natural-language contract questions.
  Lexical/BM25 matching is a secondary, precision-boosting signal, not the main retrieval path.
- This is a single-tenant internal tool (one FastAPI service, one ERPNext instance, one Streamlit UI) —
  not a multi-tenant search product.
- Operational surface matters: the team already runs Qdrant, Postgres, and Langfuse via
  `docker compose`, and backs them up with `scripts/backup_all.sh`. Every additional stateful service is
  another thing to deploy, upgrade, and restore.

## Decision

Use Qdrant as the single retrieval datastore. Lexical search is served by a small **in-memory BM25
index**, built at API startup from payload text fetched out of Qdrant and rebuilt after each webhook
upsert (`docs/ARCHITECTURE.md` line ~199), fused with Qdrant's vector search results via RRF (`k=60`) in
`retrieval/hybrid_search.py`. No Elasticsearch/OpenSearch cluster is deployed.

## Rationale

- **Scale doesn't demand a search cluster.** ES/OpenSearch are designed for sharded, replicated,
  log-scale full-text workloads. At a few hundred to a few thousand documents, an in-memory BM25 index
  rebuilds in milliseconds — a distributed search engine would add JVM heap tuning, cluster health
  monitoring, and upgrade overhead with no retrieval-quality payoff at this size.
- **Vector search is the primary need, not lexical.** Qdrant is vector-native: HNSW indexing, payload
  filtering, and collection management are first-class. ES/OpenSearch added kNN/vector search as a
  later plugin layer on top of a lexical-search core — serviceable, but secondary to their design
  center. That priority is inverted from what this project needs.
- **BM25 here is cheap and doesn't need a dedicated engine.** It's corpus-wide (no metadata filtering,
  see `docs/ARCHITECTURE.md` § hybrid search filter semantics) and rebuilt from data Qdrant already
  stores. Standing up a second distributed datastore to compute term-frequency scores over a few hundred
  documents would be disproportionate to the problem.
- **Fewer moving parts.** One datastore (Qdrant) instead of two (vector store + search cluster) means
  one thing to provision, secure, and back up, which matters for a single-tenant internal tool with a
  small ops surface.

## Consequences

- BM25 matching is corpus-wide and does not support the supplier/doctype/status metadata filters that
  the Qdrant vector-search leg supports — a documented limitation of the fused hybrid search
  (`docs/ARCHITECTURE.md` § hybrid search).
- The in-memory BM25 index is rebuilt on every API startup and after every webhook upsert; this scales
  linearly with corpus size and was flagged in `docs/ARCHITECTURE.md` as needing revisiting past ~100k
  documents.
- If the corpus grows enough that in-memory BM25 becomes a bottleneck, the documented next step is to
  move lexical search into **Qdrant's own sparse-vector support** (sparse vectors + an `idf` modifier,
  combined with Qdrant's native RRF fusion via `query_points`/`prefetch`) rather than introducing
  Elasticsearch/OpenSearch — keeping a single datastore even at larger scale. Tracked in #97.
- If a future requirement needs advanced full-text features Qdrant doesn't offer (e.g. fuzzy matching,
  language-specific analyzers, faceted search UI), this decision should be revisited.
