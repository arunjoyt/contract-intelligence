"""Hybrid BM25 + vector search with Reciprocal Rank Fusion.

Architecture (from docs/ARCHITECTURE.md):
- BM25 runs over an in-memory corpus of all indexed chunk texts (no metadata
  filtering — BM25 doesn't support it natively).
- Qdrant vector search runs in parallel with any metadata filters applied.
- Both ranked lists are fused via RRF: score(d) = Σ 1/(k + rank(d)), k=60.
- The fused list is sorted descending and the top_k results are returned.

The BM25 index is rebuilt from scratch on API startup and after every webhook
upsert.  Callers pass the full list of payload dicts from
``VectorStore.get_all_texts()`` — each dict must contain at least ``text``,
``docname``, and ``chunk_index`` fields.
"""

from __future__ import annotations

from typing import Any

from rank_bm25 import BM25Okapi

from ingestion.embedder import Embedder
from retrieval.vector_store import VectorStore

_RRF_K = 60


class HybridSearch:
    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._bm25: BM25Okapi | None = None
        self._corpus_docs: list[dict] = []

    def build_bm25_index(self, docs: list[dict]) -> None:
        """Build (or rebuild) the in-memory BM25 index.

        ``docs`` is the list of payload dicts returned by
        ``VectorStore.get_all_texts()``.  Call this at startup and after
        every webhook upsert so the lexical index stays current.
        """
        self._corpus_docs = list(docs)
        if not docs:
            self._bm25 = None
            return
        tokenized = [_tokenize(doc.get("text", "")) for doc in docs]
        self._bm25 = BM25Okapi(tokenized)

    def search(
        self,
        query: str,
        filter_conditions: dict[str, Any] | None = None,
        top_k: int = 20,
    ) -> list[dict]:
        """Hybrid search: BM25 (full corpus) fused with Qdrant (filtered) via RRF.

        Returns up to ``top_k`` payload dicts sorted by descending RRF score.
        BM25 results are unfiltered by design — the Qdrant side applies
        ``filter_conditions``.
        """
        query_vector = self._embedder.embed_query(query)
        qdrant_results = self._vector_store.search(query_vector, filter_conditions, top_k=top_k)

        rrf_scores: dict[str, float] = {}
        payloads: dict[str, dict] = {}

        # --- BM25 contribution ---
        if self._bm25 is not None and self._corpus_docs:
            bm25_scores = self._bm25.get_scores(_tokenize(query))
            ranked_indices = sorted(
                range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
            )
            for rank, doc_idx in enumerate(ranked_indices[:top_k]):
                doc = self._corpus_docs[doc_idx]
                key = _chunk_key(doc)
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
                payloads[key] = doc

        # --- Qdrant contribution ---
        for rank, scored_point in enumerate(qdrant_results):
            p = scored_point.payload or {}
            key = _chunk_key(p)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
            payloads[key] = p

        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
        return [payloads[k] for k in sorted_keys[:top_k]]


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions, easy to unit-test in isolation)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _chunk_key(payload: dict) -> str:
    return f"{payload.get('docname', '')}:{payload.get('chunk_index', 0)}"
