"""Cross-encoder reranker for the retrieval layer.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 via sentence-transformers.

The CrossEncoder is loaded lazily on the first ``rerank()`` call and cached as
an instance attribute thereafter — it is never reloaded. ``warm_up()`` forces
that load at API startup (Step 12) so the first live query has no cold-start
delay. Candidates are payload dicts from HybridSearch; the ``text`` field of
each dict is what gets scored.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL) -> None:
        self._model_name = model_name
        self._model: CrossEncoder | None = None

    def warm_up(self) -> None:
        """Force model loading — call once at API startup to avoid a cold first query."""
        self._load_model()

    def rerank(self, query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
        """Score every (query, candidate_text) pair and return the top_n by score.

        Fewer than ``top_n`` candidates → all candidates returned, sorted by score.
        Empty candidates → empty list returned without loading the model.
        """
        if not candidates:
            return []

        model = self._load_model()
        pairs = [(query, c.get("text", "")) for c in candidates]
        scores = model.predict(pairs)

        ranked = sorted(zip(candidates, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:top_n]]

    def _load_model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self._model_name)
        return self._model
