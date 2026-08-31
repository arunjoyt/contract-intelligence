"""Query rewriting for the retrieval pipeline.

Strategies, controlled by the QUERY_REWRITE_STRATEGY env var:
- 'hyde' (default): Prompt the rewrite model to write a hypothetical contract
  document that would answer the question, then embed that document.
  Improves recall by searching in answer-space rather than query-space.
- 'step_back': Prompt the rewrite model to rewrite the question at a higher
  abstraction level, then embed the rewritten question.
- 'none': No rewrite -- embed the raw question. No REWRITE_MODEL call. A real
  contender on this corpus (mostly point lookups with distinctive supplier
  tokens); the #113 tuning pass compares the three arms.

The chat call uses REWRITE_MODEL (config.py), not OPENAI_MODEL -- this step only
needs a scaffold paragraph to embed, and reusing gpt-4o made it dominate query
latency (issue #120).
"""

from __future__ import annotations

import os

from openai import OpenAI

from config import REWRITE_MODEL
from ingestion.embedder import Embedder

_HYDE_SYSTEM = (
    "You are a contract document generator. "
    "Given a user question, write a short hypothetical contract document "
    "(a Contract or Terms and Conditions document) that would directly "
    "answer the question. Be concise — 3–6 sentences maximum."
)

_STEP_BACK_SYSTEM = (
    "You are a contract expert. "
    "Rewrite the following question at a higher abstraction level, "
    "focusing on the contract concept rather than specific details. "
    "Return only the rewritten question, nothing else."
)


class QueryRewriter:
    def __init__(
        self,
        embedder: Embedder,
        strategy: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._embedder = embedder
        self._strategy = strategy or os.environ.get("QUERY_REWRITE_STRATEGY", "hyde")
        self._client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        # Token usage of the most recent rewrite_text() chat call, in the same
        # shape as QueryPipeline._generate's usage dict. None when no call was
        # made. Callers read it right after rewrite_text() to record the rewrite
        # step as a Langfuse `generation` with real cost (#130) -- the rewrite
        # call was otherwise invisible in per-query cost.
        self.last_usage: dict[str, int] | None = None
        # Token usage of the most recent embed() call, as ``{"input", "total"}``.
        # Read right after embed() to record the query embedding as its own
        # `generation` (#138).
        self.last_embed_usage: dict[str, int] | None = None

    @property
    def strategy(self) -> str:
        return self._strategy

    def rewrite_text(self, query: str) -> str:
        """Chat-only rewrite -- no embedding. Returns the rewritten text (or the
        original query for the ``none`` strategy / on an empty LLM response).
        Sets ``self.last_usage`` to this call's token usage (None if no LLM call).

        The embedding is a separate step (``embed``) so the pipeline can trace it
        as its own Langfuse observation rather than bundling it into the rewrite
        `generation` span (#138).
        """
        if self._strategy == "none":
            self.last_usage = None
            return query
        if self._strategy == "step_back":
            rewritten, self.last_usage = self._step_back(query)
        else:
            rewritten, self.last_usage = self._hyde(query)
        return rewritten

    def embed(self, text: str) -> list[float]:
        """Embed text for dense retrieval. Sets ``self.last_embed_usage`` to this
        call's token usage (#138)."""
        vector, self.last_embed_usage = self._embedder.embed_query_with_usage(text)
        return vector

    def rewrite(self, query: str) -> tuple[str, list[float]]:
        """Convenience wrapper: ``rewrite_text`` then ``embed`` in one call.

        Returns ``(rewritten_text, embedding_vector)``. Kept for callers that
        want both in one step (integration tests); ``QueryPipeline`` calls the
        two halves separately so it can trace them independently.
        """
        rewritten = self.rewrite_text(query)
        return rewritten, self.embed(rewritten)

    def _hyde(self, query: str) -> tuple[str, dict[str, int]]:
        response = self._client.chat.completions.create(
            model=REWRITE_MODEL,
            messages=[
                {"role": "system", "content": _HYDE_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0.7,
            max_tokens=256,
        )
        return response.choices[0].message.content or query, _usage_dict(response)

    def _step_back(self, query: str) -> tuple[str, dict[str, int]]:
        response = self._client.chat.completions.create(
            model=REWRITE_MODEL,
            messages=[
                {"role": "system", "content": _STEP_BACK_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0.3,
            max_tokens=128,
        )
        return response.choices[0].message.content or query, _usage_dict(response)


def _usage_dict(response) -> dict[str, int]:
    """OpenAI chat response → the pipeline's usage shape (matches QueryPipeline._generate)."""
    usage = response.usage
    return {
        "input": usage.prompt_tokens,
        "output": usage.completion_tokens,
        "total": usage.total_tokens,
    }
