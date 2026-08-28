"""Query rewriting for the retrieval pipeline.

Two strategies, controlled by the QUERY_REWRITE_STRATEGY env var:
- 'hyde' (default): Prompt the rewrite model to write a hypothetical contract
  document that would answer the question, then embed that document.
  Improves recall by searching in answer-space rather than query-space.
- 'step_back': Prompt the rewrite model to rewrite the question at a higher
  abstraction level, then embed the rewritten question.

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

    def rewrite(self, query: str) -> tuple[str, list[float]]:
        """Rewrite the query and return (rewritten_text, embedding_vector).

        The embedding is of the rewritten text; callers use it for vector search.
        Falls back to the original query if GPT-4o returns an empty response.
        """
        rewritten = self._step_back(query) if self._strategy == "step_back" else self._hyde(query)

        vector = self._embedder.embed_query(rewritten)
        return rewritten, vector

    def _hyde(self, query: str) -> str:
        response = self._client.chat.completions.create(
            model=REWRITE_MODEL,
            messages=[
                {"role": "system", "content": _HYDE_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0.7,
            max_tokens=256,
        )
        return response.choices[0].message.content or query

    def _step_back(self, query: str) -> str:
        response = self._client.chat.completions.create(
            model=REWRITE_MODEL,
            messages=[
                {"role": "system", "content": _STEP_BACK_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0.3,
            max_tokens=128,
        )
        return response.choices[0].message.content or query
