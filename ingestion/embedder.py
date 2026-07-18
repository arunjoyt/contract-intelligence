"""OpenAI embedding wrapper for the ingestion pipeline.

Model is set by EMBEDDING_MODEL (see config.py), default `text-embedding-3-small`.
`embed_texts` batches calls at OpenAI's documented limit of 2048 inputs per
request, so callers can pass arbitrarily long lists of chunk texts without
worrying about the API's batch ceiling.
"""

from __future__ import annotations

import os

from openai import OpenAI

from config import EMBEDDING_MODEL

MAX_BATCH_SIZE = 2048


class Embedder:
    def __init__(self, api_key: str | None = None, model: str = EMBEDDING_MODEL) -> None:
        self._client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self._model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, splitting into <=2048-item API calls as needed.

        Returns embeddings in the same order as `texts`.
        """
        if not texts:
            return []

        embeddings: list[list[float]] = []
        for start in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[start : start + MAX_BATCH_SIZE]
            response = self._client.embeddings.create(model=self._model, input=batch)
            embeddings.extend(item.embedding for item in response.data)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_texts([text])[0]
