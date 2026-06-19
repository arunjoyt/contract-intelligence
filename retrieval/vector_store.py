"""Qdrant wrapper for the retrieval layer.

Chunks are stored with a flat payload — every metadata field sits at the top
level alongside `text` so Qdrant filter expressions can reference them directly
(e.g. ``FieldCondition(key="supplier", ...)``).

Point IDs are deterministic: ``uuid5(NAMESPACE_DNS, "{docname}:{chunk_index}")``,
making upserts idempotent — re-ingesting a document overwrites existing points
instead of duplicating them.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

VECTOR_DIM = 1536  # text-embedding-3-small


class VectorStore:
    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection: str | None = None,
    ) -> None:
        self._url = url or os.environ["QDRANT_URL"]
        self._api_key = api_key or os.environ.get("QDRANT_API_KEY") or None
        self._collection = collection or os.environ["QDRANT_COLLECTION"]
        self._client = QdrantClient(url=self._url, api_key=self._api_key)

    def ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not already exist."""
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )

    def upsert_chunks(self, chunks: list[dict]) -> None:
        """Upsert enriched chunk dicts into Qdrant.

        Each dict must contain:
        - ``vector``: list[float] — embedding produced by Embedder
        - ``docname``: str — used for deterministic point ID
        - ``chunk_index``: int — used for deterministic point ID
        - ``text`` + any metadata fields — stored as flat payload
        """
        if not chunks:
            return

        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{c['docname']}:{c['chunk_index']}")),
                vector=c["vector"],
                payload={k: v for k, v in c.items() if k != "vector"},
            )
            for c in chunks
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    def search(
        self,
        query_vector: list[float],
        filter_conditions: dict[str, Any] | None = None,
        top_k: int = 20,
    ) -> list[ScoredPoint]:
        """Vector search with optional metadata filters.

        ``filter_conditions`` is a flat dict of ``{payload_field: value}`` pairs;
        all non-None values are ANDed together as ``must`` conditions.
        """
        qdrant_filter = self._build_filter(filter_conditions) if filter_conditions else None
        return self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=top_k,
        )

    def delete_by_docname(self, docname: str) -> None:
        """Delete all points belonging to ``docname``."""
        self._client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="docname", match=MatchValue(value=docname))]
                )
            ),
        )

    def get_all_texts(self) -> list[dict]:
        """Return every point's payload — used to rebuild the BM25 index on startup."""
        results: list[dict] = []
        records, next_offset = self._client.scroll(
            collection_name=self._collection,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        results.extend(r.payload for r in records if r.payload)
        while next_offset is not None:
            records, next_offset = self._client.scroll(
                collection_name=self._collection,
                offset=next_offset,
                limit=1000,
                with_payload=True,
                with_vectors=False,
            )
            results.extend(r.payload for r in records if r.payload)
        return results

    def _build_filter(self, conditions: dict[str, Any]) -> Filter | None:
        must = [
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in conditions.items()
            if value is not None
        ]
        return Filter(must=must) if must else None
