"""Central model configuration, env-var driven.

Mirrors the QUERY_REWRITE_STRATEGY pattern used elsewhere in the pipeline: model
choice lives here instead of being a string literal scattered across
ingestion/pipeline/evaluation call sites, so swapping a model is a one-line env
var change.

Swapping EMBEDDING_MODEL to a model with a different vector dimension requires
recreating the Qdrant collection and running a full re-ingest -- the collection's
vector size is fixed at creation time (see retrieval.vector_store.VectorStore.
ensure_collection) and existing points aren't re-embedded automatically.
"""

from __future__ import annotations

import os

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

# Only models whose dimension has been verified are listed here. Add an entry
# before pointing EMBEDDING_MODEL at a new model.
_EMBEDDING_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def embedding_dimension(model: str = EMBEDDING_MODEL) -> int:
    """Return the Qdrant vector size for an embedding model."""
    try:
        return _EMBEDDING_DIMENSIONS[model]
    except KeyError:
        raise ValueError(
            f"Unknown vector dimension for embedding model {model!r} -- add it to "
            "_EMBEDDING_DIMENSIONS in config.py. Note: changing EMBEDDING_MODEL to a "
            "model with a different dimension requires recreating the Qdrant "
            "collection and running a full re-ingest."
        ) from None
