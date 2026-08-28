"""Langfuse tracing helpers for the ingestion paths (webhook re-index + full
ingest).

These mirror the span / generation pattern in ``pipeline/query_pipeline.py`` so
an ingestion trace carries the same shape a query trace does -- per-step latency
plus a real embed cost via a ``generation`` observation. Every helper no-ops
when ``trace`` is ``None``, so ingestion runs unchanged without Langfuse
credentials (see issue #123).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from config import EMBEDDING_MODEL

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ingestion.embedder import Embedder


@contextmanager
def span(trace: Any, name: str) -> Iterator[Any]:
    """Run a block inside a Langfuse span, ending it ``ERROR`` on exception.

    Yields the span (or ``None`` when tracing is off) so the caller can attach a
    small output summary via ``s.update(output=...)``; the context manager owns
    the single ``end()`` call.
    """
    if trace is None:
        yield None
        return
    s = trace.span(name=name)
    try:
        yield s
    except Exception:
        s.end(level="ERROR")
        raise
    else:
        s.end()


def traced_embed(trace: Any, embedder: Embedder, texts: list[str]) -> list[list[float]]:
    """Embed ``texts``, recording the call as a Langfuse ``generation`` (model +
    token usage) so ingestion embedding cost is captured the same way
    ``query_pipeline``'s ``generate`` step is.

    No-ops to a plain ``embedder.embed_texts`` call when tracing is off.
    """
    if trace is None:
        return embedder.embed_texts(texts)
    gen = trace.generation(name="embed", model=EMBEDDING_MODEL)
    try:
        vectors, usage = embedder.embed_texts_with_usage(texts)
    except Exception:
        gen.end(level="ERROR")
        raise
    gen.end(usage=usage)
    return vectors
