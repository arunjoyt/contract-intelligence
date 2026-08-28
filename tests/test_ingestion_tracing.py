"""Tests for ingestion.tracing helpers. No network — Langfuse is a MagicMock."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ingestion.tracing import span, traced_embed

# ---------------------------------------------------------------------------
# span
# ---------------------------------------------------------------------------


def test_span_noops_when_trace_is_none() -> None:
    with span(None, "parse") as s:
        assert s is None


def test_span_ends_on_normal_exit() -> None:
    trace = MagicMock()
    child = trace.span.return_value

    with span(trace, "parse") as s:
        assert s is child

    trace.span.assert_called_once_with(name="parse")
    child.end.assert_called_once_with()


def test_span_ends_with_error_level_and_reraises_on_exception() -> None:
    trace = MagicMock()
    child = trace.span.return_value

    with pytest.raises(RuntimeError), span(trace, "upsert"):
        raise RuntimeError("Qdrant down")

    child.end.assert_called_once_with(level="ERROR")


# ---------------------------------------------------------------------------
# traced_embed
# ---------------------------------------------------------------------------


def test_traced_embed_without_trace_calls_plain_embed_texts() -> None:
    embedder = MagicMock()
    embedder.embed_texts.return_value = [[0.1], [0.2]]

    result = traced_embed(None, embedder, ["a", "b"])

    assert result == [[0.1], [0.2]]
    embedder.embed_texts.assert_called_once_with(["a", "b"])
    embedder.embed_texts_with_usage.assert_not_called()


def test_traced_embed_records_generation_with_model_and_usage() -> None:
    trace = MagicMock()
    gen = trace.generation.return_value
    embedder = MagicMock()
    embedder.embed_texts_with_usage.return_value = ([[0.1], [0.2]], {"input": 30, "total": 30})

    result = traced_embed(trace, embedder, ["a", "b"])

    assert result == [[0.1], [0.2]]
    trace.generation.assert_called_once_with(name="embed", model="text-embedding-3-small")
    gen.end.assert_called_once_with(usage={"input": 30, "total": 30})


def test_traced_embed_marks_generation_error_and_reraises() -> None:
    trace = MagicMock()
    gen = trace.generation.return_value
    embedder = MagicMock()
    embedder.embed_texts_with_usage.side_effect = RuntimeError("OpenAI 500")

    with pytest.raises(RuntimeError):
        traced_embed(trace, embedder, ["a"])

    gen.end.assert_called_once_with(level="ERROR")
