"""Tests for pipeline.query_pipeline. No network calls — all deps mocked."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pipeline.query_pipeline import (
    QueryPipeline,
    _build_context,
    _extract_filters,
    _parse_sources,
)

FAKE_VECTOR = [0.1] * 1536


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    docname: str,
    text: str = "some chunk text",
    source_doctype: str = "Purchase Order",
    supplier: str = "Acme Corp",
    chunk_index: int = 0,
) -> dict:
    return {
        "docname": docname,
        "text": text,
        "source_doctype": source_doctype,
        "supplier": supplier,
        "chunk_index": chunk_index,
    }


def _make_openai_response(
    content: str, prompt_tokens: int = 100, completion_tokens: int = 20
) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    resp.usage.total_tokens = prompt_tokens + completion_tokens
    return resp


def _make_pipeline(
    answer: str = "The payment terms are net 30. [PO-001]",
    candidates: list[dict] | None = None,
    top_chunks: list[dict] | None = None,
    langfuse: MagicMock | None = None,
) -> QueryPipeline:
    if candidates is None:
        candidates = [_make_chunk("PO-001")]
    if top_chunks is None:
        top_chunks = [_make_chunk("PO-001")]

    mock_rewriter = MagicMock()
    mock_rewriter.rewrite.return_value = ("hypothetical doc text", FAKE_VECTOR)

    mock_search = MagicMock()
    mock_search.search.return_value = candidates

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = top_chunks

    mock_openai = MagicMock()
    mock_openai.chat.completions.create.return_value = _make_openai_response(answer)

    pipeline = QueryPipeline(
        rewriter=mock_rewriter,
        hybrid_search=mock_search,
        reranker=mock_reranker,
        api_key="test-key",
        langfuse=langfuse,
    )
    pipeline._client = mock_openai
    return pipeline


# ---------------------------------------------------------------------------
# _extract_filters
# ---------------------------------------------------------------------------


def test_extract_filters_returns_empty_for_generic_question() -> None:
    result = _extract_filters("How are procurement processes managed?")
    assert result == {}


def test_extract_filters_detects_contract() -> None:
    result = _extract_filters("What does the contract say about liability?")
    assert result.get("source_doctype") == "Contract"


def test_extract_filters_detects_terms_and_conditions() -> None:
    result = _extract_filters("What are the terms and conditions for delivery?")
    assert result.get("source_doctype") == "Terms and Conditions"


def test_extract_filters_detects_cancelled_status() -> None:
    result = _extract_filters("List cancelled contracts")
    assert result.get("status") == "Cancelled"


def test_extract_filters_detects_unsigned_status() -> None:
    result = _extract_filters("Show me unsigned contracts")
    assert result.get("status") == "Unsigned"


def test_extract_filters_detects_both_doctype_and_status() -> None:
    result = _extract_filters("Show me all cancelled contracts")
    assert result.get("source_doctype") == "Contract"
    assert result.get("status") == "Cancelled"


def test_extract_filters_is_case_insensitive() -> None:
    result = _extract_filters("Show me all CONTRACTS")
    assert result.get("source_doctype") == "Contract"


# ---------------------------------------------------------------------------
# _build_context
# ---------------------------------------------------------------------------


def test_build_context_includes_docname_and_text() -> None:
    chunks = [_make_chunk("PO-001", text="payment terms net 30", supplier="")]
    context = _build_context(chunks)
    assert "[PO-001]" in context
    assert "payment terms net 30" in context


def test_build_context_includes_supplier_when_present() -> None:
    chunks = [_make_chunk("PO-001", supplier="Acme Corp")]
    context = _build_context(chunks)
    assert "supplier: Acme Corp" in context


def test_build_context_omits_supplier_when_empty() -> None:
    chunks = [_make_chunk("PO-001", supplier="")]
    context = _build_context(chunks)
    assert "supplier:" not in context


def test_build_context_separates_chunks_with_divider() -> None:
    chunks = [_make_chunk("PO-001", text="text a"), _make_chunk("PO-002", text="text b")]
    context = _build_context(chunks)
    assert "---" in context
    assert "text a" in context
    assert "text b" in context


def test_build_context_empty_chunks_returns_empty_string() -> None:
    assert _build_context([]) == ""


def test_build_context_includes_linked_document_when_present() -> None:
    chunks = [
        {
            **_make_chunk("CON-001", source_doctype="Contract"),
            "linked_doctype": "Purchase Order",
            "linked_docname": "PO-2024-00123",
        }
    ]
    context = _build_context(chunks)
    assert "linked_doctype: Purchase Order" in context
    assert "linked_docname: PO-2024-00123" in context


# ---------------------------------------------------------------------------
# _parse_sources
# ---------------------------------------------------------------------------


def test_parse_sources_extracts_cited_docnames() -> None:
    answer = "The order was placed on net 30. [PO-001]"
    chunks = [_make_chunk("PO-001")]
    sources = _parse_sources(answer, chunks)
    assert len(sources) == 1
    assert sources[0].docname == "PO-001"


def test_parse_sources_ignores_uncited_chunks() -> None:
    answer = "Based on [PO-001] the terms are net 30."
    chunks = [_make_chunk("PO-001"), _make_chunk("CON-002")]
    sources = _parse_sources(answer, chunks)
    assert all(s.docname != "CON-002" for s in sources)


def test_parse_sources_deduplicates_same_docname() -> None:
    answer = "[PO-001] terms are net 30. [PO-001] was submitted."
    chunks = [
        _make_chunk("PO-001", chunk_index=0),
        _make_chunk("PO-001", chunk_index=1),
    ]
    sources = _parse_sources(answer, chunks)
    assert len(sources) == 1
    assert sources[0].docname == "PO-001"


def test_parse_sources_populates_source_doc_fields() -> None:
    answer = "See [PO-001] for details."
    chunk = _make_chunk(
        "PO-001", source_doctype="Purchase Order", supplier="Acme Corp", chunk_index=2
    )
    sources = _parse_sources(answer, [chunk])
    assert sources[0].source_doctype == "Purchase Order"
    assert sources[0].supplier == "Acme Corp"
    assert sources[0].chunk_index == 2


def test_parse_sources_empty_answer_returns_empty_list() -> None:
    chunks = [_make_chunk("PO-001")]
    assert _parse_sources("", chunks) == []


def test_parse_sources_no_matching_chunks_returns_empty_list() -> None:
    answer = "Based on [MISSING-001] the answer is clear."
    chunks = [_make_chunk("PO-001")]
    assert _parse_sources(answer, chunks) == []


# ---------------------------------------------------------------------------
# QueryPipeline.run — orchestration
# ---------------------------------------------------------------------------


def test_run_calls_rewriter_with_question(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    pipeline = _make_pipeline()
    pipeline.run("What are the payment terms?")
    pipeline._rewriter.rewrite.assert_called_once_with("What are the payment terms?")


def test_run_passes_rewritten_text_to_hybrid_search(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    pipeline = _make_pipeline()
    pipeline._rewriter.rewrite.return_value = ("rewritten query text", FAKE_VECTOR)

    pipeline.run("question")

    call_args = pipeline._hybrid_search.search.call_args
    assert call_args[0][0] == "rewritten query text"


def test_run_passes_top_k_20_to_hybrid_search(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    pipeline = _make_pipeline()
    pipeline.run("question")
    kwargs = pipeline._hybrid_search.search.call_args[1]
    assert kwargs.get("top_k") == 20


def test_run_passes_candidates_to_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    candidates = [_make_chunk("PO-001"), _make_chunk("PO-002")]
    pipeline = _make_pipeline(candidates=candidates)
    pipeline._hybrid_search.search.return_value = candidates

    pipeline.run("question")

    call_args = pipeline._reranker.rerank.call_args
    assert call_args[0][1] == candidates


def test_run_passes_top_n_5_to_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    pipeline = _make_pipeline()
    pipeline.run("question")
    kwargs = pipeline._reranker.rerank.call_args[1]
    assert kwargs.get("top_n") == 5


def test_run_returns_answer_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    pipeline = _make_pipeline(answer="Net 30. [PO-001]")
    result = pipeline.run("question")
    assert result["answer"] == "Net 30. [PO-001]"


def test_run_returns_sources_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    pipeline = _make_pipeline(
        answer="Net 30. [PO-001]",
        top_chunks=[_make_chunk("PO-001")],
    )
    result = pipeline.run("question")
    assert isinstance(result["sources"], list)
    assert result["sources"][0].docname == "PO-001"


def test_run_merges_extracted_and_caller_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    pipeline = _make_pipeline()

    # "contracts" triggers doctype extraction; caller adds supplier filter
    pipeline.run("Show cancelled contracts", filters={"supplier": "Acme"})

    call_args = pipeline._hybrid_search.search.call_args
    merged = call_args[0][1]
    assert merged.get("source_doctype") == "Contract"
    assert merged.get("status") == "Cancelled"
    assert merged.get("supplier") == "Acme"


def test_run_caller_filters_override_extracted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit caller filter wins on key conflict."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    pipeline = _make_pipeline()

    # "contracts" would set source_doctype=Contract, but caller overrides
    pipeline.run("Show contracts", filters={"source_doctype": "Terms and Conditions"})

    call_args = pipeline._hybrid_search.search.call_args
    merged = call_args[0][1]
    assert merged.get("source_doctype") == "Terms and Conditions"


def test_generate_returns_answer_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    pipeline = _make_pipeline()
    pipeline._client.chat.completions.create.return_value = _make_openai_response(
        "Net 30. [PO-001]", prompt_tokens=100, completion_tokens=20
    )

    answer, usage = pipeline._generate("question", "context")

    assert answer == "Net 30. [PO-001]"
    assert usage == {"input": 100, "output": 20, "total": 120}


def test_run_without_langfuse_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    pipeline = _make_pipeline(langfuse=None)
    result = pipeline.run("question")
    assert "answer" in result


def test_run_with_langfuse_creates_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mock_lf = MagicMock()
    mock_trace = MagicMock()
    mock_lf.trace.return_value = mock_trace
    mock_trace.span.return_value = MagicMock()

    pipeline = _make_pipeline(langfuse=mock_lf)
    pipeline.run("question")

    mock_lf.trace.assert_called_once()
    # filter_extraction, hybrid_search, rerank = 3 plain spans; rewrite and
    # generate are `generation`-type observations (see test_run_with_langfuse_generation_*)
    assert mock_trace.span.call_count >= 3


def test_run_with_langfuse_generation_uses_generation_not_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The `generate` and `rewrite` steps must be Langfuse `generation`s, not plain
    `span`s -- only `generation`-type observations get auto-populated token
    usage/cost, and the rewrite call (REWRITE_MODEL) was previously invisible in
    per-query cost (#130)."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mock_lf = MagicMock()
    mock_trace = MagicMock()
    mock_lf.trace.return_value = mock_trace
    mock_trace.span.return_value = MagicMock()
    mock_trace.generation.return_value = MagicMock()

    pipeline = _make_pipeline(langfuse=mock_lf)
    pipeline.run("question")

    mock_trace.generation.assert_any_call(name="generate", model="gpt-4o")
    mock_trace.generation.assert_any_call(name="rewrite", model="gpt-4o-mini")


def test_run_records_rewrite_generation_with_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rewrite `generation` span ends with the rewriter's last_usage, so the
    HyDE / step-back call's token cost reaches the Langfuse trace total (#130)."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mock_lf = MagicMock()
    mock_trace = MagicMock()
    mock_lf.trace.return_value = mock_trace
    mock_trace.span.return_value = MagicMock()
    rewrite_span = MagicMock()
    generate_span = MagicMock()
    mock_trace.generation.side_effect = [rewrite_span, generate_span]

    pipeline = _make_pipeline(langfuse=mock_lf)
    pipeline._rewriter.last_usage = {"input": 40, "output": 12, "total": 52}

    pipeline.run("question")

    assert rewrite_span.end.call_args[1]["usage"] == {"input": 40, "output": 12, "total": 52}


def test_run_with_langfuse_generation_passes_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mock_lf = MagicMock()
    mock_trace = MagicMock()
    mock_lf.trace.return_value = mock_trace
    mock_trace.span.return_value = MagicMock()
    mock_gen_span = MagicMock()
    mock_trace.generation.return_value = mock_gen_span

    pipeline = _make_pipeline(langfuse=mock_lf)
    pipeline._client.chat.completions.create.return_value = _make_openai_response(
        "Net 30. [PO-001]", prompt_tokens=150, completion_tokens=30
    )

    pipeline.run("question")

    kwargs = mock_gen_span.end.call_args[1]
    assert kwargs["usage"] == {"input": 150, "output": 30, "total": 180}


def test_run_with_langfuse_updates_trace_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mock_lf = MagicMock()
    mock_trace = MagicMock()
    mock_lf.trace.return_value = mock_trace
    mock_trace.span.return_value = MagicMock()

    pipeline = _make_pipeline(answer="The answer. [PO-001]", langfuse=mock_lf)
    pipeline.run("question")

    mock_trace.update.assert_called()
    call_kwargs = mock_trace.update.call_args[1]
    assert "answer" in call_kwargs.get("output", {})
