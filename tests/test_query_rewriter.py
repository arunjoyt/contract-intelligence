"""Tests for pipeline.query_rewriter. No network calls — OpenAI and Embedder mocked."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pipeline.query_rewriter import (
    _HYDE_SYSTEM,
    _STEP_BACK_SYSTEM,
    QueryRewriter,
)

FAKE_VECTOR = [0.1] * 1536


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_openai_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_embedder(vector: list[float] = FAKE_VECTOR) -> MagicMock:
    e = MagicMock()
    e.embed_query.return_value = vector
    return e


# ---------------------------------------------------------------------------
# Construction / strategy
# ---------------------------------------------------------------------------


def test_default_strategy_is_hyde(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QUERY_REWRITE_STRATEGY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    r = QueryRewriter(embedder=_make_embedder())
    assert r._strategy == "hyde"


def test_strategy_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUERY_REWRITE_STRATEGY", "step_back")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    r = QueryRewriter(embedder=_make_embedder())
    assert r._strategy == "step_back"


def test_strategy_arg_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUERY_REWRITE_STRATEGY", "step_back")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    r = QueryRewriter(embedder=_make_embedder(), strategy="hyde")
    assert r._strategy == "hyde"


# ---------------------------------------------------------------------------
# HyDE strategy
# ---------------------------------------------------------------------------


def test_hyde_calls_gpt4o_with_hyde_system_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    embedder = _make_embedder()

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response("hypothetical doc")

    r = QueryRewriter(embedder=embedder, strategy="hyde")
    r._client = mock_client

    r.rewrite("What are the payment terms for Acme?")

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == _HYDE_SYSTEM
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "What are the payment terms for Acme?"


def test_hyde_embeds_gpt_response_not_original_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    embedder = _make_embedder()

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response(
        "This Purchase Order was issued to Acme on Net 30 terms."
    )

    r = QueryRewriter(embedder=embedder, strategy="hyde")
    r._client = mock_client

    r.rewrite("What are the payment terms?")

    embedder.embed_query.assert_called_once_with(
        "This Purchase Order was issued to Acme on Net 30 terms."
    )


def test_hyde_returns_rewritten_text_and_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    custom_vector = [0.9] * 1536
    embedder = _make_embedder(custom_vector)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response("hypothetical answer")

    r = QueryRewriter(embedder=embedder, strategy="hyde")
    r._client = mock_client

    text, vector = r.rewrite("query")

    assert text == "hypothetical answer"
    assert vector == custom_vector


def test_hyde_falls_back_to_original_query_on_empty_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    embedder = _make_embedder()

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response("")

    r = QueryRewriter(embedder=embedder, strategy="hyde")
    r._client = mock_client

    text, _ = r.rewrite("original query")

    assert text == "original query"
    embedder.embed_query.assert_called_once_with("original query")


# ---------------------------------------------------------------------------
# Step-back strategy
# ---------------------------------------------------------------------------


def test_step_back_calls_gpt4o_with_step_back_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    embedder = _make_embedder()

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response(
        "What are general payment terms?"
    )

    r = QueryRewriter(embedder=embedder, strategy="step_back")
    r._client = mock_client

    r.rewrite("What are the net 30 terms for PO-001?")

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    assert messages[0]["content"] == _STEP_BACK_SYSTEM


def test_step_back_embeds_rewritten_question(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    embedder = _make_embedder()

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response(
        "What are general procurement payment terms?"
    )

    r = QueryRewriter(embedder=embedder, strategy="step_back")
    r._client = mock_client

    r.rewrite("What is the payment term for PO-001?")

    embedder.embed_query.assert_called_once_with("What are general procurement payment terms?")


def test_step_back_uses_lower_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    embedder = _make_embedder()

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response("rewritten")

    r = QueryRewriter(embedder=embedder, strategy="step_back")
    r._client = mock_client

    r.rewrite("question")

    kwargs = mock_client.chat.completions.create.call_args[1]
    assert kwargs["temperature"] < 0.5


def test_step_back_falls_back_to_original_on_empty_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    embedder = _make_embedder()

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response("")

    r = QueryRewriter(embedder=embedder, strategy="step_back")
    r._client = mock_client

    text, _ = r.rewrite("original")

    assert text == "original"


# ---------------------------------------------------------------------------
# Model usage
# ---------------------------------------------------------------------------


def test_rewrite_uses_gpt4o_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response("answer")

    r = QueryRewriter(embedder=_make_embedder(), strategy="hyde")
    r._client = mock_client

    r.rewrite("query")

    kwargs = mock_client.chat.completions.create.call_args[1]
    assert kwargs["model"] == "gpt-4o"
