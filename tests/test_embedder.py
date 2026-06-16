"""Tests for ingestion.embedder. No network calls — the OpenAI client is mocked."""

from unittest.mock import MagicMock

import pytest

from ingestion.embedder import EMBEDDING_MODEL, Embedder


def _fake_response(vectors: list[list[float]]) -> MagicMock:
    response = MagicMock()
    response.data = [MagicMock(embedding=v) for v in vectors]
    return response


@pytest.fixture
def embedder() -> Embedder:
    e = Embedder(api_key="sk-test")
    e._client.embeddings.create = MagicMock()  # type: ignore[method-assign]
    return e


def test_embed_texts_returns_vectors_in_order(embedder: Embedder) -> None:
    embedder._client.embeddings.create.return_value = _fake_response([[0.1, 0.2], [0.3, 0.4]])

    result = embedder.embed_texts(["a", "b"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    embedder._client.embeddings.create.assert_called_once_with(
        model=EMBEDDING_MODEL, input=["a", "b"]
    )


def test_embed_texts_empty_list_returns_empty_without_calling_api(embedder: Embedder) -> None:
    result = embedder.embed_texts([])

    assert result == []
    embedder._client.embeddings.create.assert_not_called()


def test_embed_texts_batches_over_2048_limit(embedder: Embedder) -> None:
    call_sizes: list[int] = []

    def fake_create(model: str, input: list[str]) -> MagicMock:  # noqa: A002
        call_sizes.append(len(input))
        return _fake_response([[float(i)] for i in range(len(input))])

    embedder._client.embeddings.create.side_effect = fake_create

    texts = [f"text-{i}" for i in range(2500)]
    result = embedder.embed_texts(texts)

    assert len(result) == 2500
    assert call_sizes == [2048, 452]


def test_embed_query_returns_single_vector(embedder: Embedder) -> None:
    embedder._client.embeddings.create.return_value = _fake_response([[0.5, 0.6]])

    result = embedder.embed_query("what is the payment term?")

    assert result == [0.5, 0.6]
    embedder._client.embeddings.create.assert_called_once_with(
        model=EMBEDDING_MODEL, input=["what is the payment term?"]
    )


def test_constructor_uses_explicit_api_key_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-test")
    embedder = Embedder(api_key="sk-explicit")
    assert embedder._client.api_key == "sk-explicit"


def test_constructor_falls_back_to_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-test")
    embedder = Embedder()
    assert embedder._client.api_key == "sk-env-test"


def test_constructor_uses_custom_model() -> None:
    embedder = Embedder(api_key="sk-test", model="text-embedding-3-large")
    assert embedder._model == "text-embedding-3-large"
