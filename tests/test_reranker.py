"""Tests for retrieval.reranker. No network calls — CrossEncoder is mocked."""

from __future__ import annotations

import numpy as np
import pytest

from retrieval.reranker import RERANKER_MODEL, Reranker


def _make_mock_cross_encoder(scores: list[float]):
    """Return a mock CrossEncoder class whose predict() returns the given scores."""
    import unittest.mock as mock

    instance = mock.MagicMock()
    instance.predict.return_value = np.array(scores)
    cls = mock.MagicMock(return_value=instance)
    return cls, instance


def _doc(docname: str, text: str) -> dict:
    return {"docname": docname, "chunk_index": 0, "text": text}


# ---------------------------------------------------------------------------
# Lazy loading
# ---------------------------------------------------------------------------


def test_model_not_loaded_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_cls, _ = _make_mock_cross_encoder([])
    monkeypatch.setattr("retrieval.reranker.CrossEncoder", mock_cls)

    r = Reranker()

    mock_cls.assert_not_called()
    assert r._model is None


def test_model_loaded_on_first_rerank(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_cls, _ = _make_mock_cross_encoder([0.9])
    monkeypatch.setattr("retrieval.reranker.CrossEncoder", mock_cls)

    r = Reranker()
    r.rerank("query", [_doc("PO-001", "payment terms")])

    mock_cls.assert_called_once_with(RERANKER_MODEL)


def test_model_loaded_only_once_across_multiple_rerank_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_cls, instance = _make_mock_cross_encoder([0.5])
    monkeypatch.setattr("retrieval.reranker.CrossEncoder", mock_cls)

    r = Reranker()
    r.rerank("query", [_doc("PO-001", "payment")])
    r.rerank("query", [_doc("PO-002", "contract")])

    mock_cls.assert_called_once()  # constructor called exactly once


def test_warm_up_loads_model(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_cls, _ = _make_mock_cross_encoder([])
    monkeypatch.setattr("retrieval.reranker.CrossEncoder", mock_cls)

    r = Reranker()
    r.warm_up()

    mock_cls.assert_called_once_with(RERANKER_MODEL)
    assert r._model is not None


def test_warm_up_then_rerank_does_not_reload_model(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_cls, _ = _make_mock_cross_encoder([0.5])
    monkeypatch.setattr("retrieval.reranker.CrossEncoder", mock_cls)

    r = Reranker()
    r.warm_up()
    r.rerank("query", [_doc("PO-001", "payment")])

    mock_cls.assert_called_once()


# ---------------------------------------------------------------------------
# rerank — scoring and ordering
# ---------------------------------------------------------------------------


def test_rerank_returns_empty_for_empty_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_cls, _ = _make_mock_cross_encoder([])
    monkeypatch.setattr("retrieval.reranker.CrossEncoder", mock_cls)

    r = Reranker()
    result = r.rerank("query", [])

    assert result == []
    mock_cls.assert_not_called()


def test_rerank_returns_top_n_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    scores = [0.2, 0.9, 0.4, 0.7, 0.1]
    mock_cls, _ = _make_mock_cross_encoder(scores)
    monkeypatch.setattr("retrieval.reranker.CrossEncoder", mock_cls)

    candidates = [_doc(f"PO-{i:03d}", f"text {i}") for i in range(5)]
    r = Reranker()
    result = r.rerank("query", candidates, top_n=3)

    assert len(result) == 3


def test_rerank_orders_by_descending_score(monkeypatch: pytest.MonkeyPatch) -> None:
    # scores[1]=0.9, scores[3]=0.7, scores[2]=0.4, scores[0]=0.2, scores[4]=0.1
    scores = [0.2, 0.9, 0.4, 0.7, 0.1]
    mock_cls, _ = _make_mock_cross_encoder(scores)
    monkeypatch.setattr("retrieval.reranker.CrossEncoder", mock_cls)

    candidates = [_doc(f"DOC-{i}", f"text {i}") for i in range(5)]
    r = Reranker()
    result = r.rerank("query", candidates, top_n=3)

    assert result[0]["docname"] == "DOC-1"  # score 0.9
    assert result[1]["docname"] == "DOC-3"  # score 0.7
    assert result[2]["docname"] == "DOC-2"  # score 0.4


def test_rerank_fewer_candidates_than_top_n_returns_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_cls, _ = _make_mock_cross_encoder([0.8, 0.3])
    monkeypatch.setattr("retrieval.reranker.CrossEncoder", mock_cls)

    candidates = [_doc("PO-001", "text a"), _doc("PO-002", "text b")]
    r = Reranker()
    result = r.rerank("query", candidates, top_n=10)

    assert len(result) == 2


def test_rerank_single_candidate_returns_it(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_cls, _ = _make_mock_cross_encoder([0.75])
    monkeypatch.setattr("retrieval.reranker.CrossEncoder", mock_cls)

    doc = _doc("PO-001", "payment terms net 30")
    r = Reranker()
    result = r.rerank("what are the payment terms", [doc])

    assert result == [doc]


# ---------------------------------------------------------------------------
# rerank — pairs passed to predict
# ---------------------------------------------------------------------------


def test_rerank_passes_query_text_pairs_to_predict(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_cls, mock_instance = _make_mock_cross_encoder([0.5, 0.3])
    monkeypatch.setattr("retrieval.reranker.CrossEncoder", mock_cls)

    candidates = [
        _doc("PO-001", "payment terms net 30"),
        _doc("CON-001", "contract liability clause"),
    ]
    r = Reranker()
    r.rerank("payment", candidates)

    expected_pairs = [
        ("payment", "payment terms net 30"),
        ("payment", "contract liability clause"),
    ]
    mock_instance.predict.assert_called_once_with(expected_pairs)


def test_rerank_uses_empty_string_for_missing_text_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_cls, mock_instance = _make_mock_cross_encoder([0.5])
    monkeypatch.setattr("retrieval.reranker.CrossEncoder", mock_cls)

    candidate = {"docname": "PO-001", "chunk_index": 0}  # no "text" key
    r = Reranker()
    r.rerank("query", [candidate])

    pairs = mock_instance.predict.call_args[0][0]
    assert pairs[0] == ("query", "")


# ---------------------------------------------------------------------------
# Custom model name
# ---------------------------------------------------------------------------


def test_custom_model_name_passed_to_cross_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_cls, _ = _make_mock_cross_encoder([0.5])
    monkeypatch.setattr("retrieval.reranker.CrossEncoder", mock_cls)

    r = Reranker(model_name="cross-encoder/custom-model")
    r.rerank("query", [_doc("PO-001", "text")])

    mock_cls.assert_called_once_with("cross-encoder/custom-model")
