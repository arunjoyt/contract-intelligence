"""evaluate.py must exercise the same prompt / retrieval budget the live
pipeline uses -- it imports pipeline.constants rather than re-declaring them
(#110). These guard against a silent re-drift.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import evaluation.evaluate as ev
import pipeline.constants as pc
import pipeline.query_pipeline as qp


def test_evaluate_uses_shared_answer_prompt() -> None:
    assert ev.ANSWER_SYSTEM_PROMPT is pc.ANSWER_SYSTEM_PROMPT
    assert qp.ANSWER_SYSTEM_PROMPT is pc.ANSWER_SYSTEM_PROMPT


def test_evaluate_uses_shared_context_meta_fields() -> None:
    assert ev.CONTEXT_META_FIELDS is pc.CONTEXT_META_FIELDS
    assert qp.CONTEXT_META_FIELDS is pc.CONTEXT_META_FIELDS


def test_evaluate_uses_shared_retrieval_budget() -> None:
    assert ev.RETRIEVAL_TOP_K == pc.RETRIEVAL_TOP_K
    assert ev.RERANK_TOP_N == pc.RERANK_TOP_N
    assert ev.GENERATION_MAX_TOKENS == pc.GENERATION_MAX_TOKENS


def test_run_config_records_the_knobs() -> None:
    cfg = ev._run_config()
    assert cfg["retrieval_top_k"] == pc.RETRIEVAL_TOP_K
    assert cfg["rerank_top_n"] == pc.RERANK_TOP_N
    assert cfg["generation_max_tokens"] == pc.GENERATION_MAX_TOKENS
    assert cfg["rewrite_strategy"] in {"hyde", "step_back"}
    assert cfg["rewrite_model"] == ev.REWRITE_MODEL
    assert "ragas" in cfg["versions"]


# --- --collection flag (#127) -------------------------------------------------


def test_run_config_records_the_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scored collection is in the config block so a per-client result is
    self-describing (#127)."""
    monkeypatch.setenv("QDRANT_COLLECTION", "contract")
    assert ev._run_config("acme_staging")["qdrant_collection"] == "acme_staging"
    assert ev._run_config()["qdrant_collection"] == "contract"


def test_build_components_passes_collection_to_vector_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--collection reaches VectorStore so evaluate.py never guesses via env."""
    made = MagicMock()
    made.get_all_texts.return_value = [{"text": "x", "docname": "D", "chunk_index": 0}]
    vs_cls = MagicMock(return_value=made)
    monkeypatch.setattr("retrieval.vector_store.VectorStore", vs_cls)
    monkeypatch.setattr("ingestion.embedder.Embedder", MagicMock())
    monkeypatch.setattr("retrieval.hybrid_search.HybridSearch", MagicMock())
    monkeypatch.setattr("retrieval.reranker.Reranker", MagicMock())
    monkeypatch.setattr("pipeline.query_rewriter.QueryRewriter", MagicMock())

    ev._build_components("acme_staging")

    vs_cls.assert_called_once_with(collection="acme_staging")


def test_build_components_no_collection_uses_env_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made = MagicMock()
    made.get_all_texts.return_value = [{"text": "x", "docname": "D", "chunk_index": 0}]
    vs_cls = MagicMock(return_value=made)
    monkeypatch.setattr("retrieval.vector_store.VectorStore", vs_cls)
    monkeypatch.setattr("ingestion.embedder.Embedder", MagicMock())
    monkeypatch.setattr("retrieval.hybrid_search.HybridSearch", MagicMock())
    monkeypatch.setattr("retrieval.reranker.Reranker", MagicMock())
    monkeypatch.setattr("pipeline.query_rewriter.QueryRewriter", MagicMock())

    ev._build_components()

    vs_cls.assert_called_once_with()  # no kwargs -> VectorStore reads QDRANT_COLLECTION


def test_main_threads_collection_flag_to_evaluate(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}
    monkeypatch.setattr(
        ev, "evaluate", lambda *a, **k: captured.update(args=a)
    )
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate.py", "--split", "dev", "--collection", "acme_staging"],
    )
    ev.main()
    # evaluate(dataset_path, output_path, split, collection, no_judge)
    assert captured["args"][2] == "dev"
    assert captured["args"][3] == "acme_staging"
    assert captured["args"][4] is False


# --- --no-judge flag (deployment smoke test / latency run) -------------------


def test_run_config_records_no_judge() -> None:
    """A --no-judge run is self-describing: judge=False, judge model nulled."""
    assert ev._run_config()["judge"] is True
    assert ev._run_config()["ragas_judge_model"] is not None

    cfg = ev._run_config(no_judge=True)
    assert cfg["judge"] is False
    assert cfg["ragas_judge_model"] is None
    assert cfg["ragas_judge_embedding_model"] is None


def test_main_threads_no_judge_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}
    monkeypatch.setattr(ev, "evaluate", lambda *a, **k: captured.update(args=a))
    monkeypatch.setattr("sys.argv", ["evaluate.py", "--split", "dev", "--no-judge"])
    ev.main()
    assert captured["args"][4] is True
