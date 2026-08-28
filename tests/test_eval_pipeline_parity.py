"""evaluate.py must exercise the same prompt / retrieval budget the live
pipeline uses -- it imports pipeline.constants rather than re-declaring them
(#110). These guard against a silent re-drift.
"""

from __future__ import annotations

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
