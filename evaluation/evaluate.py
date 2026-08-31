"""RAGAS evaluation for the Contract Intelligence RAG pipeline.

Loads evaluation/test_dataset.json, runs the pipeline against each question in a
single pass (rewrite → hybrid search → rerank → generate), captures the raw
retrieved chunk texts for RAGAS, then scores faithfulness, answer_relevancy,
context_recall, and context_precision.  Results are written to
evaluation/results.json (including a `config` block recording the prompt /
retrieval / ingestion knobs and library versions the run used, and a `costs`
block with pipeline + RAGAS-judge token cost and request counts — the daily
OpenAI cap is per-request, so `request_count` is the number that matters, #130).

The generation prompt, context-framing fields and top_k / top_n come from
pipeline.constants -- the same module query_pipeline.py uses -- so a baseline
scores the exact pipeline production runs, not a drifting copy (#110).

When LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY are set, each question's trace and
RAGAS scores are also pushed to Langfuse and, if the question was pushed via
push_dataset.py first, grouped into a comparable Experiment run in the Langfuse
Datasets UI (#109). This is additive -- results.json is written exactly as
before either way, and evaluate.py works with no Langfuse credentials at all.

Usage
-----
    python evaluation/push_dataset.py   # once, and after editing test_dataset.json
    python evaluation/evaluate.py [--dataset PATH] [--output PATH] [--split dev|test|all]

The dataset carries a dev/test ``split`` on every entry (#112). Tune knobs and
iterate the prompt against ``--split dev``; freeze ``results.baseline.json``
against ``--split test`` only, so methodology work never overfits the numbers it
is judged on. ``--split all`` (the default) scores every entry.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent
_PROJECT_ROOT = _ROOT.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (  # noqa: E402
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    OPENAI_MODEL,
    REWRITE_MODEL,
)
from evaluation.langfuse_dataset import build_client, dataset_item_id  # noqa: E402
from pipeline.constants import (  # noqa: E402
    ANSWER_SYSTEM_PROMPT,
    CONTEXT_META_FIELDS,
    GENERATION_MAX_TOKENS,
    RERANK_TOP_N,
    RETRIEVAL_TOP_K,
)

_DEFAULT_DATASET = _ROOT / "test_dataset.json"
_DEFAULT_OUTPUT = _ROOT / "results.json"

# Explicitly pinned rather than left to ragas_evaluate()'s implicit default, so a
# run's judge is reproducible and visible instead of silently coupled to whatever
# ragas.llms.llm_factory/embedding_factory default to on a given ragas version (#109).
# Set to ragas's own current defaults (as of ragas 0.2.6) rather than this app's
# OPENAI_MODEL/EMBEDDING_MODEL -- the judge is deliberately a separate, cheaper
# model from the one being judged, and pinning to ragas's existing default keeps
# this baseline continuous with `results.baseline.json`'s prior (implicitly-pinned)
# runs instead of shifting the numbers as a side effect of adding the pin. In
# particular, answer_relevancy is embedding-model-sensitive -- swapping in
# EMBEDDING_MODEL (text-embedding-3-small) here measurably changed it in testing.
_RAGAS_JUDGE_MODEL = "gpt-4o-mini"
_RAGAS_JUDGE_EMBEDDING_MODEL = "text-embedding-ada-002"

# USD per 1M tokens, (input, output). OpenAI list prices -- refresh when they move.
# Used only to price an eval run after the fact (#130); the pipeline's production
# cost is metered exactly by Langfuse. Embedding models list one rate, repeated.
_PRICE_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.02),
    "text-embedding-ada-002": (0.10, 0.10),
}


def _cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Token cost from the static rate table, or None for an unpriced model."""
    rate = _PRICE_PER_1M_TOKENS.get(model)
    if rate is None:
        return None
    return (input_tokens * rate[0] + output_tokens * rate[1]) / 1_000_000


def _add_tokens(acc: dict[str, dict[str, int]], model: str, usage: dict[str, int]) -> None:
    """Fold one call's ``{input, output, ...}`` usage into a per-model accumulator.

    ``output`` is optional -- the embeddings API reports no completion tokens.
    """
    slot = acc.setdefault(model, {"input": 0, "output": 0})
    slot["input"] += int(usage["input"])
    slot["output"] += int(usage.get("output", 0))


def _judge_usage(ragas_result: Any) -> dict[str, Any]:
    """Judge-side token / request / cost totals from a RAGAS result that was run
    with ``token_usage_parser=get_token_usage_for_openai``.

    RAGAS's ``CostCallbackHandler`` appends one ``TokenUsage`` per judge LLM call,
    so ``len(usage_data)`` is the judge request count -- the number that actually
    hit the 10k/day ``gpt-4o-mini`` cap during #113, not the dollar figure.
    Embedding calls (``answer_relevancy``'s ``ada-002``) are not on this path;
    their cost is negligible next to the LLM calls ($0.10/1M tokens).
    """
    cb = getattr(ragas_result, "cost_cb", None)
    usage_data = getattr(cb, "usage_data", None)
    if not usage_data:
        return {"captured": False}
    input_tokens = sum(int(u.input_tokens) for u in usage_data)
    output_tokens = sum(int(u.output_tokens) for u in usage_data)
    cost = _cost_usd(_RAGAS_JUDGE_MODEL, input_tokens, output_tokens)
    return {
        "captured": True,
        "model": _RAGAS_JUDGE_MODEL,
        "requests": len(usage_data),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "usd": round(cost, 4) if cost is not None else None,
        "embedding_cost_excluded": True,
    }


# ---------------------------------------------------------------------------
# Langfuse experiment logging (optional -- additive on top of the local
# results.json/results.baseline.json path, see #109). Absent
# LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY this whole layer is a no-op: evaluate.py
# still runs and writes results.json exactly as before.
# ---------------------------------------------------------------------------


def _run_name() -> str:
    """A short, unique label for one evaluate.py invocation, used to group this
    run's traces together in the Langfuse Datasets UI.

    Includes a time suffix even in the git-SHA case: two invocations against the
    same uncommitted HEAD (e.g. iterating on a prompt tweak before committing)
    would otherwise collide into a single Langfuse run instead of showing as
    separate, comparable runs.
    """
    import subprocess

    stamp = datetime.now(UTC).strftime("%H%M%S")
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=_PROJECT_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        return f"eval-{sha}-{stamp}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return f"eval-{datetime.now(UTC).strftime('%Y%m%dT')}{stamp}Z"


def _link_dataset_run(langfuse, question: str, trace, run_name: str, run_metadata: dict) -> None:
    """Best-effort: group `trace` into this run's Langfuse Dataset entry.

    Silently skipped if the question was never pushed via push_dataset.py -- the
    trace and its scores are still recorded either way, just not grouped into a
    comparable Experiment run in the Datasets UI.
    """
    try:
        item = langfuse.get_dataset_item(id=dataset_item_id(question))
    except Exception:
        return
    item.link(trace, run_name=run_name, run_metadata=run_metadata)


# ---------------------------------------------------------------------------
# Component bootstrap
# ---------------------------------------------------------------------------


def _build_components():
    from ingestion.embedder import Embedder
    from pipeline.query_rewriter import QueryRewriter
    from retrieval.hybrid_search import HybridSearch
    from retrieval.reranker import Reranker
    from retrieval.vector_store import VectorStore

    logger.info("Initialising components …")
    embedder = Embedder()

    vector_store = VectorStore()
    vector_store.ensure_collection()

    existing_texts = vector_store.get_all_texts()
    if not existing_texts:
        logger.error(
            "Qdrant collection %r is empty. evaluate.py runs against the real "
            "indexed corpus — run a full ingest first (POST /ingest/full) so "
            "retrieval, chunking and parsing are all reflected in the scores.",
            vector_store._collection,
        )
        sys.exit(1)

    hybrid_search = HybridSearch(embedder=embedder, vector_store=vector_store)
    hybrid_search.build_bm25_index(existing_texts)

    reranker = Reranker()
    reranker.warm_up()

    rewriter = QueryRewriter(embedder=embedder)

    logger.info("Components ready")
    return rewriter, hybrid_search, reranker


# ---------------------------------------------------------------------------
# Single-pass inference — captures retrieved texts without a second LLM call
# ---------------------------------------------------------------------------


def _frame_chunk(chunk: dict) -> str:
    """One retrieved chunk as the generator sees it: ``[docname] (metadata):\\n text``.

    Same shape as ``query_pipeline._build_context``.  RAGAS scores against this
    framed form (not the bare ``text``) so faithfulness / context precision /
    recall see the metadata the answer actually relies on — e.g. a ``status:
    Unsigned`` that never appears in the chunk's own text.
    """
    docname = chunk.get("docname", "unknown")
    meta_bits = [f"{key}: {value}" for key in CONTEXT_META_FIELDS if (value := chunk.get(key))]
    meta_str = f" ({'; '.join(meta_bits)})" if meta_bits else ""
    return f"[{docname}]{meta_str}:\n{chunk.get('text', '')}"


def _span(trace: Any, name: str, fn, summarize=None):
    """Execute ``fn`` inside a Langfuse span if tracing is active.

    Mirrors ``QueryPipeline._span`` (pipeline/query_pipeline.py) so an eval trace
    carries the same per-step latency breakdown a production trace does, instead
    of a single opaque call with no timing. ``summarize``, if given, converts the
    result into a small span output (e.g. docnames only) instead of dumping the
    full result -- avoids duplicating chunk text/embeddings into Langfuse.
    """
    if trace is None:
        return fn()
    span = trace.span(name=name)
    try:
        result = fn()
        span.end(output=summarize(result) if summarize else None)
        return result
    except Exception:
        span.end(level="ERROR")
        raise


def _docnames(chunks: list[dict]) -> list[str]:
    """Docnames only, in rank order -- a small span output instead of full chunks."""
    return [c.get("docname", "") for c in chunks]


def _generate_answer(openai_client, question: str, context: str) -> tuple[str, dict[str, int]]:
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
        max_tokens=GENERATION_MAX_TOKENS,
    )
    usage = {
        "input": response.usage.prompt_tokens,
        "output": response.usage.completion_tokens,
        "total": response.usage.total_tokens,
    }
    return response.choices[0].message.content or "", usage


def _run_question(
    question: str,
    rewriter,
    hybrid_search,
    reranker,
    openai_client,
    trace: Any = None,
) -> tuple[str, list[str], dict[str, Any]]:
    """Return (answer, retrieved_contexts, usage) in a single pipeline pass.

    ``usage`` is ``{"rewrite": {...} | None, "embed": {...}, "generate": {...}}``
    in the pipeline's usage shape -- the caller prices and counts it (#130). The
    rewrite call was previously wrapped in a plain span with no ``model``/``usage``,
    so its ``REWRITE_MODEL`` cost never reached Langfuse's total; it is a
    ``generation`` now, matching the answer call.

    Rewrite (chat) and embed are separate steps: the query is embedded once, in
    ``rewriter.embed``, and the vector is threaded into ``hybrid_search.search``
    so the dense leg does not embed the same text again (#138). Mirrors
    ``QueryPipeline._rewrite_and_embed``.

    When ``trace`` is given, each step is wrapped in a Langfuse span/generation
    (same shape as ``QueryPipeline.run``) so the trace records real per-step
    latency and, via the generation's ``model``/``usage``, a real cost -- a flat
    trace with only ``input``/``output`` set (the original implementation) always
    shows 0.00s / $0.00 in the Langfuse UI since there's nothing to compute either
    from.
    """
    if trace and rewriter.strategy != "none":
        rewrite_span = trace.generation(name="rewrite", model=REWRITE_MODEL)
        try:
            rewritten = rewriter.rewrite_text(question)
            rewrite_span.end(output=rewritten, usage=rewriter.last_usage)
        except Exception:
            rewrite_span.end(level="ERROR")
            raise
    else:
        rewritten = rewriter.rewrite_text(question)
    rewrite_usage = rewriter.last_usage

    if trace:
        embed_span = trace.generation(name="embed_query", model=EMBEDDING_MODEL)
        try:
            query_vector = rewriter.embed(rewritten)
            embed_span.end(usage=rewriter.last_embed_usage)
        except Exception:
            embed_span.end(level="ERROR")
            raise
    else:
        query_vector = rewriter.embed(rewritten)
    embed_usage = rewriter.last_embed_usage

    candidates = _span(
        trace,
        "hybrid_search",
        lambda: hybrid_search.search(
            rewritten, None, top_k=RETRIEVAL_TOP_K, query_vector=query_vector
        ),
        summarize=_docnames,
    )
    top_chunks = _span(
        trace,
        "rerank",
        lambda: reranker.rerank(question, candidates, top_n=RERANK_TOP_N),
        summarize=_docnames,
    )

    retrieved_contexts = [_frame_chunk(c) for c in top_chunks if c.get("text")]
    context = "\n\n---\n\n".join(retrieved_contexts)

    if trace:
        gen_span = trace.generation(name="generate", model=OPENAI_MODEL)
        try:
            answer, gen_usage = _generate_answer(openai_client, question, context)
            gen_span.end(output=answer, usage=gen_usage)
        except Exception:
            gen_span.end(level="ERROR")
            raise
    else:
        answer, gen_usage = _generate_answer(openai_client, question, context)

    return answer, retrieved_contexts, {
        "rewrite": rewrite_usage,
        "embed": embed_usage,
        "generate": gen_usage,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


_VALID_SPLITS = ("dev", "test", "all")


def _filter_by_split(entries: list[dict], split: str) -> list[dict]:
    """Keep only entries whose ``split`` matches (``all`` keeps everything).

    Every entry is expected to carry a ``split`` of ``"dev"`` or ``"test"`` (#112).
    An entry missing the field is kept under ``--split all`` but dropped (with a
    warning) under ``dev`` / ``test`` — a partly-labelled dataset scores a
    predictable subset instead of silently reporting on the wrong N.
    """
    if split == "all":
        return entries
    kept, unlabelled = [], 0
    for entry in entries:
        entry_split = entry.get("split")
        if entry_split is None:
            unlabelled += 1
        elif entry_split == split:
            kept.append(entry)
    if unlabelled:
        logger.warning(
            "%d %s no 'split' field — excluded from --split %s",
            unlabelled,
            "entry has" if unlabelled == 1 else "entries have",
            split,
        )
    return kept


def evaluate(dataset_path: Path, output_path: Path, split: str = "all") -> None:
    from openai import OpenAI
    from ragas import EvaluationDataset
    from ragas import evaluate as ragas_evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    all_entries: list[dict] = json.loads(dataset_path.read_text())
    entries = _filter_by_split(all_entries, split)
    logger.info(
        "Loaded %d entries from %s (%d after --split %s)",
        len(all_entries),
        dataset_path,
        len(entries),
        split,
    )
    if not entries:
        logger.error("No entries match --split %s — aborting", split)
        sys.exit(1)

    rewriter, hybrid_search, reranker = _build_components()
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    run_config = _run_config()
    langfuse = build_client()
    run_name = _run_name() if langfuse else None
    if langfuse:
        logger.info("Langfuse experiment logging enabled -- run %r", run_name)
    else:
        logger.info("LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY not set -- experiment not recorded")
    traces_by_question: dict[str, Any] = {}

    samples: list[SingleTurnSample] = []
    sample_meta: list[dict] = []  # parallel to `samples`: {case_class, expected_fail}
    sample_questions: list[str] = []  # parallel to `samples` -- for post-RAGAS score lookup
    per_question: list[dict] = []
    refusal_results: list[dict] = []  # {question, case_class, handled}

    # Pipeline-side cost accounting (#130). Every question fires a generate call,
    # a query-embed call, and -- unless QUERY_REWRITE_STRATEGY=none -- a rewrite
    # call; refusal cases included, since they cost the same. The query embed is
    # ~150 tokens (~$0/query) but is counted for completeness (#138). Request
    # count, not dollars, is the axis the daily OpenAI cap is enforced on.
    pipeline_tokens: dict[str, dict[str, int]] = {}  # model -> {input, output}
    pipeline_requests = 0

    for i, entry in enumerate(entries, 1):
        question: str = entry["question"]
        capability: str = entry.get("capability", "")
        case_class: str = entry.get("case_class", "uncategorized")
        expected_fail: bool = bool(entry.get("expected_fail", False))
        reference_contexts: list[str] = entry["ground_truth_contexts"]
        reference_answer: str = entry.get("ground_truth_answer", "")
        # An entry with no ground-truth contexts is a grounded-refusal case
        # (e.g. the Globex supplier that is never seeded, or a clause absent from
        # a real contract). RAGAS's answer/context metrics have no meaning for
        # "correctly answered nothing", so these are checked with a plain
        # refusal-string match and kept out of the RAGAS set.
        is_refusal_case = not reference_contexts

        logger.info("[%d/%d] %s", i, len(entries), question[:80])

        trace = None
        if langfuse:
            trace = langfuse.trace(
                name="eval_question",
                input={"question": question},
                metadata={"case_class": case_class, "capability": capability},
            )

        try:
            answer, retrieved_contexts, q_usage = _run_question(
                question, rewriter, hybrid_search, reranker, openai_client, trace=trace
            )
        except Exception:
            logger.exception("Pipeline failed for question %d — skipping", i)
            if trace:
                trace.update(level="ERROR")
            continue

        if q_usage["rewrite"]:
            _add_tokens(pipeline_tokens, REWRITE_MODEL, q_usage["rewrite"])
            pipeline_requests += 1
        if q_usage["embed"]:
            _add_tokens(pipeline_tokens, EMBEDDING_MODEL, q_usage["embed"])
            pipeline_requests += 1
        _add_tokens(pipeline_tokens, OPENAI_MODEL, q_usage["generate"])
        pipeline_requests += 1

        if trace:
            trace.update(output={"answer": answer})
            traces_by_question[question] = trace
            _link_dataset_run(langfuse, question, trace, run_name, run_config)

        record = {
            "case_class": case_class,
            "capability": capability,
            "question": question,
            "answer": answer,
            "retrieved_context_count": len(retrieved_contexts),
        }

        if is_refusal_case:
            handled = _is_refusal(answer)
            refusal_results.append(
                {"question": question, "case_class": case_class, "handled": handled}
            )
            per_question.append({**record, "refusal_case": True, "refusal_handled": handled})
            if trace:
                langfuse.score(
                    trace_id=trace.id,
                    name="refusal_handled",
                    value=handled,
                    data_type="BOOLEAN",
                )
            continue

        # Safety net only — hybrid search over a populated collection should
        # never come back empty. If it does, fall back to reference_contexts so
        # context_recall/context_precision still compute, though
        # faithfulness/answer_relevancy reflect the real (broken) retrieval.
        if not retrieved_contexts:
            logger.warning("No retrieved contexts for question %d — using ground truth", i)
            retrieved_contexts = reference_contexts

        if expected_fail:
            record["expected_fail"] = True
        per_question.append(record)
        sample_meta.append({"case_class": case_class, "expected_fail": expected_fail})
        sample_questions.append(question)
        samples.append(
            SingleTurnSample(
                user_input=question,
                retrieved_contexts=retrieved_contexts,
                reference_contexts=reference_contexts,
                response=answer,
                reference=reference_answer,
            )
        )

    if not samples and not refusal_results:
        logger.error("No samples collected — aborting")
        sys.exit(1)

    metrics: dict[str, float] = {}
    metrics_by_case_class: dict[str, dict] = {}
    expected_fail_metrics: dict[str, float] = {}
    num_headline = 0
    judge_usage: dict[str, Any] = {"captured": False}

    if samples:
        logger.info("Running RAGAS on %d samples …", len(samples))
        from ragas.cost import get_token_usage_for_openai
        from ragas.embeddings import embedding_factory
        from ragas.llms import llm_factory

        ragas_result = ragas_evaluate(
            dataset=EvaluationDataset(samples=samples),
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            llm=llm_factory(model=_RAGAS_JUDGE_MODEL),
            embeddings=embedding_factory(model=_RAGAS_JUDGE_EMBEDDING_MODEL),
            # Records one TokenUsage per judge LLM call; _judge_usage() reads the
            # count (the 10k/day-cap axis) and token totals back off it (#130).
            token_usage_parser=get_token_usage_for_openai,
            raise_exceptions=False,
        )
        judge_usage = _judge_usage(ragas_result)
        df = ragas_result.to_pandas().reset_index(drop=True)
        df["case_class"] = [m["case_class"] for m in sample_meta]
        df["expected_fail"] = [m["expected_fail"] for m in sample_meta]

        if langfuse:
            for idx, question in enumerate(sample_questions):
                trace = traces_by_question.get(question)
                if not trace:
                    continue
                row = df.iloc[idx]
                for col in _RAGAS_COLS:
                    val = row.get(col)
                    if val is not None and not (isinstance(val, float) and math.isnan(val)):
                        langfuse.score(
                            trace_id=trace.id, name=col, value=float(val), data_type="NUMERIC"
                        )

        # Headline = everything RAGAS scored except the known-limitation cases,
        # which are reported on their own so they don't drag the aggregate.
        headline = df[~df["expected_fail"]]
        num_headline = len(headline)
        metrics = {col: _col_mean(headline, col) for col in _RAGAS_COLS}

        for cc in sorted(df["case_class"].unique()):
            sub = df[df["case_class"] == cc]
            metrics_by_case_class[cc] = {
                **{col: _col_mean(sub, col) for col in _RAGAS_COLS},
                "n": int(len(sub)),
            }

        failing = df[df["expected_fail"]]
        if len(failing):
            expected_fail_metrics = {
                **{col: _col_mean(failing, col) for col in _RAGAS_COLS},
                "n": int(len(failing)),
            }

        # Attach per-question RAGAS scores back onto the per_question records, so a
        # tuning pass (#113) can run paired tests across configs instead of only
        # comparing slice means. Matched by question text (per_question also holds
        # the refusal records, which RAGAS never scored).
        scores_by_q = {
            sample_questions[i]: {
                col: (None if _is_nan(df.iloc[i][col]) else round(float(df.iloc[i][col]), 4))
                for col in _RAGAS_COLS
            }
            for i in range(len(sample_questions))
        }
        for rec in per_question:
            if rec["question"] in scores_by_q:
                rec["scores"] = scores_by_q[rec["question"]]

    if refusal_results:
        handled = sum(r["handled"] for r in refusal_results)
        metrics["refusal_handled"] = round(handled / len(refusal_results), 4)

    if langfuse:
        langfuse.flush()

    logger.info("Headline metrics: %s", metrics)
    logger.info("By case_class: %s", metrics_by_case_class)

    costs = _assemble_costs(pipeline_tokens, pipeline_requests, judge_usage)
    pl, jd = costs["pipeline"], costs["judge"]
    logger.info(
        "Cost: pipeline $%s (%d req) + judge $%s (%d req) = $%s (%d req total)",
        f"{pl['usd']:.4f}" if pl["usd"] is not None else "?",
        pl["requests"],
        f"{jd['usd']:.4f}" if jd.get("usd") is not None else "?",
        jd.get("requests", 0),
        f"{costs['total_usd']:.4f}" if costs["total_usd"] is not None else "?",
        costs["request_count"],
    )

    output = {
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset": _repo_relative(dataset_path),
        "split": split,
        "model": OPENAI_MODEL,
        "config": run_config,
        "num_questions": len(entries),
        "num_headline": num_headline,
        "num_scored": len(samples),
        "num_refusal_cases": len(refusal_results),
        "num_expected_fail": sum(m["expected_fail"] for m in sample_meta),
        "metrics": metrics,
        "metrics_by_case_class": metrics_by_case_class,
        "expected_fail_metrics": expected_fail_metrics,
        "refusal_results": refusal_results,
        "per_question": per_question,
        "costs": costs,
        "request_count": costs["request_count"],
        "langfuse_run_name": run_name,
    }
    output_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info("Results written to %s", output_path)


def _assemble_costs(
    pipeline_tokens: dict[str, dict[str, int]],
    pipeline_requests: int,
    judge_usage: dict[str, Any],
) -> dict[str, Any]:
    """Pipeline + judge cost / request totals for results.json (#130).

    ``pipeline.usd`` is None if any model in the run is missing from the rate
    table (an unpriced model shouldn't silently read as $0). ``request_count`` is
    the headline number -- the daily OpenAI cap is per-request.
    """
    pipeline_cost = 0.0
    priced = True
    for model, tok in pipeline_tokens.items():
        c = _cost_usd(model, tok["input"], tok["output"])
        if c is None:
            priced = False
        else:
            pipeline_cost += c
    pipeline_usd = round(pipeline_cost, 4) if priced else None

    judge_usd = judge_usage.get("usd") if judge_usage.get("captured") else None
    total_usd = None
    if pipeline_usd is not None and judge_usd is not None:
        total_usd = round(pipeline_usd + judge_usd, 4)

    return {
        "pipeline": {
            "usd": pipeline_usd,
            "requests": pipeline_requests,
            "tokens_by_model": pipeline_tokens,
        },
        "judge": judge_usage,
        "total_usd": total_usd,
        "request_count": pipeline_requests + judge_usage.get("requests", 0),
    }


def _repo_relative(path: Path) -> str:
    """Path relative to the repo root when it lives inside it, else the bare name.

    Keeps ``results.json`` free of machine-specific absolute paths.
    """
    try:
        return str(path.resolve().relative_to(_PROJECT_ROOT))
    except ValueError:
        return path.name


def _run_config() -> dict:
    """Everything that shifts the numbers but isn't the dataset or the pipeline
    code -- so a baseline is self-describing and two runs are comparable.

    Includes the ingestion-time knobs (chunk_size/chunk_overlap/embedding_model)
    even though evaluate.py doesn't run ingestion itself -- it scores whatever is
    already indexed in Qdrant. Recording them here means a baseline still shows
    what produced that collection, so a score delta isn't silently misattributed
    to a query-time change when the corpus was actually re-chunked or re-embedded.
    Re-ingest deliberately (`POST /ingest/full`) before rerunning evaluate.py when
    testing a chunking change -- evaluate.py does not do this itself, since
    embedding a full corpus has a very different cost profile than scoring ~20
    questions.
    """
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    versions: dict[str, str] = {}
    for pkg in ("ragas", "openai", "sentence-transformers", "qdrant-client", "rank-bm25"):
        try:
            versions[pkg] = _pkg_version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not installed"

    return {
        "rewrite_strategy": os.environ.get("QUERY_REWRITE_STRATEGY", "hyde"),
        "rewrite_model": REWRITE_MODEL,
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "rerank_top_n": RERANK_TOP_N,
        "generation_model": OPENAI_MODEL,
        "generation_max_tokens": GENERATION_MAX_TOKENS,
        "ragas_judge_model": _RAGAS_JUDGE_MODEL,
        "ragas_judge_embedding_model": _RAGAS_JUDGE_EMBEDDING_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "versions": versions,
    }


_REFUSAL_MARKER = "could not find relevant information"

_RAGAS_COLS = ("faithfulness", "answer_relevancy", "context_recall", "context_precision")


def _is_refusal(answer: str) -> bool:
    """True if the answer is the pipeline's grounded-refusal response."""
    return _REFUSAL_MARKER in answer.lower()


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _col_mean(df, col: str) -> float:
    """Mean of one metric column over a (possibly filtered) RAGAS results frame,
    ignoring NaN. 0.0 if the column is absent or all-NaN."""
    try:
        vals = df[col].dropna()
        return round(float(vals.mean()), 4) if len(vals) else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on the Contract Intelligence RAG pipeline"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_DEFAULT_DATASET,
        help="Path to test_dataset.json (default: evaluation/test_dataset.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Path for results.json (default: evaluation/results.json)",
    )
    parser.add_argument(
        "--split",
        choices=_VALID_SPLITS,
        default="all",
        help=(
            "Which dev/test split to score (default: all). Tune against 'dev'; "
            "freeze results.baseline.json against 'test'."
        ),
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        logger.error("Dataset not found: %s", args.dataset)
        sys.exit(1)

    evaluate(args.dataset, args.output, args.split)


if __name__ == "__main__":
    main()
