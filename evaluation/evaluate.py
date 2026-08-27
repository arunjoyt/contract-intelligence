"""RAGAS evaluation for the Contract Intelligence RAG pipeline.

Loads evaluation/test_dataset.json, runs the pipeline against each question in a
single pass (rewrite → hybrid search → rerank → generate), captures the raw
retrieved chunk texts for RAGAS, then scores faithfulness, answer_relevancy,
context_recall, and context_precision.  Results are written to
evaluation/results.json.

Usage
-----
    python evaluation/evaluate.py [--dataset PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent
_PROJECT_ROOT = _ROOT.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import OPENAI_MODEL  # noqa: E402

_DEFAULT_DATASET = _ROOT / "test_dataset.json"
_DEFAULT_OUTPUT = _ROOT / "results.json"

_ANSWER_SYSTEM = """\
You are a contract analyst assistant. Answer the user's question using ONLY \
the context below.

Rules:
- Cite every claim with [docname] immediately after the relevant sentence.
- The context contains exact field values (status codes, dates, etc.) from \
contract records. You may use ordinary language understanding to relate the \
user's wording to those exact values -- e.g. "signed" may match "Unsigned" as its \
negation; "terminated"/"ended" may match a status like "Cancelled". Interpreting \
the plain meaning of a value that IS present in the context is not "outside knowledge."
- Do not invent facts, entities, values, or numbers that do not appear in the context.
- If the context genuinely contains nothing relevant to the question, respond with \
exactly: "I could not find relevant information in the contract documents."

Context:
{context}
"""

# Mirrors pipeline/query_pipeline.py's _CONTEXT_META_FIELDS -- keep in sync.
# (This duplication, and top_k/top_n/_ANSWER_SYSTEM below, is tracked for removal
# in #110 -- evaluate.py should import these from the pipeline, not re-declare them.)
_CONTEXT_META_FIELDS = (
    "source_doctype",
    "supplier",
    "supplier_group",
    "status",
    "company",
    "start_date",
    "end_date",
    "linked_doctype",
    "linked_docname",
)


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
    meta_bits = [f"{key}: {value}" for key in _CONTEXT_META_FIELDS if (value := chunk.get(key))]
    meta_str = f" ({'; '.join(meta_bits)})" if meta_bits else ""
    return f"[{docname}]{meta_str}:\n{chunk.get('text', '')}"


def _run_question(
    question: str,
    rewriter,
    hybrid_search,
    reranker,
    openai_client,
) -> tuple[str, list[str]]:
    """Return (answer, retrieved_contexts) in a single pipeline pass."""
    rewritten, _ = rewriter.rewrite(question)
    candidates = hybrid_search.search(rewritten, None, top_k=20)
    top_chunks = reranker.rerank(question, candidates, top_n=5)

    retrieved_contexts = [_frame_chunk(c) for c in top_chunks if c.get("text")]
    context = "\n\n---\n\n".join(retrieved_contexts)

    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": _ANSWER_SYSTEM.format(context=context)},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    answer = response.choices[0].message.content or ""
    return answer, retrieved_contexts


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def evaluate(dataset_path: Path, output_path: Path) -> None:
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

    entries: list[dict] = json.loads(dataset_path.read_text())
    logger.info("Loaded %d entries from %s", len(entries), dataset_path)

    rewriter, hybrid_search, reranker = _build_components()
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    samples: list[SingleTurnSample] = []
    sample_meta: list[dict] = []  # parallel to `samples`: {case_class, expected_fail}
    per_question: list[dict] = []
    refusal_results: list[dict] = []  # {question, case_class, handled}

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
        try:
            answer, retrieved_contexts = _run_question(
                question, rewriter, hybrid_search, reranker, openai_client
            )
        except Exception:
            logger.exception("Pipeline failed for question %d — skipping", i)
            continue

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

    if samples:
        logger.info("Running RAGAS on %d samples …", len(samples))
        ragas_result = ragas_evaluate(
            dataset=EvaluationDataset(samples=samples),
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            raise_exceptions=False,
        )
        df = ragas_result.to_pandas().reset_index(drop=True)
        df["case_class"] = [m["case_class"] for m in sample_meta]
        df["expected_fail"] = [m["expected_fail"] for m in sample_meta]

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

    if refusal_results:
        handled = sum(r["handled"] for r in refusal_results)
        metrics["refusal_handled"] = round(handled / len(refusal_results), 4)

    logger.info("Headline metrics: %s", metrics)
    logger.info("By case_class: %s", metrics_by_case_class)

    output = {
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset": _repo_relative(dataset_path),
        "model": OPENAI_MODEL,
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
    }
    output_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info("Results written to %s", output_path)


def _repo_relative(path: Path) -> str:
    """Path relative to the repo root when it lives inside it, else the bare name.

    Keeps ``results.json`` free of machine-specific absolute paths.
    """
    try:
        return str(path.resolve().relative_to(_PROJECT_ROOT))
    except ValueError:
        return path.name


_REFUSAL_MARKER = "could not find relevant information"

_RAGAS_COLS = ("faithfulness", "answer_relevancy", "context_recall", "context_precision")


def _is_refusal(answer: str) -> bool:
    """True if the answer is the pipeline's grounded-refusal response."""
    return _REFUSAL_MARKER in answer.lower()


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
    args = parser.parse_args()

    if not args.dataset.exists():
        logger.error("Dataset not found: %s", args.dataset)
        sys.exit(1)

    evaluate(args.dataset, args.output)


if __name__ == "__main__":
    main()
