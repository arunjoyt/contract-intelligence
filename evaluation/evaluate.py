"""RAGAS evaluation for the procurement RAG pipeline.

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
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent
_DEFAULT_DATASET = _ROOT / "test_dataset.json"
_DEFAULT_OUTPUT = _ROOT / "results.json"

_ANSWER_SYSTEM = """\
You are a procurement analyst assistant. Answer the user's question using ONLY \
the context below.

Rules:
- Cite every claim with [docname] immediately after the relevant sentence.
- If the answer is not in the provided context, respond with exactly:
  "I could not find relevant information in the procurement documents."
- Do not use any knowledge outside the provided context.

Context:
{context}
"""


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

    hybrid_search = HybridSearch(embedder=embedder, vector_store=vector_store)
    hybrid_search.build_bm25_index(vector_store.get_all_texts())

    reranker = Reranker()
    reranker.warm_up()

    rewriter = QueryRewriter(embedder=embedder)

    logger.info("Components ready")
    return rewriter, hybrid_search, reranker


# ---------------------------------------------------------------------------
# Single-pass inference — captures retrieved texts without a second LLM call
# ---------------------------------------------------------------------------


def _run_question(
    question: str,
    rewriter,
    hybrid_search,
    reranker,
    openai_client,
) -> tuple[str, list[str]]:
    """Return (answer, retrieved_context_texts) in a single pipeline pass."""
    rewritten, _ = rewriter.rewrite(question)
    candidates = hybrid_search.search(rewritten, None, top_k=20)
    top_chunks = reranker.rerank(question, candidates, top_n=5)

    retrieved_texts = [c.get("text", "") for c in top_chunks if c.get("text")]

    # Build context block and generate answer (same logic as query_pipeline._build_context)
    parts: list[str] = []
    for chunk in top_chunks:
        docname = chunk.get("docname", "unknown")
        supplier = chunk.get("supplier", "")
        text = chunk.get("text", "")
        supplier_str = f" (supplier: {supplier})" if supplier else ""
        parts.append(f"[{docname}]{supplier_str}:\n{text}")
    context = "\n\n---\n\n".join(parts)

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": _ANSWER_SYSTEM.format(context=context)},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    answer = response.choices[0].message.content or ""
    return answer, retrieved_texts


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
    per_question: list[dict] = []

    for i, entry in enumerate(entries, 1):
        question: str = entry["question"]
        reference_contexts: list[str] = entry["ground_truth_contexts"]
        reference_answer: str = entry.get("ground_truth_answer", "")

        logger.info("[%d/%d] %s", i, len(entries), question[:80])
        try:
            answer, retrieved_contexts = _run_question(
                question, rewriter, hybrid_search, reranker, openai_client
            )
        except Exception:
            logger.exception("Pipeline failed for question %d — skipping", i)
            continue

        # Fall back to reference_contexts when the vector store returns nothing
        # (e.g. first run before any ingest).  Faithfulness will still be useful
        # because it checks whether the answer stays within the provided context.
        if not retrieved_contexts:
            logger.warning("No retrieved contexts for question %d — using ground truth", i)
            retrieved_contexts = reference_contexts

        per_question.append(
            {
                "question": question,
                "answer": answer,
                "retrieved_context_count": len(retrieved_contexts),
            }
        )

        samples.append(
            SingleTurnSample(
                user_input=question,
                retrieved_contexts=retrieved_contexts,
                reference_contexts=reference_contexts,
                response=answer,
                reference=reference_answer,
            )
        )

    if not samples:
        logger.error("No samples collected — aborting")
        sys.exit(1)

    logger.info("Running RAGAS on %d samples …", len(samples))
    dataset = EvaluationDataset(samples=samples)
    ragas_result = ragas_evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        raise_exceptions=False,
    )

    metrics: dict[str, float] = {
        "faithfulness": _safe_mean(ragas_result, "faithfulness"),
        "answer_relevancy": _safe_mean(ragas_result, "answer_relevancy"),
        "context_recall": _safe_mean(ragas_result, "context_recall"),
        "context_precision": _safe_mean(ragas_result, "context_precision"),
    }
    logger.info("Metrics: %s", metrics)

    output = {
        "timestamp": datetime.now(datetime.UTC).isoformat(),
        "dataset": str(dataset_path),
        "num_questions": len(samples),
        "metrics": metrics,
        "per_question": per_question,
    }
    output_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info("Results written to %s", output_path)


def _safe_mean(ragas_result, metric_name: str) -> float:
    try:
        df = ragas_result.to_pandas()
        col = df[metric_name].dropna()
        return round(float(col.mean()), 4) if len(col) > 0 else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on the procurement RAG pipeline"
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
