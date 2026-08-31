#!/usr/bin/env python3
"""Draft candidate eval questions from a live Qdrant collection (#112).

Reads every indexed chunk from the configured Qdrant collection, then asks an
LLM to draft candidate ``question`` / ``ground_truth_answer`` triples per
``case_class``, each grounded in real ``docname:chunk_index`` keys taken straight
from the collection. Writes a **human-review file** — nothing here is a finished
dataset entry. A person must read every candidate, verify the answer against the
real document, fix the framing, set a dev/test ``split``, and only then paste it
into ``evaluation/test_dataset.json``.

Why generate rather than hand-author from scratch: at ~150 questions across 7
slices, drafting is the slow part and an LLM over the real corpus gets a usable
first pass. The same generator is the per-client onboarding tool (#127) — point
it at the client's own collection to draft their validation set without hand-
authoring ~150 questions per client.

This calls the OpenAI API (one chat completion per ``case_class``). It is a
deliberate, opt-in spend — run it only when you mean to.

Usage
-----
    # draft ~8 candidates for every case_class into evaluation/candidates.review.json
    python scripts/generate_eval_set.py

    # just the two slices you want to deepen, 15 each
    python scripts/generate_eval_set.py --case-class refusal --case-class temporal --per-class 15

    # per-client (#127): the staging stack's collection, output under the gitignored client dir
    python scripts/generate_eval_set.py --collection acme_staging \
        --output evaluation/client/acme_candidates.review.json

Per #118, a real client's questions and contract text must never be committed to this
repo — write per-client output under evaluation/client/ (gitignored).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import REWRITE_MODEL  # noqa: E402
from pipeline.constants import CONTEXT_META_FIELDS  # noqa: E402

_DEFAULT_OUTPUT = _PROJECT_ROOT / "evaluation" / "candidates.review.json"

# The slice taxonomy the reference set is built on. Keep in step with
# docs/ARCHITECTURE.md § Evaluation and docs/PIPELINE_TUNING.md.
CASE_CLASSES = (
    "showcase",
    "disambiguation",
    "precision-multi",
    "semantic-no-anchor",
    "refusal",
    "aggregation",
    "temporal",
)

_EXPECTED_FAIL_CLASSES = ("aggregation", "temporal")

_SLICE_BRIEF = {
    "showcase": (
        "One clear question per retrieval capability (exact-term / BM25, paraphrase / "
        "vector, cross-document synthesis, abstract 'what recourse' phrasing, plain-"
        "language vs an exact status value, PDF-only clause). Easy by construction — a "
        "distinctive supplier token, an answer stated almost verbatim in one chunk."
    ),
    "disambiguation": (
        "A supplier with two or more contracts where the near-duplicate must NOT be "
        "cited — current vs superseded, primary vs backup, live vs cancelled, base vs "
        "addendum. The answer names the right docname and says why the other is wrong."
    ),
    "precision-multi": (
        "'Which contracts / suppliers do X' — retrieval must return ALL and ONLY the "
        "matches against a top-5 budget. List every matching docname; note any close-"
        "but-excluded contract."
    ),
    "semantic-no-anchor": (
        "No supplier name and no doc number in the question — BM25 has nothing to grab, "
        "the dense side must carry it. Phrase it the way a buyer would ('what happens "
        "if we pay late', 'who fixes broken leased kit')."
    ),
    "refusal": (
        "The honest answer is 'I could not find relevant information in the contract "
        "documents.' Two flavours: (a) a supplier that does not exist in the corpus, "
        "(b) a clause type genuinely absent from a real, named contract. "
        "ground_truth_context_keys MUST be empty for every refusal candidate."
    ),
    "aggregation": (
        "Counting / averaging / enumerating across the whole corpus — a known "
        "limitation (top-5 budget drops records, no arithmetic step). expected_fail. "
        "The answer states the true result AND why the pipeline cannot produce it."
    ),
    "temporal": (
        "'active now' / 'expires within N months' / 'most recent' / 'started in YEAR' — "
        "a known limitation (no date arithmetic in the query path, status frozen at "
        "ingest). expected_fail. The answer gives the date-based truth AND why the "
        "pipeline cannot compute it reliably."
    ),
}


def _load_corpus(collection: str | None) -> list[dict]:
    """Every indexed chunk's payload, from the configured Qdrant collection."""
    from retrieval.vector_store import VectorStore

    vector_store = VectorStore(collection=collection) if collection else VectorStore()
    vector_store.ensure_collection()
    payloads = vector_store.get_all_texts()
    if not payloads:
        logger.error(
            "Qdrant collection %r is empty — run a full ingest first so the "
            "generator has a real corpus to ground questions in.",
            getattr(vector_store, "_collection", collection),
        )
        sys.exit(1)
    return payloads


def _chunk_key(payload: dict) -> str:
    return f"{payload.get('docname')}:{payload.get('chunk_index')}"


def _frame(payload: dict) -> str:
    """``[docname] (metadata): text`` — the exact shape evaluate.py feeds RAGAS
    (``evaluation/evaluate.py:_frame_chunk``) so a reviewed candidate can be
    pasted into ``ground_truth_contexts`` unchanged."""
    docname = payload.get("docname", "unknown")
    meta_bits = [
        f"{key}: {value}" for key in CONTEXT_META_FIELDS if (value := payload.get(key))
    ]
    meta_str = f" ({'; '.join(meta_bits)})" if meta_bits else ""
    return f"[{docname}]{meta_str}: {payload.get('text', '')}".strip()


def _corpus_digest(payloads: list[dict]) -> str:
    """The whole corpus as a flat, key-addressed list for the prompt."""
    by_key = {_chunk_key(p): _frame(p) for p in payloads if p.get("text")}
    return "\n\n".join(f"{key}\n{framed}" for key, framed in sorted(by_key.items()))


def _prompt(case_class: str, per_class: int, corpus_digest: str) -> str:
    is_ef = case_class in _EXPECTED_FAIL_CLASSES
    keys_line = (
        '- "ground_truth_context_keys": exact `docname:chunk_index` keys whose text '
        "supports the answer. Empty list for refusal candidates."
    )
    answer_line = (
        '- "ground_truth_answer": a correct grounded answer with [docname] citations. '
        'For refusal candidates use exactly: "I could not find relevant information in '
        'the contract documents."'
    )
    notes_line = (
        '- "notes_for_reviewer": what a human must double-check before accepting '
        '(e.g. "confirm no price-escalation clause exists in CON-2026-00026").'
    )
    rules = [
        "Rules:",
        "- Every candidate must be answerable (or correctly refusable) from the corpus "
        "below — no outside knowledge.",
        keys_line,
        answer_line,
        '- "capability": a short label for what this specific question exercises.',
        notes_line,
        f"- Vary the suppliers and clause types across the {per_class} candidates.",
    ]
    if is_ef:
        rules.append('- Set "expected_fail": true on every candidate (known limitation).')
    out_keys = (
        "question, ground_truth_answer, ground_truth_context_keys, capability, "
        "notes_for_reviewer"
    )
    if is_ef:
        out_keys += ", expected_fail"
    return "\n".join(
        [
            "You are drafting candidate evaluation questions for a contract-QA RAG system.",
            "Below is the ENTIRE indexed corpus, one chunk per entry, addressed by a",
            "`docname:chunk_index` key.",
            "",
            f'Draft {per_class} candidate questions for the "{case_class}" slice.',
            "",
            f"Slice brief: {_SLICE_BRIEF[case_class]}",
            "",
            *rules,
            "",
            "Return ONLY a JSON array of objects with keys:",
            out_keys,
            "",
            "CORPUS:",
            corpus_digest,
        ]
    )


def _generate_for_class(
    client, model: str, case_class: str, per_class: int, digest: str
) -> list[dict]:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": _prompt(case_class, per_class, digest)}],
        temperature=0.4,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    parsed = json.loads(raw)
    # response_format=json_object forces an object wrapper; accept a bare list too.
    candidates = (
        parsed
        if isinstance(parsed, list)
        else next((v for v in parsed.values() if isinstance(v, list)), [])
    )
    logger.info("  %s: %d candidates", case_class, len(candidates))
    return candidates


def _resolve(candidate: dict, framed_by_key: dict[str, str], case_class: str) -> dict:
    """Attach the framed context text for each key and flag anything unresolved."""
    keys = candidate.get("ground_truth_context_keys", []) or []
    contexts, unknown = [], []
    for key in keys:
        if key in framed_by_key:
            contexts.append(framed_by_key[key])
        else:
            unknown.append(key)
    entry = {
        "_reviewed": False,
        "case_class": case_class,
        "split": None,
        "capability": candidate.get("capability", ""),
        "question": candidate.get("question", ""),
        "ground_truth_contexts": contexts,
        "ground_truth_answer": candidate.get("ground_truth_answer", ""),
        "_context_keys": keys,
        "_notes_for_reviewer": candidate.get("notes_for_reviewer", ""),
    }
    if case_class in _EXPECTED_FAIL_CLASSES:
        entry["expected_fail"] = bool(candidate.get("expected_fail", True))
    if unknown:
        entry["_UNRESOLVED_KEYS"] = unknown
    if case_class == "refusal" and contexts:
        entry["_WARNING"] = "refusal candidate has non-empty contexts — clear before accepting"
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--case-class",
        action="append",
        choices=CASE_CLASSES,
        dest="case_classes",
        help="Slice to draft for (repeatable). Default: all seven.",
    )
    parser.add_argument(
        "--per-class", type=int, default=8, help="Candidates per slice (default 8)."
    )
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument(
        "--model",
        default=REWRITE_MODEL,
        help=f"OpenAI model for drafting (default: REWRITE_MODEL = {REWRITE_MODEL}).",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Qdrant collection (default: QDRANT_COLLECTION env / VectorStore default).",
    )
    args = parser.parse_args()

    # #118 guardrail: a non-default collection means a real client corpus — its drafted
    # questions and chunk text must not land somewhere git-tracked.
    non_default_collection = args.collection or os.environ.get("QDRANT_COLLECTION") not in (
        None,
        "contract",
    )
    out_under_client_dir = "client" in args.output.parts
    if non_default_collection and not out_under_client_dir:
        logger.warning(
            "Non-default collection but --output is %s, not under evaluation/client/. "
            "Per #118, per-client output must stay gitignored — redirect it.",
            args.output,
        )

    case_classes = args.case_classes or list(CASE_CLASSES)

    payloads = _load_corpus(args.collection)
    framed_by_key = {_chunk_key(p): _frame(p) for p in payloads if p.get("text")}
    digest = _corpus_digest(payloads)
    logger.info(
        "Corpus: %d chunks across %d documents",
        len(framed_by_key),
        len({p.get("docname") for p in payloads}),
    )

    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    review: list[dict] = []
    for case_class in case_classes:
        logger.info("Drafting %r …", case_class)
        drafted = _generate_for_class(
            client, args.model, case_class, args.per_class, digest
        )
        for candidate in drafted:
            review.append(_resolve(candidate, framed_by_key, case_class))

    args.output.write_text(json.dumps(review, indent=2))
    unresolved = sum("_UNRESOLVED_KEYS" in e for e in review)
    logger.info(
        "Wrote %d candidates to %s (%d with unresolved context keys — fix by hand)",
        len(review),
        args.output,
        unresolved,
    )
    logger.info(
        "NEXT: review every entry, verify the answer against the real document, set "
        '"split" to "dev" or "test", drop the _-prefixed helper keys, and merge into '
        "evaluation/test_dataset.json. Then re-run push_dataset.py."
    )


if __name__ == "__main__":
    main()
