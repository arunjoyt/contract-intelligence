"""One-off, idempotent sync of evaluation/test_dataset.json into a Langfuse Dataset.

Run this once (and again whenever test_dataset.json changes) before evaluate.py, so
evaluate.py's per-question traces link to a dataset item and group into a comparable
Experiment run in the Langfuse UI. Re-running is safe: items upsert by a stable id
derived from the question text (see langfuse_dataset.dataset_item_id), so editing or
re-pushing the same file doesn't create duplicates.

For a client deployment, point LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY/LANGFUSE_HOST at
their own Langfuse project and run this against their own (uncommitted) dataset file --
their golden-set questions never need to enter this repo's git history (#118).

Usage
-----
    python evaluation/push_dataset.py [--dataset PATH] [--name NAME]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent
_PROJECT_ROOT = _ROOT.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.langfuse_dataset import DATASET_NAME, build_client, dataset_item_id  # noqa: E402

_DEFAULT_DATASET = _ROOT / "test_dataset.json"


def push(dataset_path: Path = _DEFAULT_DATASET, dataset_name: str = DATASET_NAME) -> None:
    langfuse = build_client()
    if langfuse is None:
        logger.error(
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set -- nothing to push to"
        )
        sys.exit(1)

    entries: list[dict] = json.loads(dataset_path.read_text())
    logger.info("Loaded %d entries from %s", len(entries), dataset_path)

    langfuse.create_dataset(
        name=dataset_name,
        description="Contract Intelligence golden-set eval questions",
    )

    for entry in entries:
        question: str = entry["question"]
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            id=dataset_item_id(question),
            input={"question": question},
            expected_output=entry.get("ground_truth_answer", ""),
            metadata={
                "case_class": entry.get("case_class", "uncategorized"),
                "capability": entry.get("capability", ""),
                "expected_fail": bool(entry.get("expected_fail", False)),
                "split": entry.get("split", ""),
                "ground_truth_contexts": entry.get("ground_truth_contexts", []),
            },
        )

    langfuse.flush()
    logger.info("Pushed %d items to Langfuse dataset %r", len(entries), dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--name", default=DATASET_NAME)
    args = parser.parse_args()
    push(args.dataset, args.name)
