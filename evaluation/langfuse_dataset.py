"""Shared identity for syncing evaluation/test_dataset.json with a Langfuse Dataset.

Used by both push_dataset.py (the one-off idempotent sync) and evaluate.py (per-run
score/trace logging) so the two agree on dataset name and item identity without
either importing the other.

For a client deployment, LANGFUSE_EVAL_DATASET/LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY/
LANGFUSE_HOST point at that client's own Langfuse project -- their golden-set questions
and scores never need to be committed to this repo (#118). This repo's own demo
test_dataset.json stays committed and is pushed into its own project's dataset the
same way.
"""

from __future__ import annotations

import hashlib
import os

DATASET_NAME = os.environ.get("LANGFUSE_EVAL_DATASET", "contract-intelligence-golden-set")


def dataset_item_id(question: str) -> str:
    """Stable id derived from the question text.

    Lets push_dataset.py re-push the dataset and upsert existing items instead of
    duplicating them, and lets evaluate.py look up the matching Langfuse dataset
    item to link a run without re-reading test_dataset.json's row order (which
    isn't a stable identity -- entries get inserted/reordered over time).
    """
    return hashlib.sha1(question.encode()).hexdigest()[:16]


def build_client():
    """Return a configured Langfuse client, or None if credentials aren't set.

    Mirrors api/main.py's tracing-optional pattern -- callers proceed without
    Langfuse (offline path) rather than failing when credentials are absent.
    """
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return None

    from langfuse import Langfuse

    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=os.environ.get("LANGFUSE_HOST"),
    )
