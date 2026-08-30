"""Shared query-pipeline constants.

These define the generation contract and the retrieval budget for the query
pipeline. ``evaluation/evaluate.py`` imports the same values so a baseline run
exercises the exact prompt and top-k / top-n the live pipeline uses (#110) --
before this module they were hand-copied into evaluate.py and had drifted.

This module deliberately has no heavy imports, so it can be pulled in from
anywhere (including evaluate.py's module scope) without loading openai /
sentence-transformers / qdrant-client.
"""

from __future__ import annotations

import json
import os
from typing import Any

# Hybrid-search candidate pool size, then the cross-encoder rerank budget the
# generator actually sees. Env-overridable so a tuning pass (#113) can sweep them
# without editing this file between runs; the defaults are the shipped values.
RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "20"))
RERANK_TOP_N = int(os.environ.get("RERANK_TOP_N", "5"))


def _json_env(name: str, default: str) -> Any:
    """Parse a JSON-valued env var, failing fast with a pointed message.

    Used for the metadata-filter vocabulary below -- the one part of the pipeline
    that is genuinely per-client (a client with a customized Contract doctype or
    different status values), so it must be configurable without editing source.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return json.loads(default)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON: {exc}") from exc


# Heuristic metadata-filter vocabulary for _extract_filters() in query_pipeline.py
# -- a keyword in the question text pins source_doctype / status before retrieval.
# Per-client config (see docs/CLIENT_DEPLOYMENT_RUNBOOK.md § Phase 5): override
# when the client uses custom doctype names or status values. The status form is
# a {keyword: status-value} map, not a list, so a client can point e.g. "expired"
# at "Inactive". Defaults reproduce the previous hardcoded behaviour exactly.
METADATA_FILTER_DOCTYPE_KEYWORDS: dict[str, list[str]] = _json_env(
    "METADATA_FILTER_DOCTYPE_KEYWORDS",
    '{"Contract": ["contract"], "Terms and Conditions": ["terms and conditions"]}',
)
METADATA_FILTER_STATUS_KEYWORDS: dict[str, str] = _json_env(
    "METADATA_FILTER_STATUS_KEYWORDS",
    '{"cancelled": "Cancelled", "active": "Active", "unsigned": "Unsigned"}',
)

# Generation call.
GENERATION_MAX_TOKENS = 1024

ANSWER_SYSTEM_PROMPT = """\
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

# Metadata fields surfaced into the per-chunk context block, in order. Some
# doctypes carry fields (status, linked_doctype/linked_docname) that never make
# it into the chunk's own text, so without this they'd be invisible to
# generation even though retrieval has them.
CONTEXT_META_FIELDS = (
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
