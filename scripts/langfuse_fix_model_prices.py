"""Idempotently correct stale model prices in a self-hosted Langfuse project.

Self-hosted Langfuse 2.x ships a bundled model-price table that still prices
``gpt-4o`` at its **mid-2024 launch rate** ($5.00 / $15.00 per 1M tokens) rather
than the current list price ($2.50 / $10.00, in effect since 2024-08-06). Every
``totalCost`` / ``calculatedTotalCost`` in the Langfuse UI and public API is then
~1.9x too high for ``gpt-4o`` generations (token counts are fine — only the price
multiplier is wrong). ``gpt-4o-mini`` and ``text-embedding-3-small`` are already
correct in the bundled table. See issue #137.

This script creates **project-level** price overrides (which take precedence over
Langfuse-managed prices) via ``POST /api/public/models``, so cost in the Langfuse
UI matches ``evaluation/results.json``'s ``costs`` block and ``docs/BENCHMARKS.md``.
Rates here are the same current OpenAI list prices as
``evaluation/evaluate.py::_PRICE_PER_1M_TOKENS`` — keep the two in sync.

Idempotent: re-running detects an existing correct override and does nothing;
a stale or duplicated override is deleted and recreated. Only touches models whose
Langfuse-managed price actually differs from the target (unless ``--force``).

Run once per deployment, right after the stack is up and ``LANGFUSE_*`` is in
``.env`` (see ``docs/CLIENT_DEPLOYMENT_RUNBOOK.md`` Phase 3), and again after any
Langfuse version bump or a restore into a fresh Langfuse database.

Usage
-----
    python scripts/langfuse_fix_model_prices.py [--dry-run] [--force]

Reads ``LANGFUSE_HOST`` / ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` from
the environment (``.env`` is loaded automatically).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Per-1M-token USD list prices (input, output). Mirror of
# evaluation/evaluate.py::_PRICE_PER_1M_TOKENS for the models this project runs.
# `pattern` is the exact-match regex Langfuse uses to bind a generation's `model`
# string to a price row; `tokenizer` populates the row's tokenizer config.
_TARGET_MODELS: list[dict] = [
    {
        "modelName": "gpt-4o",
        "pattern": r"(?i)^(gpt-4o)$",
        "input_per_1m": 2.50,
        "output_per_1m": 10.00,
        "tokenizer": "gpt-4o",
    },
]


def _per_1m(price: float | None) -> float:
    """A Langfuse price row stores $/token; report it as $/1M tokens."""
    return round((price or 0.0) * 1_000_000, 6)


def _row_current(m: dict, spec: dict) -> bool:
    """True if this price row already carries the target $/1M rates."""
    return (
        _per_1m(m.get("inputPrice")) == spec["input_per_1m"]
        and _per_1m(m.get("outputPrice")) == spec["output_per_1m"]
    )


def _env_or_die(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        logger.error("%s is not set (need it to reach the Langfuse API)", name)
        sys.exit(1)
    return val


def _client() -> httpx.Client:
    host = _env_or_die("LANGFUSE_HOST").rstrip("/")
    pk = _env_or_die("LANGFUSE_PUBLIC_KEY")
    sk = _env_or_die("LANGFUSE_SECRET_KEY")
    return httpx.Client(base_url=host, auth=(pk, sk), timeout=20)


def _all_models(client: httpx.Client) -> list[dict]:
    models: list[dict] = []
    page = 1
    while True:
        r = client.get("/api/public/models", params={"limit": 100, "page": page})
        r.raise_for_status()
        body = r.json()
        models.extend(body.get("data", []))
        meta = body.get("meta", {})
        if page >= meta.get("totalPages", page):
            break
        page += 1
    return models


def _create_override(client: httpx.Client, spec: dict) -> str:
    payload = {
        "modelName": spec["modelName"],
        "matchPattern": spec["pattern"],
        "unit": "TOKENS",
        "inputPrice": spec["input_per_1m"] / 1_000_000,
        "outputPrice": spec["output_per_1m"] / 1_000_000,
    }
    if spec.get("tokenizer"):
        payload["tokenizerId"] = "openai"
        payload["tokenizerConfig"] = {
            "tokenizerModel": spec["tokenizer"],
            "tokensPerMessage": 3,
            "tokensPerName": 1,
        }
    r = client.post("/api/public/models", json=payload)
    r.raise_for_status()
    return r.json()["id"]


def plan_changes(spec: dict, existing: list[dict], *, force: bool = False) -> dict:
    """Pure decision step: given the project's current model rows, decide what to do
    for ``spec``. Returns ``{"status", "delete", "create"}`` where ``status`` is
    ``'ok'`` / ``'skipped'`` / ``'fixed'``, ``delete`` is a list of model-id strings,
    and ``create`` is a bool. No I/O — unit-tested directly.
    """
    rows = [m for m in existing if m.get("matchPattern") == spec["pattern"]]
    overrides = [m for m in rows if not m.get("isLangfuseManaged")]
    managed = [m for m in rows if m.get("isLangfuseManaged")]

    good = [m for m in overrides if _row_current(m, spec)]
    stale = [m for m in overrides if not _row_current(m, spec)]

    # Steady state: exactly one correct override, nothing stale.
    if len(good) == 1 and not stale and not force:
        return {"status": "ok", "delete": [], "create": False}

    # No override at all, and Langfuse's own bundled price is already right.
    if not overrides and any(_row_current(m, spec) for m in managed) and not force:
        return {"status": "skipped", "delete": [], "create": False}

    delete = [m["id"] for m in stale + good[1:]]  # stale rows + duplicate correct rows
    return {"status": "fixed", "delete": delete, "create": not good}


def _sync_model(
    client: httpx.Client, spec: dict, existing: list[dict], *, dry_run: bool, force: bool
) -> str:
    name = spec["modelName"]
    rate = f"${spec['input_per_1m']:.2f}/${spec['output_per_1m']:.2f} per 1M"

    for m in existing:
        if (
            m.get("matchPattern") == spec["pattern"]
            and m.get("isLangfuseManaged")
            and not _row_current(m, spec)
        ):
            logger.info(
                "%s: Langfuse-managed price is stale ($%s/$%s per 1M)",
                name,
                _per_1m(m.get("inputPrice")),
                _per_1m(m.get("outputPrice")),
            )

    plan = plan_changes(spec, existing, force=force)
    if plan["status"] == "ok":
        logger.info("%s: override already current (%s) — no change", name, rate)
        return "ok"
    if plan["status"] == "skipped":
        logger.info("%s: Langfuse-managed price already current — no override needed", name)
        return "skipped"

    prefix = "[dry-run] would " if dry_run else ""
    for mid in plan["delete"]:
        if not dry_run:
            client.delete(f"/api/public/models/{mid}").raise_for_status()
        logger.info("%s: %sdelete stale/duplicate override %s", name, prefix, mid)
    if plan["create"]:
        if dry_run:
            logger.info("%s: [dry-run] would create project override -> %s", name, rate)
        else:
            new_id = _create_override(client, spec)
            logger.info("%s: created project override -> %s (id %s)", name, rate, new_id)

    return "fixed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="print what would change, touch nothing"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="create the override even when the Langfuse-managed price is already correct",
    )
    args = parser.parse_args()

    with _client() as client:
        try:
            existing = _all_models(client)
        except httpx.HTTPError as exc:
            logger.error("could not list Langfuse models: %s", exc)
            return 1

        results = [
            _sync_model(client, spec, existing, dry_run=args.dry_run, force=args.force)
            for spec in _TARGET_MODELS
        ]

    fixed = results.count("fixed")
    if args.dry_run and fixed:
        logger.info("dry-run: %d model(s) need correcting — re-run without --dry-run", fixed)
    elif fixed:
        logger.info("done: corrected %d model price(s)", fixed)
    else:
        logger.info("done: all model prices already current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
