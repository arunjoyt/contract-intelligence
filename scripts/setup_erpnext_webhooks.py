#!/usr/bin/env python3
"""Idempotently create/update the ERPNext Webhook records for incremental re-indexing.

The RAG stack re-indexes a Contract / Terms and Conditions document whenever
ERPNext fires a webhook at ``POST {PUBLIC_API_URL}/webhook/erpnext``. This script
creates the five records that cover the relevant events (see ``WEBHOOKS`` below and
``docs/DEPLOYMENT.md`` § "ERPNext Webhook Setup"), or updates them in place if they
already exist — so it is safe to re-run after a URL change or a secret rotation.

``contract-on-update-after-submit`` matters: a Desk edit to an ``allow_on_submit``
field (e.g. ``is_signed``) on an already-submitted Contract fires
``on_update_after_submit``, *not* ``on_update`` — missing this webhook silently
skips re-indexing those edits (issue #96).

Usage
-----
    python scripts/setup_erpnext_webhooks.py                # create/update all five
    python scripts/setup_erpnext_webhooks.py --dry-run      # print the plan, write nothing
    python scripts/setup_erpnext_webhooks.py --verify       # report drift, write nothing
    python scripts/setup_erpnext_webhooks.py --url http://127.0.0.1:8000/webhook/erpnext

Reads ``ERPNEXT_URL`` / ``ERPNEXT_API_KEY`` / ``ERPNEXT_API_SECRET`` /
``WEBHOOK_SECRET`` from the environment (``.env`` is loaded automatically). The
request URL defaults to ``{PUBLIC_API_URL}/webhook/erpnext``; override with
``--url`` for an Option A / loopback deployment.
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

# (Webhook name, doctype, docevent). Single source of truth — docs/DEPLOYMENT.md
# points here. `on_submit` is not valid for Terms and Conditions (not submittable).
WEBHOOKS: list[tuple[str, str, str]] = [
    ("contract-on-submit", "Contract", "on_submit"),
    ("contract-on-update", "Contract", "on_update"),
    ("contract-on-update-after-submit", "Contract", "on_update_after_submit"),
    ("contract-on-cancel", "Contract", "on_cancel"),
    ("terms-on-update", "Terms and Conditions", "on_update"),
]

_WEBHOOK_JSON = '{"doctype": "{{ doc.doctype }}", "docname": "{{ doc.name }}"}'

# Fields we manage and can read back to detect drift. `webhook_secret` is a
# Password field — ERPNext returns it masked, so it is written but never compared.
_MANAGED_FIELDS = (
    "webhook_doctype",
    "webhook_docevent",
    "request_url",
    "request_method",
    "request_structure",
    "enabled",
    "enable_security",
    "webhook_json",
)


def desired_record(name: str, doctype: str, event: str, url: str, secret: str) -> dict:
    """The full Webhook doc we want ERPNext to hold for this event."""
    return {
        "doctype": "Webhook",
        "name": name,
        "webhook_doctype": doctype,
        "webhook_docevent": event,
        "request_url": url,
        "request_method": "POST",
        "request_structure": "JSON",
        "enabled": 1,
        "enable_security": 1,
        "webhook_secret": secret,
        "webhook_json": _WEBHOOK_JSON,
    }


def plan_action(existing: dict | None, desired: dict) -> str:
    """`create` if absent, `update` if any managed field drifted, else `ok`.

    Pure — no I/O. `webhook_secret` is excluded from `_MANAGED_FIELDS` because
    ERPNext returns it masked; it is re-sent on every create/update regardless.
    """
    if existing is None:
        return "create"
    for field in _MANAGED_FIELDS:
        if str(existing.get(field, "")) != str(desired[field]):
            return "update"
    return "ok"


def _client() -> httpx.Client:
    base = _env_or_die("ERPNEXT_URL").rstrip("/")
    key = _env_or_die("ERPNEXT_API_KEY")
    secret = _env_or_die("ERPNEXT_API_SECRET")
    return httpx.Client(
        base_url=base,
        headers={"Authorization": f"token {key}:{secret}"},
        timeout=20,
    )


def _fetch(client: httpx.Client, name: str) -> dict | None:
    r = client.get(f"/api/resource/Webhook/{name}")
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()["data"]


def _write(client: httpx.Client, existing: dict | None, desired: dict) -> None:
    body = {k: v for k, v in desired.items() if k not in ("doctype", "name")}
    if existing is None:
        r = client.post("/api/resource/Webhook", json={**body, "name": desired["name"]})
    else:
        r = client.put(f"/api/resource/Webhook/{desired['name']}", json=body)
    r.raise_for_status()


def _env_or_die(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        logger.error("%s is not set", name)
        sys.exit(1)
    return val


def _resolve_url(cli_url: str | None) -> str:
    if cli_url:
        return cli_url
    public = os.environ.get("PUBLIC_API_URL", "").rstrip("/")
    if not public:
        logger.error("neither --url nor PUBLIC_API_URL is set")
        sys.exit(1)
    return f"{public}/webhook/erpnext"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--url", help="webhook request URL (default: {PUBLIC_API_URL}/webhook/erpnext)"
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    parser.add_argument(
        "--verify", action="store_true", help="report drift/missing records, write nothing"
    )
    args = parser.parse_args()

    url = _resolve_url(args.url)
    secret = _env_or_die("WEBHOOK_SECRET")
    read_only = args.dry_run or args.verify
    logger.info("target URL: %s%s", url, "  (read-only)" if read_only else "")

    counts = {"create": 0, "update": 0, "ok": 0}
    with _client() as client:
        for name, doctype, event in WEBHOOKS:
            desired = desired_record(name, doctype, event, url, secret)
            try:
                existing = _fetch(client, name)
            except httpx.HTTPError as exc:
                logger.error("%s: could not read (%s)", name, exc)
                return 1

            action = plan_action(existing, desired)
            counts[action] += 1
            if action == "ok":
                logger.info("%s: up to date", name)
                continue
            if read_only:
                logger.info("%s: needs %s", name, action)
                continue
            _write(client, existing, desired)
            logger.info("%s: %sd", name, action)

    logger.info(
        "done: %d created, %d updated, %d already correct",
        counts["create"],
        counts["update"],
        counts["ok"],
    )
    if args.verify and (counts["create"] or counts["update"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
