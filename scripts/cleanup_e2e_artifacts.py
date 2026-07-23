#!/usr/bin/env python3
"""One-time (and re-runnable) cleanup of E2E test Contracts that leaked into
the live ERPNext site and the production Qdrant collection.

Background (see issue #95): tests/e2e/conftest.py's old cleanup helper only
cancelled its test Contract, never deleted it or its Qdrant points -- every
live RUN_E2E=1 run left its test Contract permanently cancelled in ERPNext
*and* permanently indexed in the same Qdrant collection real /query requests
hit. The fixture-level bug is fixed (conftest.py's `cleanup_contract`); this
script sweeps up what already accumulated before that fix landed.

Identification: E2E test Contracts are identified by distinctive markers
that only ever appear in `contract_terms` planted by tests/e2e/*.py --
"TRACKING-CODE", "PDF-CODE", "CANCEL-CODE", "E2E test", "debug" -- never
demo-data content. This same marker convention is the guard against
accidentally sweeping up real data in future runs of this script; if test
authors add a new planted-fact prefix, add it to MARKERS below.

Usage:
    python scripts/cleanup_e2e_artifacts.py            # dry run (default) --
                                                          lists matches only
    python scripts/cleanup_e2e_artifacts.py --apply     # actually delete
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.erpnext_client import ERPNextClient

MARKERS = ["TRACKING-CODE", "PDF-CODE", "CANCEL-CODE", "E2E test", "debug"]


async def find_candidates() -> list[dict]:
    async with ERPNextClient() as client:
        contracts = await client.get_list(
            "Contract",
            fields=["name", "contract_terms", "docstatus"],
            limit=0,
        )
    return [
        c
        for c in contracts
        if any(marker in (c.get("contract_terms") or "") for marker in MARKERS)
    ]


def cancel_and_delete(docname: str, docstatus: int, erpnext_url: str, headers: dict) -> None:
    if docstatus == 1:
        httpx.post(
            f"{erpnext_url}/api/method/frappe.client.cancel",
            data={"doctype": "Contract", "name": docname},
            headers=headers,
            timeout=10,
        ).raise_for_status()
    httpx.delete(
        f"{erpnext_url}/api/resource/Contract/{docname}", headers=headers, timeout=10
    ).raise_for_status()


def delete_qdrant_points(docname: str, qdrant_url: str, collection: str) -> None:
    httpx.post(
        f"{qdrant_url}/collections/{collection}/points/delete",
        json={"filter": {"must": [{"key": "docname", "match": {"value": docname}}]}},
        timeout=10,
    ).raise_for_status()


async def main() -> None:
    import os

    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    args = parser.parse_args()

    candidates = await find_candidates()
    print(f"Found {len(candidates)} candidate E2E test Contract(s):")
    for c in candidates:
        print(f"  {c['name']} (docstatus={c['docstatus']})")

    if not candidates:
        return
    if not args.apply:
        print("\nDry run only -- rerun with --apply to delete these Contracts and their")
        print("Qdrant points.")
        return

    erpnext_url = os.environ["ERPNEXT_URL"]
    headers = {
        "Authorization": f"token {os.environ['ERPNEXT_API_KEY']}:{os.environ['ERPNEXT_API_SECRET']}"
    }
    qdrant_url = os.environ["QDRANT_URL"]
    collection = os.environ["QDRANT_COLLECTION"]

    for c in candidates:
        docname = c["name"]
        try:
            cancel_and_delete(docname, c["docstatus"], erpnext_url, headers)
            delete_qdrant_points(docname, qdrant_url, collection)
            print(f"  deleted {docname}")
        except Exception as exc:
            print(f"  FAILED {docname}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
