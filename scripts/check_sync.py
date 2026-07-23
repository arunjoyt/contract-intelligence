#!/usr/bin/env python3
"""Check whether ERPNext (Contract, Terms and Conditions) and Qdrant agree.

Compares docnames present in each ERPNext doctype against the docnames
indexed in Qdrant, and cross-checks each document's chunk count against its
own ``total_chunks`` payload value. Read-only — makes no writes to either
system.

Usage:
    python scripts/check_sync.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.erpnext_client import ERPNextClient
from retrieval.vector_store import VectorStore

DOCTYPES = ["Contract", "Terms and Conditions"]


async def fetch_erpnext_docnames() -> dict[str, set[str]]:
    async with ERPNextClient() as client:
        return {
            dt: {d["name"] for d in await client.get_list(dt, fields=["name"], limit=0)}
            for dt in DOCTYPES
        }


def group_qdrant_payloads(payloads: list[dict]) -> dict[str, dict[str, set[int]]]:
    by_doctype: dict[str, dict[str, set[int]]] = {}
    for p in payloads:
        doctype_chunks = by_doctype.setdefault(p.get("source_doctype"), {})
        doctype_chunks.setdefault(p.get("docname"), set()).add(p.get("chunk_index"))
    return by_doctype


def check_chunk_counts(docname: str, indices: set[int], payloads: list[dict]) -> str | None:
    expected_total = next(
        (p["total_chunks"] for p in payloads if p.get("docname") == docname), None
    )
    if expected_total is not None and len(indices) != expected_total:
        return (
            f"  WARNING: {docname} has {len(indices)} chunks in Qdrant "
            f"but total_chunks={expected_total}"
        )
    return None


async def main() -> None:
    load_dotenv()

    erpnext_names = await fetch_erpnext_docnames()

    vs = VectorStore()
    payloads = vs.get_all_texts()
    qdrant_by_doctype = group_qdrant_payloads(payloads)

    print(f"Total Qdrant points: {len(payloads)}")
    print()

    in_sync = True
    for dt in DOCTYPES:
        erp_set = erpnext_names[dt]
        qdrant_docs = qdrant_by_doctype.get(dt, {})
        qdrant_set = set(qdrant_docs.keys())

        missing_from_qdrant = erp_set - qdrant_set
        extra_in_qdrant = qdrant_set - erp_set

        print(f"=== {dt} ===")
        print(f"  ERPNext docs: {len(erp_set)}")
        print(f"  Qdrant docs:  {len(qdrant_set)}")
        if missing_from_qdrant:
            in_sync = False
            print(
                f"  MISSING from Qdrant ({len(missing_from_qdrant)}): "
                f"{sorted(missing_from_qdrant)}"
            )
        if extra_in_qdrant:
            in_sync = False
            print(
                f"  STALE in Qdrant, not in ERPNext ({len(extra_in_qdrant)}): "
                f"{sorted(extra_in_qdrant)}"
            )

        for docname, indices in qdrant_docs.items():
            warning = check_chunk_counts(docname, indices, payloads)
            if warning:
                in_sync = False
                print(warning)

        if not missing_from_qdrant and not extra_in_qdrant:
            print("  IN SYNC")
        print()

    sys.exit(0 if in_sync else 1)


if __name__ == "__main__":
    asyncio.run(main())
