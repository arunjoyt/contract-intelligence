#!/usr/bin/env python3
"""Ingest a hand-picked sample of ERPNext documents for the pre-flight smoke check.

Runbook Phase 4a: before the full ingest, index 15-30 representative documents into
a throwaway collection and inspect the points, so client-specific data-shape
problems (customized Contract doctype, scanned-image PDFs, mojibake) surface while
they are still cheap to fix.

This reuses the exact production helpers (`prepare_doc_for_indexing`,
`gather_chunks_for_doc`, `resolve_supplier_group`) so it exercises the real code
path — it is a trimmed `_do_full_ingest` from `api/main.py` with no BM25 rebuild
and an explicit document list instead of a full listing.

Usage
-----
    # deliberate pick (repeat --doc; cover largest contracts, PDFs, custom fields, T&C)
    QDRANT_COLLECTION=contract_smoke python scripts/sample_ingest.py \
        --doc "Contract:CON-2026-00012" --doc "Terms and Conditions:Standard-Terms"

    # or first N of each doctype
    python scripts/sample_ingest.py --collection contract_smoke --limit 15

    python scripts/sample_ingest.py --collection contract_smoke --limit 5 --dry-run

Then inspect with the Qdrant scroll snippets in `docs/DEPLOYMENT.md` §11 (swap in
your smoke collection), and drop the collection when done:
``DELETE /collections/<name>``.

Reads `ERPNEXT_URL` / `ERPNEXT_API_KEY` / `ERPNEXT_API_SECRET` / `QDRANT_URL` /
`OPENAI_API_KEY` from the environment (`.env` is loaded automatically).
`--collection` overrides `QDRANT_COLLECTION` for this run.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

from ingestion.embedder import Embedder  # noqa: E402
from ingestion.erpnext_client import ERPNextClient  # noqa: E402
from ingestion.webhook_handler import (  # noqa: E402
    gather_chunks_for_doc,
    prepare_doc_for_indexing,
    resolve_supplier_group,
)
from retrieval.vector_store import VectorStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DOCTYPES = ("Contract", "Terms and Conditions")
_METADATA_SIGNAL_FIELDS = ("supplier", "start_date", "end_date", "status", "company")


def parse_doc_arg(value: str) -> tuple[str, str]:
    """`"Contract:CON-2026-00012"` -> `("Contract", "CON-2026-00012")`."""
    doctype, _, name = value.partition(":")
    if not doctype or not name:
        raise argparse.ArgumentTypeError(f"expected 'Doctype:Name', got {value!r}")
    if doctype not in _DOCTYPES:
        raise argparse.ArgumentTypeError(f"doctype must be one of {_DOCTYPES}, got {doctype!r}")
    return doctype, name


def metadata_warnings(doctype: str, name: str, metadata: dict) -> list[str]:
    """Cheap sanity checks on one document's derived metadata. Pure — no I/O."""
    warnings: list[str] = []
    if doctype == "Contract" and all(metadata.get(f) is None for f in _METADATA_SIGNAL_FIELDS):
        warnings.append(
            f"{name}: every metadata field is None — customized Contract doctype? "
            "prepare_doc_for_indexing needs field-name updates"
        )
    return warnings


async def _ingest_one(
    doctype: str, name: str, client: ERPNextClient, embedder: Embedder, store: VectorStore,
    *, dry_run: bool,
) -> dict:
    doc = await client.get_doc(doctype, name)
    supplier_group = await resolve_supplier_group(doctype, doc, client)
    text, metadata, force_single = prepare_doc_for_indexing(doctype, doc, supplier_group)
    chunks = await gather_chunks_for_doc(doctype, doc, text, force_single, client)

    result = {"chunks": len(chunks), "warnings": metadata_warnings(doctype, name, metadata)}
    if not chunks:
        result["warnings"].insert(0, f"{name}: 0 chunks — empty body or unreadable PDF?")
        return result
    if not dry_run:
        vectors = embedder.embed_texts([c["text"] for c in chunks])
        store.upsert_chunks(
            [{**c, **metadata, "vector": v} for c, v in zip(chunks, vectors, strict=True)]
        )
    return result


async def _run(targets: list[tuple[str, str]], *, dry_run: bool) -> int:
    collection = os.environ["QDRANT_COLLECTION"]
    embedder = Embedder()
    store = VectorStore()
    logger.info("collection %r%s", collection, "  (dry run — no writes)" if dry_run else "")
    if not dry_run:
        store.ensure_collection()

    total_chunks = 0
    all_warnings: list[str] = []
    async with ERPNextClient() as client:
        for doctype, name in targets:
            try:
                r = await _ingest_one(doctype, name, client, embedder, store, dry_run=dry_run)
            except Exception as exc:  # noqa: BLE001 — smoke tool, report and continue
                logger.error("%s %s: failed — %s", doctype, name, exc)
                all_warnings.append(f"{name}: fetch/parse failed — {exc}")
                continue
            total_chunks += r["chunks"]
            level = logging.WARNING if r["warnings"] else logging.INFO
            logger.log(level, "%s %s: %d chunks", doctype, name, r["chunks"])
            all_warnings.extend(r["warnings"])

    logger.info("--- %d documents, %d chunks ---", len(targets), total_chunks)
    for w in all_warnings:
        logger.warning(w)
    logger.info(
        "inspect the points with the Qdrant scroll snippets in docs/DEPLOYMENT.md §11, "
        "then DELETE /collections/%s when done",
        collection,
    )
    return 1 if all_warnings else 0


async def _list_targets(limit: int) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    async with ERPNextClient() as client:
        for doctype in _DOCTYPES:
            rows = await client.get_list(doctype, fields=["name"], limit=limit)
            targets.extend((doctype, r["name"]) for r in rows)
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--doc",
        type=parse_doc_arg,
        action="append",
        metavar="Doctype:Name",
        help="a specific document to ingest; repeat for more (preferred — pick deliberately)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="fallback when no --doc given: first N of each of Contract / Terms and Conditions",
    )
    parser.add_argument(
        "--collection", help="Qdrant collection for this run (overrides QDRANT_COLLECTION)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="fetch/parse/chunk only, no embed or upsert"
    )
    args = parser.parse_args()

    if not args.doc and not args.limit:
        parser.error("pass --doc (repeatable) or --limit N")
    if args.collection:
        os.environ["QDRANT_COLLECTION"] = args.collection

    targets = args.doc or asyncio.run(_list_targets(args.limit))
    if not targets:
        logger.error("no documents to ingest")
        return 1
    return asyncio.run(_run(targets, dry_run=args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
