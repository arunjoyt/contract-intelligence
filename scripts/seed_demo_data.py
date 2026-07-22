#!/usr/bin/env python3
"""Demo data seed tool for a fresh or dirty ERPNext site. See docs/DEMO_DATA_PLAN.md.

Usage:
    python scripts/seed_demo_data.py --reset          # delete all Contract/T&C docs + reset Qdrant
    python scripts/seed_demo_data.py --seed           # idempotent create-if-missing
    python scripts/seed_demo_data.py --reset --seed   # typical full rebuild
    python scripts/seed_demo_data.py --seed --dry-run # print planned actions, no writes
    python scripts/seed_demo_data.py --verify         # read-only post-checks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.seed_demo_data.cleanup import run_cleanup
from scripts.seed_demo_data.erp_admin_client import ERPAdminClient
from scripts.seed_demo_data.fixtures import load_fixture
from scripts.seed_demo_data.seed import run_seed

FIXTURE_PATH = Path(__file__).resolve().parent / "seed_data" / "demo_data.yaml"


def run_verify() -> None:
    with ERPAdminClient() as client:
        contracts = client.get_list("Contract", fields=["name", "status", "docstatus"], limit=0)
        suppliers = client.get_list(
            "Supplier", fields=["name", "supplier_group"], limit=0
        )
        tandc = client.get_list("Terms and Conditions", fields=["name"], limit=0)
        pdf_files = client.get_list(
            "File",
            filters=[["attached_to_doctype", "=", "Contract"], ["file_name", "like", "%.pdf"]],
            fields=["name", "attached_to_name"],
            limit=0,
        )

    status_counts: dict[str, int] = {}
    for c in contracts:
        status_counts[c["status"]] = status_counts.get(c["status"], 0) + 1
    docstatus_counts: dict[int, int] = {}
    for c in contracts:
        docstatus_counts[c["docstatus"]] = docstatus_counts.get(c["docstatus"], 0) + 1

    missing_group = [s["name"] for s in suppliers if not s.get("supplier_group")]
    globex = [c for c in contracts if "globex" in c["name"].lower()]

    print(f"Contracts: {len(contracts)} total")
    print(f"  by status: {status_counts}")
    print(f"  by docstatus: {docstatus_counts}")
    print(f"Suppliers: {len(suppliers)} total, {len(missing_group)} missing supplier_group")
    if missing_group:
        print(f"  MISSING GROUP: {missing_group}")
    print(f"Terms and Conditions: {len(tandc)} total")
    print(f"PDF attachments on Contracts: {len(pdf_files)}")
    if globex:
        print(f"  WARNING: found contract(s) referencing 'Globex' by docname: {globex}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--reset", action="store_true", help="Delete all Contract/T&C docs, reset Qdrant"
    )
    parser.add_argument(
        "--seed", action="store_true", help="Idempotently create docs from the fixture"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print planned actions without making any writes"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Read-only post-checks against the live site"
    )
    parser.add_argument(
        "--fixture", type=Path, default=FIXTURE_PATH, help="Path to the demo data YAML fixture"
    )
    args = parser.parse_args()

    if not any([args.reset, args.seed, args.verify]):
        parser.print_help()
        sys.exit(1)

    load_dotenv()

    if args.reset:
        print("=== reset ===")
        run_cleanup(dry_run=args.dry_run)

    if args.seed:
        print("=== seed ===")
        data = load_fixture(args.fixture)
        run_seed(data, dry_run=args.dry_run)

    if args.verify:
        print("=== verify ===")
        run_verify()


if __name__ == "__main__":
    main()
