"""Seed phase for `--seed`: create-if-missing Suppliers, Terms and Conditions, Contracts,
and PDF attachments from a loaded `fixtures.DemoData`. Every step is idempotent so re-running
against a partially-seeded site never duplicates records.
"""

from __future__ import annotations

from . import pdf_gen
from .erp_admin_client import ERPAdminClient
from .fixtures import Contract, DemoData, Supplier, TermsAndConditions


def seed_supplier_groups(
    client: ERPAdminClient, suppliers: list[Supplier], *, dry_run: bool = False
) -> None:
    """Create-if-missing any `Supplier Group` master record the fixture references.

    `Supplier`.`supplier_group` is a Link field — ERPNext rejects a Supplier create/update
    if the referenced group doesn't already exist as its own doctype record.
    """
    needed = {s.supplier_group for s in suppliers}
    existing = {g["name"] for g in client.get_list("Supplier Group", fields=["name"], limit=0)}
    for group in sorted(needed - existing):
        if dry_run:
            print(f"[dry-run] would create Supplier Group {group!r}")
            continue
        client.create_doc(
            "Supplier Group",
            {
                "supplier_group_name": group,
                "parent_supplier_group": "All Supplier Groups",
                "is_group": 0,
            },
        )
        print(f"created Supplier Group {group!r}")


def seed_suppliers(
    client: ERPAdminClient, suppliers: list[Supplier], *, dry_run: bool = False
) -> dict[str, str]:
    """Create-if-missing. Returns a map of fixture supplier name -> actual ERPNext docname."""
    name_map: dict[str, str] = {}
    for s in suppliers:
        existing = client.get_list(
            "Supplier",
            filters=[["supplier_name", "=", s.name]],
            fields=["name", "supplier_group"],
            limit=1,
        )
        if existing:
            name_map[s.name] = existing[0]["name"]
            if existing[0]["supplier_group"] != s.supplier_group and not dry_run:
                client.update_doc(
                    "Supplier", existing[0]["name"], {"supplier_group": s.supplier_group}
                )
                print(f"updated Supplier {existing[0]['name']} supplier_group={s.supplier_group!r}")
            continue
        if dry_run:
            print(f"[dry-run] would create Supplier {s.name!r} (group={s.supplier_group!r})")
            name_map[s.name] = s.name
            continue
        created = client.create_doc(
            "Supplier", {"supplier_name": s.name, "supplier_group": s.supplier_group}
        )
        name_map[s.name] = created["name"]
        print(f"created Supplier {created['name']}")
    return name_map


def seed_terms_and_conditions(
    client: ERPAdminClient, docs: list[TermsAndConditions], *, dry_run: bool = False
) -> None:
    """Upsert (not just create-if-missing): a stale doc that `--reset` couldn't delete
    (e.g. still linked from an old Purchase Order) must still converge to the fixture's
    authored content on re-seed, rather than silently keeping outdated text.
    """
    for t in docs:
        existing = client.get_list(
            "Terms and Conditions", filters=[["title", "=", t.title]], fields=["name"], limit=1
        )
        payload = {"terms": t.terms_html, "disabled": 1 if t.disabled else 0}
        if existing:
            if dry_run:
                print(f"[dry-run] would update Terms and Conditions {t.title!r}")
                continue
            client.update_doc("Terms and Conditions", existing[0]["name"], payload)
            print(f"updated Terms and Conditions {t.title!r}")
            continue
        if dry_run:
            print(f"[dry-run] would create Terms and Conditions {t.title!r}")
            continue
        client.create_doc("Terms and Conditions", {"title": t.title, **payload})
        print(f"created Terms and Conditions {t.title!r}")


def _get_or_create_contract(
    client: ERPAdminClient,
    contract: Contract,
    supplier_name_map: dict[str, str],
    *,
    dry_run: bool = False,
) -> str | None:
    """Returns the Contract's docname, or None in dry-run mode (nothing was created)."""
    real_party_name = (
        supplier_name_map.get(contract.party_name, contract.party_name)
        if contract.party_type == "Supplier"
        else contract.party_name
    )
    existing = client.get_list(
        "Contract",
        filters=[
            ["party_name", "=", real_party_name],
            ["start_date", "=", contract.start_date],
            ["end_date", "=", contract.end_date],
        ],
        fields=["name", "docstatus"],
        limit=1,
    )
    if existing:
        return existing[0]["name"]

    if dry_run:
        print(
            f"[dry-run] would create Contract {contract.key!r} "
            f"(party={real_party_name!r}, docstatus={contract.docstatus})"
        )
        return None

    created = client.create_doc(
        "Contract",
        {
            "party_type": contract.party_type,
            "party_name": real_party_name,
            "status": contract.status,
            "start_date": contract.start_date,
            "end_date": contract.end_date,
            "contract_terms": contract.contract_terms_html,
            "is_signed": 1 if contract.is_signed else 0,
        },
    )
    docname = created["name"]

    if contract.docstatus in (1, 2):
        client.submit_doc(created)
    if contract.docstatus == 2:
        client.cancel_doc("Contract", docname)

    print(f"created Contract {docname} ({contract.key}, docstatus={contract.docstatus})")
    return docname


def _attach_pdf_if_needed(
    client: ERPAdminClient, docname: str, contract: Contract, *, dry_run: bool = False
) -> None:
    if contract.pdf_attachment is None:
        return
    filename = f"{contract.key}.pdf"
    existing_files = client.get_attached_files("Contract", docname)
    if any(f["file_name"] == filename for f in existing_files):
        return

    if dry_run:
        print(f"[dry-run] would attach {filename} to Contract {docname}")
        return

    pdf_bytes = pdf_gen.render_pdf(
        title=contract.pdf_attachment.get("title", contract.key),
        party_name=contract.party_name,
        clauses=contract.pdf_attachment["clauses"],
    )
    client.upload_file("Contract", docname, filename, pdf_bytes, is_private=True)
    print(f"attached {filename} to Contract {docname}")


def seed_contracts(
    client: ERPAdminClient,
    contracts: list[Contract],
    supplier_name_map: dict[str, str],
    *,
    dry_run: bool = False,
) -> None:
    for contract in contracts:
        docname = _get_or_create_contract(client, contract, supplier_name_map, dry_run=dry_run)
        if docname is None:
            continue
        _attach_pdf_if_needed(client, docname, contract, dry_run=dry_run)


def run_seed(data: DemoData, *, dry_run: bool = False) -> None:
    with ERPAdminClient() as client:
        seed_supplier_groups(client, data.suppliers, dry_run=dry_run)
        supplier_name_map = seed_suppliers(client, data.suppliers, dry_run=dry_run)
        seed_terms_and_conditions(client, data.terms_and_conditions, dry_run=dry_run)
        seed_contracts(client, data.contracts, supplier_name_map, dry_run=dry_run)
    print(
        f"seed done: {len(data.suppliers)} supplier(s), "
        f"{len(data.terms_and_conditions)} T&C doc(s), {len(data.contracts)} contract(s) processed"
    )
