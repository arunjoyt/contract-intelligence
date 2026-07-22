"""Cleanup phase for `--reset`: wipes every Contract and Terms and Conditions doc, and
resets the Qdrant collection. Generic and allowlist-free — safe to run against an empty
site (no-op) or a dirty one alike. Suppliers are never touched: they may be referenced by
unrelated ERPNext data outside this system's ingestion scope.
"""

from __future__ import annotations

from retrieval.vector_store import VectorStore

from .erp_admin_client import ERPAdminClient, ERPAdminError


def cleanup_contracts(client: ERPAdminClient, *, dry_run: bool = False) -> int:
    """Delete every Contract doc, cancelling submitted ones first. Returns count deleted.

    A doc that can't be deleted because something outside this system's scope links to it
    (e.g. an old Purchase Order) is left alone with a warning, rather than aborting the run
    — the same reasoning that keeps Suppliers untouched.
    """
    contracts = client.get_list("Contract", fields=["name", "docstatus"], limit=0)
    deleted = 0
    for c in contracts:
        name, docstatus = c["name"], c["docstatus"]
        if dry_run:
            action = "cancel + delete" if docstatus == 1 else "delete"
            print(f"[dry-run] would {action} Contract {name} (docstatus={docstatus})")
            continue
        try:
            if docstatus == 1:
                client.cancel_doc("Contract", name)
            client.delete_doc("Contract", name)
            print(f"deleted Contract {name} (was docstatus={docstatus})")
            deleted += 1
        except ERPAdminError as e:
            print(f"WARNING: could not delete Contract {name}, leaving it in place: {e}")
    return deleted


def cleanup_terms_and_conditions(client: ERPAdminClient, *, dry_run: bool = False) -> int:
    """Delete every Terms and Conditions doc (not submittable, always a direct delete).

    See `cleanup_contracts` for why a link-exists failure is a warning, not an abort.
    """
    docs = client.get_list("Terms and Conditions", fields=["name"], limit=0)
    deleted = 0
    for d in docs:
        name = d["name"]
        if dry_run:
            print(f"[dry-run] would delete Terms and Conditions {name}")
            continue
        try:
            client.delete_doc("Terms and Conditions", name)
            print(f"deleted Terms and Conditions {name}")
            deleted += 1
        except ERPAdminError as e:
            print(
                f"WARNING: could not delete Terms and Conditions {name}, leaving it in place: {e}"
            )
    return deleted


def reset_qdrant(*, dry_run: bool = False) -> None:
    if dry_run:
        print("[dry-run] would reset Qdrant collection")
        return
    VectorStore().reset_collection()
    print("Qdrant collection reset")


def run_cleanup(*, dry_run: bool = False) -> None:
    with ERPAdminClient() as client:
        n_contracts = cleanup_contracts(client, dry_run=dry_run)
        n_tandc = cleanup_terms_and_conditions(client, dry_run=dry_run)
    reset_qdrant(dry_run=dry_run)
    print(f"cleanup done: {n_contracts} Contract(s), {n_tandc} Terms and Conditions doc(s)")
