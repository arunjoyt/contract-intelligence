"""Desk-driven save on an already-submitted Contract -> webhook -> Qdrant update.

**FIXED (2026-08-28):** the gap recorded here since 2026-07-23 -- Frappe not firing a
webhook for a Desk save on an already-submitted Contract -- had two separate causes,
both now resolved on this dev bench:

1. Root cause (see #96): a Desk save on a submitted doc where only `allow_on_submit`
   fields changed calls Frappe's `on_update_after_submit` hook, not `on_update` --
   this project's only registered webhook for Contract post-submit changes was
   `contract-on-update` (`on_update`), which was never going to fire for this case.
   Frappe's webhook dispatcher also hardcoded an event allow-list that omitted
   `on_update_after_submit` even though it's a valid, selectable event -- fixed
   upstream (frappe/frappe#39164, in `develop` since 2026-05-08) and present in this
   bench's installed frappe (16.30.0). A `contract-on-update-after-submit` webhook
   (`webhook_docevent = on_update_after_submit`) is now registered per
   `docs/DEPLOYMENT.md`.
2. Unrelated: this dev site was migrated from bench `b2` to `b3` on 2026-08-21 (see
   `CLAUDE.local.md`), and every `Webhook.webhook_secret` value carried over from
   `b2` was encrypted under `b2`'s site `encryption_key` -- undecryptable under
   `b3`'s different key (`cryptography.fernet.InvalidToken`), so *no* Contract/Terms
   webhook fired at all until each was re-saved (re-encrypting under the current
   key). Any future bench/site migration needs the same re-save step for every
   `Password`-type field carried over in a restore.

Contract's `is_signed` field is `allow_on_submit`, so a Desk edit that only checks
"Signed" is a legal, saveable edit on a submitted doc -- that's what these tests
attempt via the Desk UI, to test with the most realistic legal post-submit edit
available, not a synthetic one.

Note: Contract's `status` field is also `allow_on_submit` but is `hidden` in the
Desk UI -- it's computed automatically by ERPNext's own Contract controller from
`is_signed` + the date range (confirmed live: setting is_signed=1 flips status
from "Unsigned" to "Active" -- verified via REST immediately after the Desk
save). So these tests edit `is_signed` via the Desk UI and assert on the
resulting `status` change in Qdrant's payload -- that's the observable signal
that the webhook fired and re-indexing happened.
"""

from __future__ import annotations

from tests.e2e.conftest import (
    ERPNEXT_URL,
    create_draft_contract,
    live,
    poll_qdrant,
    set_is_signed_and_update,
    submit_contract_via_rest,
)
from tests.e2e.conftest import cleanup_contract as _cleanup


@live
def test_contract_desk_is_signed_change_after_submit_triggers_reindex(desk_page) -> None:
    docname, _ = create_draft_contract()
    try:
        submit_contract_via_rest(docname)
        points = poll_qdrant(docname, predicate=lambda pts: len(pts) > 0)  # baseline: indexed
        assert points, (
            f"{docname} was never indexed after submit -- contract-on-submit webhook may not "
            "be firing (check Frappe's Webhook Request Log before assuming this is the "
            "on_update_after_submit gap)"
        )

        desk_page.goto(f"{ERPNEXT_URL}/app/contract/{docname}")
        set_is_signed_and_update(desk_page, checked=True)

        points = poll_qdrant(
            docname,
            predicate=lambda pts: any(p["payload"].get("status") == "Active" for p in pts),
        )
        assert any(p["payload"].get("status") == "Active" for p in points), (
            f"Qdrant payload for {docname} was not updated to status=Active after Desk save "
            "-- on_update_after_submit may not be firing for post-submit Desk saves on Contract"
        )
    finally:
        _cleanup(docname)


@live
def test_contract_desk_repeated_saves_do_not_duplicate_index_points(desk_page) -> None:
    """Repeated Desk saves on a submitted Contract each fire on_update_after_submit
    and re-index (delete_by_docname + re-upsert) -- this checks that idempotent
    upsert (deterministic uuid5 point ids, see docs/ARCHITECTURE.md) holds across
    several such cycles, rather than accumulating duplicate points per save.
    """
    docname, _ = create_draft_contract()
    try:
        submit_contract_via_rest(docname)
        points_before = poll_qdrant(docname, predicate=lambda pts: len(pts) > 0)
        assert points_before, f"{docname} was never indexed after submit"

        desk_page.goto(f"{ERPNEXT_URL}/app/contract/{docname}")
        set_is_signed_and_update(desk_page, checked=True)
        set_is_signed_and_update(desk_page, checked=False)
        set_is_signed_and_update(desk_page, checked=True)

        # Each save re-indexes asynchronously; wait for the final save's effect
        # (status back to Active) before comparing point counts.
        points_after = poll_qdrant(
            docname,
            predicate=lambda pts: any(p["payload"].get("status") == "Active" for p in pts),
        )

        assert len(points_after) == len(points_before), (
            f"Point count for {docname} changed across repeated Desk saves "
            f"({len(points_before)} -> {len(points_after)}) -- idempotent upsert may not "
            "be holding across repeated on_update_after_submit re-indexing"
        )
    finally:
        _cleanup(docname)
