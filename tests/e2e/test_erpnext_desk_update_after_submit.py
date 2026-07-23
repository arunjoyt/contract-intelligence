"""Desk-driven save on an already-submitted Contract -> webhook -> Qdrant update.

**CONFIRMED GAP (2026-07-23, via a live RUN_E2E=1 run of this exact file):**
Frappe's own `Webhook Request Log` doctype shows **zero delivery attempts** --
not a failed delivery, a delivery that was never even tried -- for the window
between an ERPNext Desk "Update" action on a submitted Contract and the next
webhook call. `on_update` does not fire for a Desk save on an already-submitted
Contract, full stop. This was previously only verified against Purchase Order
(docs/DEPLOYMENT.md's Future Enhancements section); this run confirms the same
gap exists for Contract's plain `on_update` webhook too, not just PO's
`on_update_after_submit`. Both tests below are marked `xfail(strict=True)`
documenting this as a known platform limitation -- if a future Frappe version
(or a workaround) starts firing the webhook, `strict=True` turns that into a
loud XPASS instead of a silent pass, so it gets noticed and the xfail removed.

Contract's `is_signed` field is `allow_on_submit`, so a Desk edit that only
checks "Signed" is a legal, saveable edit on a submitted doc -- that's what
these tests attempt via the Desk UI, to test with the most realistic legal
post-submit edit available, not a synthetic one.

Note: Contract's `status` field is also `allow_on_submit` but is `hidden` in the
Desk UI -- it's computed automatically by ERPNext's own Contract controller from
`is_signed` + the date range (confirmed live: setting is_signed=1 flips status
from "Unsigned" to "Active" -- verified via REST immediately after the Desk
save). So these tests edit `is_signed` via the Desk UI and assert on the
resulting `status` change in Qdrant's payload -- that would be the observable
signal, if the webhook fired.
"""

from __future__ import annotations

import pytest

from tests.e2e.conftest import (
    ERPNEXT_URL,
    create_draft_contract,
    live,
    poll_qdrant,
    set_is_signed_and_update,
    submit_contract_via_rest,
)
from tests.e2e.conftest import cancel_contract_via_rest as _cancel

_XFAIL_REASON = (
    "Confirmed via Frappe's Webhook Request Log (2026-07-23): on_update does not "
    "fire for a Desk 'Update' action on an already-submitted Contract -- zero "
    "delivery attempts recorded, not a failed one. Known platform limitation, "
    "see module docstring."
)


@live
@pytest.mark.xfail(reason=_XFAIL_REASON, strict=True)
def test_contract_desk_is_signed_change_after_submit_triggers_reindex(desk_page) -> None:
    docname, _ = create_draft_contract()
    try:
        submit_contract_via_rest(docname)
        poll_qdrant(docname, predicate=lambda pts: len(pts) > 0)  # baseline: indexed

        desk_page.goto(f"{ERPNEXT_URL}/app/contract/{docname}")
        set_is_signed_and_update(desk_page, checked=True)

        # Short timeout: this is expected to never succeed (see xfail reason) --
        # no need to burn the default 60s poll window on every run.
        points = poll_qdrant(
            docname,
            predicate=lambda pts: any(p["payload"].get("status") == "Active" for p in pts),
            timeout=15,
        )
        assert any(p["payload"].get("status") == "Active" for p in points), (
            f"Qdrant payload for {docname} was not updated to status=Active after Desk save "
            "-- on_update may not be firing for post-submit Desk saves on Contract"
        )
    finally:
        _cancel(docname)


@live
def test_contract_desk_repeated_saves_do_not_silently_corrupt_index(desk_page) -> None:
    """Companion to the xfail'd test above: given on_update doesn't fire for
    post-submit Desk saves (confirmed gap, see module docstring), repeated Desk
    saves should at least be inert -- not silently duplicate points or corrupt
    the existing index -- rather than untested territory once a webhook fix
    lands and this starts actually re-indexing on every save.
    """
    docname, _ = create_draft_contract()
    try:
        submit_contract_via_rest(docname)
        points_before = poll_qdrant(docname, predicate=lambda pts: len(pts) > 0)

        desk_page.goto(f"{ERPNEXT_URL}/app/contract/{docname}")
        set_is_signed_and_update(desk_page, checked=True)
        set_is_signed_and_update(desk_page, checked=False)
        set_is_signed_and_update(desk_page, checked=True)

        # No predicate to wait on (nothing is expected to change) -- just give
        # Frappe's request/response cycle a moment, then compare directly.
        desk_page.wait_for_timeout(3000)
        points_after = poll_qdrant(docname, predicate=lambda pts: len(pts) > 0, timeout=5)

        assert len(points_after) == len(points_before), (
            f"Point count for {docname} changed across repeated Desk saves "
            f"({len(points_before)} -> {len(points_after)}) with no webhook expected to "
            "have fired -- investigate before assuming this is still just the known gap"
        )
    finally:
        _cancel(docname)
