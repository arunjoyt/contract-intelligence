"""Tests for scripts/setup_erpnext_webhooks.py pure helpers (#141).

Only the record shape and the idempotency decision (`plan_action`) are covered —
the ERPNext REST calls are a live ops step, not unit-tested (same convention as
the other scripts/, see tests/test_benchmark_script.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from setup_erpnext_webhooks import (  # noqa: E402
    WEBHOOKS,
    desired_record,
    plan_action,
)

_URL = "https://api-x.example.com/webhook/erpnext"


def _desired() -> dict:
    return desired_record("contract-on-update", "Contract", "on_update", _URL, "sek")


def test_webhooks_cover_the_post_submit_event() -> None:
    events = {(dt, ev) for _, dt, ev in WEBHOOKS}
    assert ("Contract", "on_update_after_submit") in events  # issue #96
    assert ("Terms and Conditions", "on_submit") not in events  # not submittable
    assert len(WEBHOOKS) == 5


def test_desired_record_has_security_on() -> None:
    d = _desired()
    assert d["enable_security"] == 1
    assert d["webhook_secret"] == "sek"
    assert d["request_structure"] == "JSON"
    assert d["webhook_json"] == '{"doctype": "{{ doc.doctype }}", "docname": "{{ doc.name }}"}'


def test_plan_action_create_when_absent() -> None:
    assert plan_action(None, _desired()) == "create"


def test_plan_action_ok_when_all_managed_fields_match() -> None:
    d = _desired()
    remote = {k: str(v) for k, v in d.items()}
    assert plan_action(remote, d) == "ok"


def test_plan_action_ignores_the_masked_secret() -> None:
    d = _desired()
    remote = {k: str(v) for k, v in d.items()} | {"webhook_secret": "*" * 9}
    assert plan_action(remote, d) == "ok"


def test_plan_action_update_on_url_drift() -> None:
    d = _desired()
    remote = {k: str(v) for k, v in d.items()} | {"request_url": "https://old/webhook/erpnext"}
    assert plan_action(remote, d) == "update"


def test_plan_action_update_when_security_disabled_remotely() -> None:
    d = _desired()
    remote = {k: str(v) for k, v in d.items()} | {"enable_security": "0"}
    assert plan_action(remote, d) == "update"
