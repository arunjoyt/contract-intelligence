"""Tests for scripts/sample_ingest.py pure helpers (#141).

Only argument parsing and the metadata sanity check are covered — the ingest run
hits a live ERPNext + Qdrant + OpenAI and is a manual pre-flight step.
"""

from __future__ import annotations

import sys
from argparse import ArgumentTypeError
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from sample_ingest import metadata_warnings, parse_doc_arg  # noqa: E402


def test_parse_doc_arg_splits_on_first_colon() -> None:
    assert parse_doc_arg("Contract:CON-2026-00012") == ("Contract", "CON-2026-00012")


def test_parse_doc_arg_rejects_missing_name() -> None:
    with pytest.raises(ArgumentTypeError):
        parse_doc_arg("Contract")


def test_parse_doc_arg_rejects_unknown_doctype() -> None:
    with pytest.raises(ArgumentTypeError):
        parse_doc_arg("Purchase Order:PO-0001")


def test_metadata_warnings_flags_all_none_contract() -> None:
    md = dict.fromkeys(("supplier", "start_date", "end_date", "status", "company"))
    warnings = metadata_warnings("Contract", "CON-1", md)
    assert len(warnings) == 1
    assert "customized Contract doctype" in warnings[0]


def test_metadata_warnings_quiet_when_any_field_present() -> None:
    assert metadata_warnings("Contract", "CON-1", {"supplier": "Acme Ltd"}) == []


def test_metadata_warnings_quiet_for_terms() -> None:
    # Terms and Conditions has no supplier/date/company fields by design.
    assert metadata_warnings("Terms and Conditions", "Std-Terms", {"status": None}) == []
