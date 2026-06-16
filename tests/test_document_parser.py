"""Tests for ingestion.document_parser. No network calls."""

from unittest.mock import MagicMock

import ingestion.document_parser as document_parser
from ingestion.document_parser import (
    extract_text_from_html,
    extract_text_from_pdf,
    invoice_to_text,
    po_to_text,
    supplier_scorecard_to_text,
)

# --- extract_text_from_html -------------------------------------------------


def test_extract_text_from_html_strips_tags() -> None:
    html = "<p>Payment due in <b>30</b> days.</p><p>Goods inspected on receipt.</p>"
    assert extract_text_from_html(html) == "Payment due in 30 days. Goods inspected on receipt."


def test_extract_text_from_html_collapses_whitespace() -> None:
    html = "<div>\n   Line one   \n\n   Line two   </div>"
    assert extract_text_from_html(html) == "Line one Line two"


def test_extract_text_from_html_handles_none_and_empty() -> None:
    assert extract_text_from_html(None) == ""
    assert extract_text_from_html("") == ""


# --- extract_text_from_pdf --------------------------------------------------


def test_extract_text_from_pdf_joins_pages(monkeypatch) -> None:
    page1 = MagicMock()
    page1.extract_text.return_value = "Page one content"
    page2 = MagicMock()
    page2.extract_text.return_value = "Page two content"
    fake_reader = MagicMock()
    fake_reader.pages = [page1, page2]
    monkeypatch.setattr(document_parser, "PdfReader", lambda _: fake_reader)

    result = extract_text_from_pdf(b"%PDF-fake-bytes")

    assert result == "Page one content\nPage two content"


def test_extract_text_from_pdf_skips_blank_pages(monkeypatch) -> None:
    page1 = MagicMock()
    page1.extract_text.return_value = "Real content"
    page2 = MagicMock()
    page2.extract_text.return_value = None  # pypdf returns None for unreadable pages
    fake_reader = MagicMock()
    fake_reader.pages = [page1, page2]
    monkeypatch.setattr(document_parser, "PdfReader", lambda _: fake_reader)

    result = extract_text_from_pdf(b"%PDF-fake-bytes")

    assert result == "Real content"


# --- po_to_text ---------------------------------------------------------


def test_po_to_text_with_full_data() -> None:
    po = {
        "name": "PUR-ORD-2026-00001",
        "supplier": "Zuckerman Security Ltd.",
        "transaction_date": "2026-01-13",
        "grand_total": 40000.0,
        "currency": "EUR",
        "schedule_date": "2026-01-13",
        "items": [{"item_code": "SKU001", "item_name": "T-shirt"}],
        "payment_terms_template": None,
        "status": "To Receive and Bill",
    }

    text = po_to_text(po)

    assert (
        "Purchase Order PUR-ORD-2026-00001 issued to Zuckerman Security Ltd. on 2026-01-13."
        in text
    )
    assert "Total value: 40000.0 EUR." in text
    assert "Items: T-shirt." in text
    assert "Payment terms: not specified." in text
    assert "Status: To Receive and Bill." in text


def test_po_to_text_falls_back_to_item_code_when_no_item_name() -> None:
    po = {"name": "PO-1", "items": [{"item_code": "SKU999"}]}
    assert "Items: SKU999." in po_to_text(po)


def test_po_to_text_handles_no_items() -> None:
    po = {"name": "PO-1", "items": []}
    assert "Items: none listed." in po_to_text(po)


# --- invoice_to_text ------------------------------------------------------


def test_invoice_to_text_with_full_data() -> None:
    inv = {
        "name": "ACC-PINV-2026-00001",
        "supplier": "Zuckerman Security Ltd.",
        "posting_date": "2026-01-21",
        "grand_total": 39000.0,
        "currency": "EUR",
        "outstanding_amount": 39000.0,
        "due_date": "2026-02-20",
        "items": [{"item_name": "T-shirt"}],
        "payment_terms_template": "Net 30",
        "status": "Overdue",
    }

    text = invoice_to_text(inv)

    assert (
        "Purchase Invoice ACC-PINV-2026-00001 from Zuckerman Security Ltd. dated 2026-01-21."
        in text
    )
    assert "Total value: 39000.0 EUR." in text
    assert "Outstanding amount: 39000.0 EUR." in text
    assert "Due date: 2026-02-20." in text
    assert "Payment terms: Net 30." in text
    assert "Status: Overdue." in text


# --- supplier_scorecard_to_text --------------------------------------------


def test_supplier_scorecard_to_text_with_criteria() -> None:
    sc = {
        "name": "SSC-001",
        "supplier": "Summit Traders Ltd.",
        "period": "Per Month",
        "supplier_score": 87.5,
        "indicator_color": "Green",
        "status": "Active",
        "criteria": [
            {"criteria_name": "Delivery", "score": 18, "max_score": 20},
            {"criteria_name": "Quality", "score": 9, "max_score": 10},
        ],
    }

    text = supplier_scorecard_to_text(sc)

    assert "Supplier Scorecard SSC-001 for Summit Traders Ltd., evaluated Per Month." in text
    assert "Overall score: 87.5 (Green standing)." in text
    assert "Criteria: Delivery 18/20, Quality 9/10." in text
    assert "Status: Active." in text


def test_supplier_scorecard_to_text_with_no_criteria() -> None:
    sc = {"name": "SSC-002", "supplier": "MA Inc.", "criteria": []}
    text = supplier_scorecard_to_text(sc)
    assert "Criteria: not specified." in text
