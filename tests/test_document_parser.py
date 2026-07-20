"""Tests for ingestion.document_parser. No network calls."""

from unittest.mock import MagicMock

import ingestion.document_parser as document_parser
from ingestion.document_parser import extract_text_from_html, extract_text_from_pdf

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
