"""Document parsing for the ingestion pipeline.

Raw text extraction (`extract_text_from_html`, `extract_text_from_pdf`) for the
unstructured doc path (Contract, Terms and Conditions, attached PDFs) — the only
doctype path this system indexes.
"""

from __future__ import annotations

import io

from bs4 import BeautifulSoup
from pypdf import PdfReader


def extract_text_from_html(html: str | None) -> str:
    """Strip HTML tags and return visible text with collapsed whitespace."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator=" ")
    return " ".join(text.split())


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract and join text from all pages of a PDF."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n".join(page for page in pages if page)
