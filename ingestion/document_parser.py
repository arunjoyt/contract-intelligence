"""Document parsing for the ingestion pipeline.

Two jobs live here:
1. Raw text extraction (`extract_text_from_html`, `extract_text_from_pdf`) for the
   *unstructured* doc path (Contract, Terms and Conditions, attached PDFs).
2. Structured → natural-language serialization (`po_to_text`, `invoice_to_text`,
   `supplier_scorecard_to_text`) for the *structured* doc path (Purchase Order,
   Purchase Invoice, Supplier Scorecard) — each becomes a single string, embedded as
   one vector with no chunking.

These functions only serialize already-fetched ERPNext dicts; they don't call
`ERPNextClient` themselves and don't know about `supplier_group` enrichment (see
docs/ARCHITECTURE.md "Supplier Metadata Enrichment") — that's the caller's job.
"""

from __future__ import annotations

import io
from typing import Any

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


def _fmt(value: Any, default: str = "not specified") -> Any:
    """Render a possibly-missing field for embedding text. Real ERPNext records
    frequently leave optional fields (e.g. payment_terms_template) null — a bare
    "None" in indexed text reads as a hallucination-prone artifact, not data."""
    if value is None or value == "":
        return default
    return value


def _item_names(items: list[dict[str, Any]]) -> str:
    names = [item.get("item_name") or item.get("item_code") or "" for item in items]
    joined = ", ".join(name for name in names if name)
    return joined or "none listed"


def po_to_text(po: dict[str, Any]) -> str:
    """Serialize a Purchase Order dict into a single natural-language string."""
    return (
        f"Purchase Order {po.get('name')} issued to {po.get('supplier')} "
        f"on {po.get('transaction_date')}.\n"
        f"Total value: {po.get('grand_total')} {_fmt(po.get('currency'))}.\n"
        f"Delivery expected by {_fmt(po.get('schedule_date'))}.\n"
        f"Items: {_item_names(po.get('items', []))}.\n"
        f"Payment terms: {_fmt(po.get('payment_terms_template'))}.\n"
        f"Status: {po.get('status')}."
    )


def invoice_to_text(inv: dict[str, Any]) -> str:
    """Serialize a Purchase Invoice dict into a single natural-language string."""
    return (
        f"Purchase Invoice {inv.get('name')} from {inv.get('supplier')} "
        f"dated {inv.get('posting_date')}.\n"
        f"Total value: {inv.get('grand_total')} {_fmt(inv.get('currency'))}. "
        f"Outstanding amount: {inv.get('outstanding_amount')} {_fmt(inv.get('currency'))}.\n"
        f"Due date: {_fmt(inv.get('due_date'))}.\n"
        f"Items: {_item_names(inv.get('items', []))}.\n"
        f"Payment terms: {_fmt(inv.get('payment_terms_template'))}.\n"
        f"Status: {inv.get('status')}."
    )


def _criteria_summary(criteria: list[dict[str, Any]]) -> str:
    parts = []
    for c in criteria:
        name = c.get("criteria_name")
        if not name:
            continue
        score, max_score = c.get("score"), c.get("max_score")
        if score is not None and max_score is not None:
            parts.append(f"{name} {score}/{max_score}")
        else:
            parts.append(name)
    return ", ".join(parts)


def supplier_scorecard_to_text(sc: dict[str, Any]) -> str:
    """Serialize a Supplier Scorecard dict into a single natural-language string."""
    criteria_summary = _criteria_summary(sc.get("criteria", [])) or "not specified"
    return (
        f"Supplier Scorecard {sc.get('name')} for {sc.get('supplier')}, "
        f"evaluated {_fmt(sc.get('period'))}.\n"
        f"Overall score: {_fmt(sc.get('supplier_score'))} "
        f"({_fmt(sc.get('indicator_color'))} standing).\n"
        f"Criteria: {criteria_summary}.\n"
        f"Status: {sc.get('status')}."
    )
