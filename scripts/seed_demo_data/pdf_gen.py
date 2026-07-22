"""Renders a fixture entry's PDF-attachment content into a formatted PDF via reportlab."""

from __future__ import annotations

from io import BytesIO
from typing import Any

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def render_pdf(title: str, party_name: str, clauses: list[dict[str, Any]]) -> bytes:
    """Render a contract's clauses into a formatted PDF, returning raw bytes.

    `clauses` is a list of `{"heading": str, "body": str}` dicts rendered as numbered
    sections, followed by a signature block.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        topMargin=1 * inch,
        bottomMargin=1 * inch,
        leftMargin=1 * inch,
        rightMargin=1 * inch,
    )
    styles = getSampleStyleSheet()
    heading_style = ParagraphStyle(
        "ClauseHeading", parent=styles["Heading2"], spaceBefore=12, spaceAfter=4
    )
    body_style = ParagraphStyle("ClauseBody", parent=styles["Normal"], spaceAfter=8)

    story = [
        Paragraph(title, styles["Title"]),
        Paragraph(f"Party: {party_name}", styles["Normal"]),
        Spacer(1, 0.3 * inch),
    ]
    for i, clause in enumerate(clauses, start=1):
        story.append(Paragraph(f"{i}. {clause['heading']}", heading_style))
        story.append(Paragraph(clause["body"], body_style))

    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph("Signatures", heading_style))
    story.append(Paragraph("_______________________  Date: __________", body_style))
    story.append(Paragraph("_______________________  Date: __________", body_style))

    doc.build(story)
    return buffer.getvalue()
