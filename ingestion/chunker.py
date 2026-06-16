"""Text chunking for the ingestion pipeline.

Unstructured documents (Contract, Terms and Conditions, attached PDFs) get split via
`RecursiveCharacterTextSplitter` so each piece fits the embedding model's effective
context. Structured documents (Purchase Order, Purchase Invoice, Supplier Scorecard)
are short, single-vector NL strings from `document_parser.py` — they bypass splitting
entirely via `force_single_chunk`, so a `PO` with an unusually long item list can never
accidentally get fragmented across vectors (see docs/ARCHITECTURE.md "Document
Indexing Strategy").
"""

from __future__ import annotations

from typing import TypedDict

from langchain.text_splitter import RecursiveCharacterTextSplitter


class Chunk(TypedDict):
    text: str
    chunk_index: int
    total_chunks: int


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    force_single_chunk: bool = False,
) -> list[Chunk]:
    """Split `text` into overlapping chunks for embedding.

    For short text (structured docs typically fit within `chunk_size` already), this
    naturally returns a single chunk without any special-casing. `force_single_chunk`
    guarantees that regardless of length — callers serializing structured docs should
    pass it so the "no chunking" rule holds even for unusually long records.
    """
    if not text or not text.strip():
        return []

    if force_single_chunk:
        return [{"text": text, "chunk_index": 0, "total_chunks": 1}]

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    pieces = splitter.split_text(text)
    total_chunks = len(pieces)
    return [
        {"text": piece, "chunk_index": i, "total_chunks": total_chunks}
        for i, piece in enumerate(pieces)
    ]
