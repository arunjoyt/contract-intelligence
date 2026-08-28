"""Text chunking for the ingestion pipeline.

Contract, Terms and Conditions, and their attached PDFs get split via
`RecursiveCharacterTextSplitter` so each piece fits the embedding model's effective
context (see docs/ARCHITECTURE.md "Document Indexing Strategy").
"""

from __future__ import annotations

from typing import TypedDict

from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE


class Chunk(TypedDict):
    text: str
    chunk_index: int
    total_chunks: int


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
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
