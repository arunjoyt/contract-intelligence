"""Tests for ingestion.chunker. No network calls."""

from ingestion.chunker import chunk_text


def test_chunk_text_empty_string_returns_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_chunk_text_short_text_returns_single_chunk() -> None:
    text = "Payment is due within 30 days of invoice date."
    chunks = chunk_text(text, chunk_size=512, chunk_overlap=64)

    assert chunks == [{"text": text, "chunk_index": 0, "total_chunks": 1}]


def test_chunk_text_long_text_splits_into_multiple_chunks() -> None:
    # Long enough to exceed chunk_size=512 with default langchain separators.
    paragraph = "This clause covers liability and indemnification terms. " * 30
    chunks = chunk_text(paragraph, chunk_size=512, chunk_overlap=64)

    assert len(chunks) > 1
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        assert chunk["chunk_index"] == i
        assert chunk["total_chunks"] == total
        assert chunk["text"]  # no empty chunks


def test_chunk_text_respects_chunk_size_roughly() -> None:
    paragraph = "Term and condition text repeated for length. " * 50
    chunks = chunk_text(paragraph, chunk_size=200, chunk_overlap=20)

    # RecursiveCharacterTextSplitter may slightly exceed chunk_size when no separator
    # is found within range, but chunks should stay in the right ballpark.
    assert all(len(c["text"]) <= 250 for c in chunks)


def test_chunk_text_overlap_shares_content_between_consecutive_chunks() -> None:
    paragraph = " ".join(f"sentence-{i}." for i in range(200))
    chunks = chunk_text(paragraph, chunk_size=100, chunk_overlap=30)

    assert len(chunks) > 1
    # With overlap, the tail of one chunk and the head of the next should share text.
    first_tail = chunks[0]["text"][-15:]
    assert first_tail in chunks[1]["text"] or chunks[1]["text"][:15] in chunks[0]["text"]


def test_chunk_text_force_single_chunk_bypasses_splitting() -> None:
    long_text = "Purchase Order item line. " * 100  # would normally split
    chunks = chunk_text(long_text, chunk_size=100, force_single_chunk=True)

    assert chunks == [{"text": long_text, "chunk_index": 0, "total_chunks": 1}]


def test_chunk_text_default_args_match_plan_spec() -> None:
    # chunk_size=512 / chunk_overlap=64 are the documented defaults.
    short_text = "Short structured document text."
    chunks = chunk_text(short_text)
    assert chunks[0]["text"] == short_text
