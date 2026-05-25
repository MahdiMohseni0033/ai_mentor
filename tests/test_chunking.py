"""Chunking respects word bounds, avoids empties, and is stable."""

from CONFIG import CONFIG
from src.ingestion.chunk_text import chunk_text
from src.ingestion.load_transcripts import Document
from src.ingestion.pipeline import build_chunk_records, content_hash
from src.utils.text_utils import count_words

# ~900 words of distinct sentences.
LONG_TEXT = " ".join(f"This is sentence number {i} about leadership and growth." for i in range(120))


def test_long_text_is_split():
    chunks = chunk_text(LONG_TEXT)
    assert len(chunks) > 1


def test_no_empty_chunks():
    chunks = chunk_text(LONG_TEXT)
    assert all(c.strip() for c in chunks)


def test_chunks_within_max_words():
    for c in chunk_text(LONG_TEXT):
        assert count_words(c) <= CONFIG.chunking.max_chunk_words


def test_small_chunks_merged_when_possible():
    chunks = chunk_text(LONG_TEXT)
    # With many chunks, every chunk but the last should meet the minimum.
    for c in chunks[:-1]:
        assert count_words(c) >= CONFIG.chunking.min_chunk_words


def test_short_text_stays_single_chunk():
    chunks = chunk_text("Just one short sentence here.")
    assert len(chunks) == 1


def _doc() -> Document:
    return Document(
        doc_id="ted_999", title="T", speaker="S", topic="career",
        source_type="TED", source_url="", year="2020", language="en",
        original_row_id="999", raw_text=LONG_TEXT,
    )


def test_chunk_ids_are_stable_and_well_formed():
    recs1 = build_chunk_records(_doc())
    recs2 = build_chunk_records(_doc())
    ids1 = [r["chunk_id"] for r in recs1]
    assert ids1 == [r["chunk_id"] for r in recs2]          # deterministic
    assert ids1[0] == "ted_999_chunk_000"                  # expected format
    assert len(set(ids1)) == len(ids1)                     # unique


def test_content_hash_is_stable():
    assert content_hash("hello world") == content_hash("hello world")
    assert content_hash("a") != content_hash("b")
