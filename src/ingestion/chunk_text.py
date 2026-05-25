"""Paragraph/sentence-aware chunking.

Strategy (see architecture.md for the reasoning):
  * Pack whole sentences until ~chunk_size words, so chunks never cut mid-sentence.
  * Carry ~chunk_overlap words of trailing sentences into the next chunk for context.
  * Merge chunks shorter than min_chunk_words into the previous chunk.
  * Hard-split any single sentence longer than max_chunk_words (rare).
"""

from __future__ import annotations

from CONFIG import CONFIG
from src.utils.text_utils import count_words, normalize_spaces, split_sentences


def _split_long_span(text: str, max_words: int, overlap_words: int) -> list[str]:
    """Word-window fallback for a span longer than max_words."""
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(normalize_spaces(" ".join(words[start:end])))
        if end >= len(words):
            break
        start = max(0, end - overlap_words)
    return chunks


def _trailing_overlap(sentences: list[str], overlap_words: int) -> list[str]:
    """Return the last sentences that fit within overlap_words."""
    overlap, total = [], 0
    for sentence in reversed(sentences):
        words = count_words(sentence)
        if total + words > overlap_words:
            break
        overlap.insert(0, sentence)
        total += words
    return overlap


def _merge_small(chunks: list[str], min_words: int) -> list[str]:
    merged: list[str] = []
    for chunk in chunks:
        chunk = normalize_spaces(chunk)
        if not chunk:
            continue
        if merged and count_words(chunk) < min_words:
            merged[-1] = normalize_spaces(merged[-1] + " " + chunk)
        else:
            merged.append(chunk)
    return merged


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    min_chunk_words: int | None = None,
    max_chunk_words: int | None = None,
) -> list[str]:
    """Split cleaned text into overlapping, sentence-aligned chunks."""
    chunk_size = chunk_size or CONFIG.chunking.chunk_size
    chunk_overlap = chunk_overlap or CONFIG.chunking.chunk_overlap
    min_chunk_words = min_chunk_words or CONFIG.chunking.min_chunk_words
    max_chunk_words = max_chunk_words or CONFIG.chunking.max_chunk_words

    sentences = split_sentences(text)
    if not sentences:
        return _split_long_span(text, chunk_size, chunk_overlap)

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        words = count_words(sentence)

        # A single very long sentence: flush, then hard-split it.
        if words > max_chunk_words:
            if current:
                chunks.append(normalize_spaces(" ".join(current)))
                current, current_words = [], 0
            chunks.extend(_split_long_span(sentence, max_chunk_words, chunk_overlap))
            continue

        # Adding this sentence would exceed the target -> close the chunk and
        # start the next one seeded with overlap sentences.
        if current and current_words + words > chunk_size:
            chunks.append(normalize_spaces(" ".join(current)))
            current = _trailing_overlap(current, chunk_overlap)
            current_words = count_words(" ".join(current))

        current.append(sentence)
        current_words += words

    if current:
        chunks.append(normalize_spaces(" ".join(current)))

    return _merge_small(chunks, min_chunk_words)
