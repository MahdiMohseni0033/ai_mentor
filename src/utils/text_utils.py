"""Small, deterministic text helpers shared across the pipeline.

These functions never call an LLM — cleaning and splitting must be reproducible.
"""

from __future__ import annotations

import math
import re

# A "word" is a run of letters/digits, allowing internal hyphens/apostrophes
# (so "low-hanging" and "don't" count as one word each).
WORD_PATTERN = re.compile(r"\b[\w]+(?:[-'][\w]+)*\b", re.UNICODE)

# Abbreviations whose trailing period must not be treated as a sentence end.
ABBREVIATIONS = [
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.",
    "U.S.", "U.K.", "U.N.", "E.U.", "e.g.", "i.e.", "etc.", "vs.",
]


def is_missing(value) -> bool:
    """True for None, NaN, or blank/"nan"/"none"/"null" strings."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "null"}:
        return True
    return False


def safe_str(value) -> str:
    """Coerce any cell value to a clean string ('' for missing)."""
    if is_missing(value):
        return ""
    return str(value).strip()


def normalize_spaces(text: str) -> str:
    """Collapse whitespace runs and tidy spacing before punctuation."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def count_words(text: str) -> int:
    return len(WORD_PATTERN.findall(text))


def _protect_abbreviations(text: str) -> tuple[str, dict[str, str]]:
    protected = text
    mapping: dict[str, str] = {}
    for i, abbr in enumerate(ABBREVIATIONS):
        token = f"__ABBR_{i}__"
        mapping[token] = abbr
        protected = protected.replace(abbr, token)
    return protected, mapping


def _restore_abbreviations(text: str, mapping: dict[str, str]) -> str:
    for token, abbr in mapping.items():
        text = text.replace(token, abbr)
    return text


def split_sentences(text: str) -> list[str]:
    """Split text into sentences without cutting common abbreviations.

    Splits on .!? followed by whitespace and an opening quote/capital/digit.
    """
    text = normalize_spaces(text)
    if not text:
        return []

    protected, mapping = _protect_abbreviations(text)
    parts = re.split(r'(?<=[.!?])\s+(?=["\'A-Z0-9])', protected)

    sentences = []
    for part in parts:
        sentence = normalize_spaces(_restore_abbreviations(part, mapping))
        if sentence and count_words(sentence) > 0:
            sentences.append(sentence)
    return sentences
