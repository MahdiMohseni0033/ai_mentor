"""Deterministic transcript cleaning.

We deliberately do NOT use an LLM to rewrite transcripts — cleaning must be
reproducible and must not change the meaning of what a speaker said.
"""

from __future__ import annotations

import re
import unicodedata

from src.utils.text_utils import normalize_spaces, safe_str

# Stage directions in TED transcripts, e.g. "(Laughter)", "[Applause]", "(Music)".
_STAGE_DIRECTION = re.compile(
    r"[\(\[][^)\]]*"
    r"(laughter|applause|laughs|laughing|cheers|cheering|music|song|singing|"
    r"video|audio|inaudible|mock sob|sob|clapping|audience)"
    r"[^)\]]*[\)\]]",
    flags=re.IGNORECASE,
)

# Timestamps like "00:12", "1:02:33", "[00:12:30]".
_TIMESTAMP = re.compile(r"\[?\b\d{1,2}:\d{2}(?::\d{2})?\b\]?")


def clean_transcript(text: str) -> str:
    """Normalize a raw transcript into clean prose.

    Steps: Unicode NFKC, repair line-wrapped hyphens, strip HTML, remove stage
    directions and timestamps, normalize quotes, collapse whitespace.
    """
    text = safe_str(text)
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # "low-\nhanging" -> "low-hanging" (line-wrap artifacts)
    text = re.sub(r"([A-Za-z])-\s*\n\s*([A-Za-z])", r"\1-\2", text)

    text = re.sub(r"<[^>]+>", " ", text)          # HTML tags
    text = _STAGE_DIRECTION.sub(" ", text)         # (Laughter), [Applause], ...
    text = _TIMESTAMP.sub(" ", text)               # 00:12, [01:02:33]
    text = re.sub(r"\n+", " ", text)               # remaining newlines -> spaces

    # Normalize smart quotes / dashes lightly (don't change wording).
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")

    return normalize_spaces(text)
