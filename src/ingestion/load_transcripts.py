"""Load raw transcripts from the two supported sources into a common shape.

  * TED talks: filtered to the curated set in data/curated_talks.csv.
  * Interviews: optional JSON files dropped into data/raw/interviews/.

Both end up as `Document` objects so the rest of the pipeline treats them
identically.  If the interviews folder is empty, the pipeline simply runs on
the TED talks — we never invent fake sources.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from CONFIG import PathConfig
from src.utils.text_utils import safe_str


@dataclass
class Document:
    doc_id: str
    title: str
    speaker: str
    topic: str          # primary "mentor topic" used as retrieval metadata
    source_type: str    # "TED" or "interview"
    source_url: str
    year: str
    language: str
    original_row_id: str
    raw_text: str       # uncleaned transcript text


def _year_from_date(date_str: str) -> str:
    """Pull the year out of a YYYY-MM-DD style date string."""
    date_str = safe_str(date_str)
    return date_str[:4] if len(date_str) >= 4 and date_str[:4].isdigit() else ""


def load_curated_topics(curated_csv: Path) -> dict[int, str]:
    """Read the curated selection CSV into {talk_id: mentor_topic}."""
    if not curated_csv.exists():
        raise FileNotFoundError(
            f"Curated talk list not found: {curated_csv}\n"
            "Generate it with: python scripts/select_curated_talks.py"
        )
    df = pd.read_csv(curated_csv)
    return {int(r.talk_id): str(r.mentor_topic) for r in df.itertuples()}


def load_ted_documents(csv_path: Path, curated: dict[int, str]) -> list[Document]:
    """Load only the curated TED talks from the dataset CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"TED dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[df["talk_id"].isin(curated.keys())]

    documents = []
    for _, row in df.iterrows():
        talk_id = int(row["talk_id"])
        year = _year_from_date(row.get("recorded_date", "")) or _year_from_date(
            row.get("published_date", "")
        )
        documents.append(
            Document(
                doc_id=f"ted_{talk_id}",
                title=safe_str(row.get("title", "")),
                speaker=safe_str(row.get("speaker_1", "")),
                topic=curated[talk_id],
                source_type="TED",
                source_url=safe_str(row.get("url", "")),
                year=year,
                language=safe_str(row.get("native_lang", "")) or "en",
                original_row_id=str(talk_id),
                raw_text=safe_str(row.get("transcript", "")),
            )
        )
    # Stable order regardless of CSV/filter ordering.
    documents.sort(key=lambda d: d.doc_id)
    return documents


def _slug(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in text]
    return "".join(keep).strip("_") or "interview"


def load_interview_documents(interviews_dir: Path) -> list[Document]:
    """Load interview transcripts from JSON files (if any exist).

    Expected JSON schema (see data/raw/interviews/README.md):
        {
          "title": "...", "speaker": "...", "topic": "career",
          "source_url": "...", "year": "2023",
          "turns": [{"speaker": "Host", "text": "..."}, ...]
        }
    A flat "transcript": "..." field is also accepted instead of "turns".
    """
    if not interviews_dir.exists():
        return []

    documents = []
    for path in sorted(interviews_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))

        if "turns" in data:
            # Preserve speaker names in interview-style data.
            raw_text = "\n".join(
                f"{safe_str(t.get('speaker', ''))}: {safe_str(t.get('text', ''))}".strip(": ")
                for t in data["turns"]
            )
        else:
            raw_text = safe_str(data.get("transcript", ""))

        documents.append(
            Document(
                doc_id=f"interview_{_slug(path.stem)}",
                title=safe_str(data.get("title", path.stem)),
                speaker=safe_str(data.get("speaker", "")),
                topic=safe_str(data.get("topic", "")) or "career",
                source_type="interview",
                source_url=safe_str(data.get("source_url", "")),
                year=safe_str(data.get("year", "")),
                language=safe_str(data.get("language", "")) or "en",
                original_row_id=path.name,
                raw_text=raw_text,
            )
        )
    return documents


def load_all_documents(paths: PathConfig | None = None) -> list[Document]:
    """Load curated TED talks (from curated_talks.csv) plus any interviews."""
    paths = paths or PathConfig()
    curated = load_curated_topics(paths.curated_talks_csv)
    ted = load_ted_documents(paths.raw_ted_csv, curated)
    interviews = load_interview_documents(paths.interviews_dir)
    return ted + interviews
