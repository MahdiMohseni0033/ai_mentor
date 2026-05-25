"""Select ~50 mentor-relevant TED talks and write data/curated_talks.csv.

The CSV (talk_id, mentor_topic, speaker, title) is the hand-editable source of
truth for the knowledge base — the ingestion pipeline reads it. A vetted SEED of
famous on-theme talks is combined with an auto-fill of high-view talks tagged
with core mentor topics (deduplicated per speaker).

    python scripts/select_curated_talks.py        # writes data/curated_talks.csv
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from CONFIG import CONFIG

# Vetted, famous, on-theme talks (talk_id -> mentor topic). Hand-curated to 50,
# balanced across topics and capped at ~2 talks per speaker.
SEED: dict[int, str] = {
    # leadership
    848: "leadership", 1040: "leadership", 1998: "leadership",
    2652: "leadership", 814: "leadership",
    # career
    70: "career", 93: "career", 453: "career", 618: "career", 1183: "career",
    1384: "career", 1682: "career", 1733: "career", 1906: "career",
    2474: "career", 1741: "career", 2341: "career",
    # confidence
    517: "confidence", 1569: "confidence", 2448: "confidence",
    2799: "confidence", 13587: "confidence", 2638: "confidence",
    # communication
    1200: "communication", 2034: "communication", 2217: "communication",
    2435: "communication", 13589: "communication",
    # productivity
    1344: "productivity", 2458: "productivity", 13585: "productivity",
    53740: "productivity", 2647: "productivity", 1640: "productivity",
    # psychology
    97: "psychology", 1042: "psychology", 1377: "psychology", 1391: "psychology",
    1815: "psychology", 1894: "psychology", 2012: "psychology", 2156: "psychology",
    2193: "psychology", 2420: "psychology", 9463: "psychology",
    # purpose
    2005: "purpose", 2399: "purpose", 2861: "purpose", 13584: "purpose",
    1728: "purpose",
}

TARGET = 50
MAX_PER_SPEAKER = 3

# High-view talks the auto-fill would grab but that are off-theme for a mentor
# (comedy / pure tech). Excluded after eyeballing the first pass.
BLOCKLIST = {2577, 2405, 2774, 24138}

# Tight tag set for auto-fill — kept conservative to stay mentor-relevant.
TIGHT_TAGS = {
    "leadership", "personal growth", "motivation", "success", "work",
    "psychology", "happiness", "communication", "productivity", "business",
}

# Dominant-tag -> mentor topic (first match wins).
TOPIC_PRIORITY = [
    ("leadership", {"leadership"}),
    ("career", {"business", "success", "work", "entrepreneur"}),
    ("productivity", {"productivity"}),
    ("communication", {"communication", "public speaking", "language"}),
    ("confidence", {"self", "confidence"}),
    ("psychology", {"psychology", "brain", "mental health", "emotions", "happiness"}),
    ("purpose", {"personal growth", "life", "philosophy", "meaning", "motivation"}),
]


def parse_tags(value) -> set[str]:
    try:
        return {str(t).lower() for t in ast.literal_eval(value)}
    except Exception:
        return set()


def topic_for(tags: set[str]) -> str:
    for topic, keys in TOPIC_PRIORITY:
        if tags & keys:
            return topic
    return "personal growth"


def main() -> None:
    df = pd.read_csv(CONFIG.paths.raw_ted_csv)
    df["tagset"] = df["topics"].apply(parse_tags)

    rows: list[dict] = []
    speaker_count: dict[str, int] = {}

    def add(talk_id: int, topic: str) -> None:
        r = df[df["talk_id"] == talk_id]
        if r.empty:
            return
        r = r.iloc[0]
        speaker = str(r["speaker_1"])
        speaker_count[speaker] = speaker_count.get(speaker, 0) + 1
        rows.append({"talk_id": int(talk_id), "mentor_topic": topic,
                     "speaker": speaker, "title": str(r["title"])})

    # 1) seed (in id order for stability)
    for tid in sorted(SEED):
        add(tid, SEED[tid])

    chosen = {r["talk_id"] for r in rows}

    # 2) auto-fill from tagged, high-view talks, deduped per speaker
    candidates = df[df["tagset"].apply(lambda s: bool(s & TIGHT_TAGS))]
    candidates = candidates.sort_values("views", ascending=False)
    for _, r in candidates.iterrows():
        if len(rows) >= TARGET:
            break
        tid = int(r["talk_id"])
        speaker = str(r["speaker_1"])
        if tid in chosen or tid in BLOCKLIST:
            continue
        if speaker_count.get(speaker, 0) >= MAX_PER_SPEAKER:
            continue
        add(tid, topic_for(r["tagset"]))
        chosen.add(tid)

    out = pd.DataFrame(rows).sort_values(["mentor_topic", "talk_id"])
    out.to_csv(CONFIG.paths.curated_talks_csv, index=False)

    print(f"Wrote {len(out)} talks to {CONFIG.paths.curated_talks_csv}\n")
    print(out.to_string(index=False))
    print("\nby topic:", out["mentor_topic"].value_counts().to_dict())


if __name__ == "__main__":
    main()
