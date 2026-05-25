# Interview transcripts (optional)

The curated knowledge base currently uses **TED talks only** — the local
dataset (`ted-ultimate-dataset/`) contains TED transcripts and no interviews.
We do **not** invent fake sources.

The pipeline already supports interview-style transcripts: drop one JSON file
per interview into this folder and they will be cleaned, chunked, and indexed
alongside the TED talks (with `source_type = "interview"`).

## JSON schema

```json
{
  "title": "On building a career you love",
  "speaker": "Jane Doe",
  "topic": "career",
  "source_url": "https://example.com/interview",
  "year": "2023",
  "turns": [
    {"speaker": "Host", "text": "How did you find your direction?"},
    {"speaker": "Jane Doe", "text": "I followed what energized me..."}
  ]
}
```

* `turns` preserves speaker names (good for interview-style Q&A).
* Alternatively, provide a flat `"transcript": "..."` field instead of `turns`.
* `topic` should be one of the supported mentor topics (career, leadership,
  psychology, productivity, confidence, purpose, personal growth).

After adding files, rebuild:

```bash
python -m src.ingestion.pipeline
python scripts/build_index.py
```
