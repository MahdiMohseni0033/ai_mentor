"""Offline ingestion pipeline: raw transcripts -> clean -> chunk -> processed files.

Run it with:
    python -m src.ingestion.pipeline

Outputs (under data/processed/):
    chunks.jsonl              one record per chunk (text + metadata)
    documents.csv             one row per source document
    preprocessing_summary.json
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from CONFIG import CONFIG, PathConfig
from src.ingestion.chunk_text import chunk_text
from src.ingestion.clean_text import clean_transcript
from src.ingestion.load_transcripts import Document, load_all_documents
from src.utils.text_utils import count_words


def content_hash(text: str) -> str:
    """Stable hash of chunk text — used to detect changes for smart resume."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_chunk_records(doc: Document) -> list[dict]:
    """Clean one document and turn it into chunk records with metadata."""
    cleaned = clean_transcript(doc.raw_text)
    if not cleaned:
        return []

    records = []
    for i, text in enumerate(chunk_text(cleaned)):
        chunk_id = f"{doc.doc_id}_chunk_{i:03d}"
        records.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "metadata": {
                    "chunk_id": chunk_id,
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "speaker": doc.speaker,
                    "topic": doc.topic,
                    "source_type": doc.source_type,
                    "source_url": doc.source_url,
                    "year": doc.year,
                    "language": doc.language,
                    "original_row_id": doc.original_row_id,
                    "chunk_index": i,
                    "word_count": count_words(text),
                    "content_hash": content_hash(text),
                    "preprocessing_version": CONFIG.chunking.preprocessing_version,
                },
            }
        )
    return records


def run_pipeline(paths: PathConfig | None = None) -> dict:
    """Run the full pipeline and write processed files. Returns the summary."""
    paths = paths or CONFIG.paths
    paths.processed_dir.mkdir(parents=True, exist_ok=True)

    documents = load_all_documents(paths)

    all_chunks: list[dict] = []
    doc_rows: list[dict] = []
    for doc in documents:
        chunks = build_chunk_records(doc)
        if not chunks:
            continue
        all_chunks.extend(chunks)
        doc_rows.append(
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "speaker": doc.speaker,
                "topic": doc.topic,
                "source_type": doc.source_type,
                "source_url": doc.source_url,
                "year": doc.year,
                "num_chunks": len(chunks),
                "word_count": sum(c["metadata"]["word_count"] for c in chunks),
            }
        )

    # Write chunks.jsonl
    with paths.chunks_jsonl.open("w", encoding="utf-8") as f:
        for record in all_chunks:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Write documents.csv
    pd.DataFrame(doc_rows).to_csv(paths.documents_csv, index=False, encoding="utf-8")

    word_counts = [c["metadata"]["word_count"] for c in all_chunks]
    summary = {
        "preprocessing_version": CONFIG.chunking.preprocessing_version,
        "num_documents": len(doc_rows),
        "num_ted": sum(r["source_type"] == "TED" for r in doc_rows),
        "num_interviews": sum(r["source_type"] == "interview" for r in doc_rows),
        "num_chunks": len(all_chunks),
        "chunk_size_target": CONFIG.chunking.chunk_size,
        "chunk_overlap": CONFIG.chunking.chunk_overlap,
        "min_chunk_words": CONFIG.chunking.min_chunk_words,
        "max_chunk_words": CONFIG.chunking.max_chunk_words,
        "word_count_min": min(word_counts) if word_counts else 0,
        "word_count_max": max(word_counts) if word_counts else 0,
        "word_count_mean": round(sum(word_counts) / len(word_counts), 1) if word_counts else 0,
        "outputs": {
            "chunks_jsonl": str(paths.chunks_jsonl),
            "documents_csv": str(paths.documents_csv),
        },
    }
    paths.preprocessing_summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    summary = run_pipeline()
    print("Ingestion complete.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
