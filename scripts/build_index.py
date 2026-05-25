"""Build (or incrementally update) the Chroma vector index.

    python scripts/build_index.py [--rebuild]

Smart resume: a chunk is re-embedded only if it is missing from the collection,
or its content_hash / preprocessing_version / embedding_model changed.  Unchanged
chunks are skipped, so re-running after an interrupted build only does the
remaining work.  Use --rebuild to force re-embedding everything.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `python scripts/build_index.py`

from CONFIG import CONFIG
from src.vectorstore.chroma_store import ChromaStore


def load_chunks(jsonl_path: Path) -> list[dict]:
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Processed chunks not found: {jsonl_path}\n"
            "Run the ingestion pipeline first:  python -m src.ingestion.pipeline"
        )
    return [json.loads(line) for line in jsonl_path.open(encoding="utf-8")]


def needs_embedding(chunk: dict, existing: dict | None, model_name: str) -> bool:
    """Decide whether a chunk must be (re)embedded."""
    if existing is None:
        return True
    meta = chunk["metadata"]
    return (
        existing.get("content_hash") != meta["content_hash"]
        or existing.get("preprocessing_version") != meta["preprocessing_version"]
        or existing.get("embedding_model") != model_name
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Chroma vector index.")
    parser.add_argument("--rebuild", action="store_true", help="re-embed all chunks")
    args = parser.parse_args()

    model_name = CONFIG.model.embedding_model
    chunks = load_chunks(CONFIG.paths.chunks_jsonl)
    store = ChromaStore()

    existing = {} if args.rebuild else store.get_existing_metadata(
        [c["chunk_id"] for c in chunks]
    )

    todo = [c for c in chunks if needs_embedding(c, existing.get(c["chunk_id"]), model_name)]
    skipped = len(chunks) - len(todo)

    print(f"chunks found:     {len(chunks)}")
    print(f"already indexed:  {skipped}")
    print(f"to embed:         {len(todo)}")

    embedded = 0
    if todo:
        # Load the embedding model only when there is real work to do.
        from src.embeddings.embedding_model import EmbeddingModel

        print(f"loading embedding model '{model_name}' ...")
        t0 = time.time()
        model = EmbeddingModel()
        print(f"model loaded in {time.time() - t0:.1f}s (dim={model.dim})")

        texts = [c["text"] for c in todo]
        embeddings = model.embed_documents(texts)
        metadatas = []
        for c in todo:
            meta = dict(c["metadata"])
            meta["embedding_model"] = model_name  # for future resume checks
            metadatas.append(meta)

        store.upsert(
            ids=[c["chunk_id"] for c in todo],
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        embedded = len(todo)

    # Write a manifest describing the built index.
    manifest = {
        "collection_name": CONFIG.retrieval.collection_name,
        "embedding_model": model_name,
        "embedding_dim": CONFIG.model.embedding_dim,
        "preprocessing_version": CONFIG.chunking.preprocessing_version,
        "num_chunks_total": len(chunks),
        "num_embedded_this_run": embedded,
        "num_skipped_this_run": skipped,
        "collection_count": store.count(),
        "build_time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    CONFIG.paths.index_manifest.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"embedded this run: {embedded}")
    print(f"collection count:  {store.count()}")
    print(f"manifest written:  {CONFIG.paths.index_manifest}")


if __name__ == "__main__":
    main()
