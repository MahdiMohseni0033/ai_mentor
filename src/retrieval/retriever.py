"""Query embedding + top-k retrieval from Chroma.

The embedder is injected so the same retriever works with an in-process model
(offline / tests) or, in principle, any object exposing `embed_query`.
"""

from __future__ import annotations

from dataclasses import dataclass

from CONFIG import CONFIG
from src.vectorstore.chroma_store import ChromaStore


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    title: str
    speaker: str
    topic: str
    source_type: str
    source_url: str
    text: str
    similarity: float   # cosine similarity in [-1, 1]; higher = more relevant
    distance: float     # raw cosine distance from Chroma (1 - similarity)

    def source_label(self) -> str:
        return f"{self.speaker}, \"{self.title}\" ({self.chunk_id})"


class Retriever:
    def __init__(self, embedder, store: ChromaStore | None = None):
        self.embedder = embedder
        self.store = store or ChromaStore()

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or CONFIG.retrieval.top_k
        query_vector = self.embedder.embed_query(query)
        hits = self.store.query(query_vector, top_k=top_k)

        results = []
        for hit in hits:
            meta = hit["metadata"]
            distance = float(hit["distance"])
            results.append(
                RetrievedChunk(
                    chunk_id=meta.get("chunk_id", ""),
                    doc_id=meta.get("doc_id", ""),
                    title=meta.get("title", ""),
                    speaker=meta.get("speaker", ""),
                    topic=meta.get("topic", ""),
                    source_type=meta.get("source_type", ""),
                    source_url=meta.get("source_url", ""),
                    text=hit["text"],
                    similarity=round(1.0 - distance, 3),  # cosine space
                    distance=round(distance, 3),
                )
            )
        return results
