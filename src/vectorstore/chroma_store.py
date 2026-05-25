"""Thin wrapper around a persistent ChromaDB collection.

Kept embedded (PersistentClient) for demo simplicity — it could be promoted to
the Chroma server, but that is unnecessary here.  The collection uses cosine
space, so for our L2-normalized vectors: similarity = 1 - distance.
"""

from __future__ import annotations

import chromadb

from CONFIG import CONFIG


class ChromaStore:
    def __init__(self, persist_dir=None, collection_name: str | None = None):
        persist_dir = persist_dir or CONFIG.paths.chroma_dir
        collection_name = collection_name or CONFIG.retrieval.collection_name
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": CONFIG.retrieval.distance_metric},
        )

    def count(self) -> int:
        return self.collection.count()

    def get_existing_metadata(self, ids: list[str]) -> dict[str, dict]:
        """Return {id: metadata} for ids already present (for smart resume)."""
        if not ids:
            return {}
        result = self.collection.get(ids=ids, include=["metadatas"])
        return dict(zip(result["ids"], result["metadatas"]))

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        if not ids:
            return
        self.collection.upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def query(self, embedding: list[float], top_k: int) -> list[dict]:
        """Return up to top_k hits as dicts with text, metadata, and distance."""
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append({"text": doc, "metadata": meta, "distance": dist})
        return hits
