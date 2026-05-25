"""Health checks for the LLM, embedding model, and vector store.

Used by `scripts/run_healthcheck.py` and the Streamlit app to fail gracefully
with clear messages when a local model or the index is unavailable.
"""

from __future__ import annotations

from CONFIG import CONFIG


def check_llm() -> tuple[bool, str]:
    """LLM endpoint reachable and returns a short reply."""
    from src.generation.llm_client import LLMClient

    return LLMClient().health_check()


def check_embedding() -> tuple[bool, str]:
    """Embedding model loads and produces a vector of the expected size."""
    try:
        from src.embeddings.embedding_model import EmbeddingModel

        model = EmbeddingModel()
        vec = model.embed_query("test query")
        ok = len(vec) == CONFIG.model.embedding_dim
        return (ok, f"vector dim={len(vec)} (expected {CONFIG.model.embedding_dim})")
    except Exception as exc:  # noqa: BLE001 - report any load/encode failure
        return (False, f"{type(exc).__name__}: {exc}")


def check_chroma() -> tuple[bool, str]:
    """Chroma collection is accessible and reports its chunk count."""
    try:
        from src.vectorstore.chroma_store import ChromaStore

        count = ChromaStore().count()
        if count == 0:
            return (False, "collection is empty — run: python scripts/build_index.py")
        return (True, f"{count} chunks indexed")
    except Exception as exc:  # noqa: BLE001
        return (False, f"{type(exc).__name__}: {exc}")


def run_all(include_embedding: bool = True) -> dict[str, tuple[bool, str]]:
    """Run all checks and return {name: (ok, detail)}."""
    results = {"LLM": check_llm(), "Chroma": check_chroma()}
    if include_embedding:
        results["Embedding"] = check_embedding()
    return results
