"""Pytest setup: make the project root importable, pin the GPU, and share the
embedding model / retriever across the (slow) integration tests."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from CONFIG import CONFIG  # noqa: E402

if CONFIG.model.use_gpu:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", CONFIG.model.gpu_device)


@pytest.fixture(scope="session")
def embedding_model():
    """Load the embedding model once; skip dependent tests if it can't load."""
    try:
        from src.embeddings.embedding_model import EmbeddingModel

        return EmbeddingModel()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"embedding model unavailable: {exc}")


@pytest.fixture(scope="session")
def retriever(embedding_model):
    """A retriever over the built index; skip if the index is empty/missing."""
    from src.retrieval.retriever import Retriever
    from src.vectorstore.chroma_store import ChromaStore

    store = ChromaStore()
    if store.count() == 0:
        pytest.skip("Chroma index is empty — run scripts/build_index.py")
    return Retriever(embedding_model, store)
