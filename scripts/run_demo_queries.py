"""Run a handful of example queries through the full pipeline (no UI).

    python scripts/run_demo_queries.py

Handy for a terminal demo or a quick end-to-end sanity check.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from CONFIG import CONFIG

if CONFIG.model.use_gpu:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", CONFIG.model.gpu_device)

DEMO_QUERIES = [
    "I feel stuck in my career and don't know what to do next.",
    "How can I become a better leader?",
    "Help",
    "What is the capital of France?",
]


def main() -> None:
    from src.controller import MentorController
    from src.embeddings.embedding_model import EmbeddingModel
    from src.retrieval.retriever import Retriever

    controller = MentorController(Retriever(EmbeddingModel()))

    for query in DEMO_QUERIES:
        result = controller.respond(query)
        print("=" * 72)
        print(f"QUERY: {query}")
        print(f"  [{result.classification.label}] {result.classification.reason}")
        if result.retrieved:
            srcs = ", ".join(f"{c.speaker} ({c.similarity:.3f})" for c in result.retrieved)
            print(f"  sources: {srcs}")
        print("-" * 72)
        print(result.answer)
        print()


if __name__ == "__main__":
    main()
