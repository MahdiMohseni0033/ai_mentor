"""Interactive terminal chat with the mentor (the Streamlit demo, in your shell).

    python chat_cli.py

Same functionality as app.py: conversation memory, the decision-engine label,
and the retrieved chunks + similarity scores behind every grounded answer.

Commands:
    /sources    toggle showing retrieved chunks
    /examples   list example queries
    /clear      reset the conversation
    /help       show commands
    /exit       quit  (Ctrl-D / Ctrl-C also work)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from CONFIG import CONFIG

if CONFIG.model.use_gpu:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", CONFIG.model.gpu_device)

# Minimal ANSI styling (harmless if the terminal ignores it).
BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
CYAN, GREEN = "\033[36m", "\033[32m"

LABEL_ICON = {"emotional": "💙", "strategic": "🎯", "vague": "❓",
              "general": "💬", "off_topic": "🧭", "follow_up": "🔁",
              "appreciation": "✅"}

EXAMPLE_QUERIES = [
    "I feel stuck in my career and don't know what to do next.",
    "How can I become a better leader?",
    "Give me a 7-day plan to improve my focus.",
    "Help",
    "How do I find my purpose?",
    "What is the capital of France?",
]


def build_controller():
    from src.controller import MentorController
    from src.embeddings.embedding_model import EmbeddingModel
    from src.retrieval.retriever import Retriever
    from src.vectorstore.chroma_store import ChromaStore

    store = ChromaStore()
    if store.count() == 0:
        raise SystemExit(
            "The vector index is empty. Build it first:\n"
            "  python -m src.ingestion.pipeline\n"
            "  python scripts/build_index.py"
        )
    return MentorController(Retriever(EmbeddingModel(), store))


def print_sources(chunks) -> None:
    if not chunks:
        return
    print(f"{DIM}retrieved context:{RESET}")
    for c in chunks:
        print(f"  {GREEN}{c.similarity:.3f}{RESET} · {c.speaker} — "
              f"\"{c.title}\" {DIM}({c.chunk_id}, topic={c.topic}){RESET}")


def print_help() -> None:
    print(f"{DIM}commands: /sources  /examples  /clear  /help  /exit{RESET}")


def main() -> None:
    from src.generation.llm_client import LLMUnavailableError

    print("Loading embedding model and vector store (~10s)...")
    controller = build_controller()

    show_sources = CONFIG.app.show_retrieved_chunks
    history: list[dict] = []

    print(f"\n{BOLD}🧭 Mini AI Mentor Engine — terminal chat{RESET}")
    print(f"{DIM}LLM={CONFIG.model.llm_model} · top_k={CONFIG.retrieval.top_k}{RESET}")
    print_help()

    while True:
        try:
            query = input(f"\n{BOLD}You ›{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query in ("/exit", "/quit"):
            print("Goodbye.")
            break
        if query == "/help":
            print_help()
            continue
        if query == "/clear":
            history = []
            print(f"{DIM}(conversation cleared){RESET}")
            continue
        if query == "/sources":
            show_sources = not show_sources
            print(f"{DIM}(sources display: {'on' if show_sources else 'off'}){RESET}")
            continue
        if query == "/examples":
            for q in EXAMPLE_QUERIES:
                print(f"  - {q}")
            continue

        try:
            result = controller.respond(query, history)
        except LLMUnavailableError as exc:
            print(f"\n[LLM error] {exc}")
            continue

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": result.answer})

        cls = result.classification
        icon = LABEL_ICON.get(cls.label, "•")
        print(f"\n{CYAN}{icon} {cls.label}{RESET} {DIM}— {cls.reason}{RESET}")
        print(f"\n{BOLD}Mentor ›{RESET}\n{result.answer}\n")
        if show_sources:
            print_sources(result.retrieved)


if __name__ == "__main__":
    main()
