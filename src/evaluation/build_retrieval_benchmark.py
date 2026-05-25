"""Build a small retrieval benchmark.

For a deterministic sample of chunks (one per talk), we ask the local LLM to
write a natural, paraphrased question that the chunk would answer — without
copying its wording.  Each query is saved with the chunk/doc it should recover,
so retrieval quality can be measured later (see evaluate_retrieval.py).

    python -m src.evaluation.build_retrieval_benchmark [--force]

The benchmark is written once to data/evaluation/retrieval_benchmark.json and
reused; pass --force to regenerate.  If the LLM is unavailable, a small manual
benchmark (doc-level targets) is written instead.
"""

from __future__ import annotations

import argparse
import json
import random

from CONFIG import CONFIG
from src.generation.llm_client import LLMClient, LLMUnavailableError

PARAPHRASE_PROMPT = (
    "Below is an excerpt from a talk. Write ONE natural question that a person "
    "seeking advice might ask, which this excerpt would answer well.\n"
    "Rules: first person, conversational, ONE sentence ending with '?', do NOT "
    "copy distinctive words or phrases from the excerpt, and do NOT mention any "
    "names or talk titles.\n\nExcerpt:\n{excerpt}\n\nQuestion:"
)

# Used only if the LLM cannot be reached. Doc-level targets are robust to
# chunking changes; chunk-level fields are left null.
MANUAL_BENCHMARK = [
    ("How do I get people to truly buy into my vision as a leader?", "ted_848", "leadership"),
    ("Why do I feel like I have to hide my imperfections from others?", "ted_1042", "psychology"),
    ("Does raw talent matter more than sticking with something long-term?", "ted_1733", "career"),
    ("I keep putting off important work until the last minute — why?", "ted_2458", "productivity"),
    ("Can being quiet and reserved actually be a strength at work?", "ted_1377", "psychology"),
    ("Is there a quick way to feel more confident before a big moment?", "ted_1569", "confidence"),
    ("What actually makes people happy and healthy over a whole lifetime?", "ted_2399", "purpose"),
    ("Is chasing happiness really the point, or is there something more?", "ted_2861", "purpose"),
    ("How should I think about stress instead of just fearing it?", "ted_1815", "psychology"),
    ("Why do I talk myself out of pursuing the career I really want?", "ted_1384", "career"),
]


def load_chunks() -> list[dict]:
    path = CONFIG.paths.chunks_jsonl
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run: python -m src.ingestion.pipeline"
        )
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def sample_chunks(chunks: list[dict], n: int, seed: int) -> list[dict]:
    """Deterministically pick one substantive chunk from each of n talks."""
    by_doc: dict[str, list[dict]] = {}
    for c in chunks:
        by_doc.setdefault(c["metadata"]["doc_id"], []).append(c)

    doc_ids = sorted(by_doc)
    rng = random.Random(seed)
    rng.shuffle(doc_ids)

    selected = []
    for doc_id in doc_ids[:n]:
        doc_chunks = sorted(by_doc[doc_id], key=lambda c: c["metadata"]["chunk_index"])
        # Prefer a middle chunk (skip the intro) for a more "answerable" excerpt.
        selected.append(doc_chunks[len(doc_chunks) // 2])
    return selected


def make_query(llm: LLMClient, chunk: dict) -> str:
    prompt = PARAPHRASE_PROMPT.format(excerpt=chunk["text"][:1200])
    reply = llm.chat([{"role": "user", "content": prompt}]).strip()
    return reply.strip().strip('"').splitlines()[0].strip() if reply else ""


def build_with_llm(chunks: list[dict]) -> list[dict]:
    llm = LLMClient(think=False, num_predict=80, temperature=0.3)
    selected = sample_chunks(chunks, CONFIG.evaluation.num_benchmark_queries,
                             CONFIG.evaluation.random_seed)
    benchmark = []
    for chunk in selected:
        meta = chunk["metadata"]
        query = make_query(llm, chunk)
        if not query:
            continue
        benchmark.append({
            "query": query,
            "expected_chunk_id": meta["chunk_id"],
            "expected_doc_id": meta["doc_id"],
            "topic": meta["topic"],
            "speaker": meta["speaker"],
            "title": meta["title"],
            "notes": "LLM-paraphrased from the expected chunk.",
        })
    return benchmark


def build_manual() -> list[dict]:
    return [
        {
            "query": q, "expected_chunk_id": None, "expected_doc_id": doc_id,
            "topic": topic, "speaker": "", "title": "",
            "notes": "Manual fallback (LLM unavailable); doc-level target only.",
        }
        for q, doc_id, topic in MANUAL_BENCHMARK
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="regenerate even if it exists")
    args = parser.parse_args()

    out = CONFIG.paths.retrieval_benchmark
    if out.exists() and not args.force:
        print(f"Benchmark already exists: {out}\nUse --force to regenerate.")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks()
    try:
        benchmark = build_with_llm(chunks)
        source = "LLM-paraphrased"
    except LLMUnavailableError as exc:
        print(f"LLM unavailable ({exc}); writing manual benchmark.")
        benchmark = build_manual()
        source = "manual"

    out.write_text(json.dumps(benchmark, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(benchmark)} {source} benchmark queries to {out}")


if __name__ == "__main__":
    main()
