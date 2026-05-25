"""Evaluate the retriever against the benchmark.

    python -m src.evaluation.evaluate_retrieval

For every benchmark query we retrieve the top-k chunks and check whether the
expected chunk (and its talk) is recovered. Reports hit@1/3/5, MRR, and a
doc-level hit@k that is robust to chunk-boundary changes.
"""

from __future__ import annotations

import json

from CONFIG import CONFIG


def load_benchmark() -> list[dict]:
    path = CONFIG.paths.retrieval_benchmark
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run: python -m src.evaluation.build_retrieval_benchmark"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _first_rank(targets: list[str], expected: str) -> int | None:
    """1-based rank of expected in targets, or None if absent."""
    for i, t in enumerate(targets, 1):
        if t == expected:
            return i
    return None


def main() -> None:
    from src.embeddings.embedding_model import EmbeddingModel
    from src.retrieval.retriever import Retriever

    k_values = CONFIG.evaluation.hit_k_values
    max_k = max(k_values)
    benchmark = load_benchmark()
    retriever = Retriever(EmbeddingModel())

    chunk_ranks: list[int | None] = []   # only for queries with a chunk-level target
    doc_ranks: list[int | None] = []

    print(f"Evaluating {len(benchmark)} queries (top_k={max_k})\n")
    for item in benchmark:
        hits = retriever.retrieve(item["query"], top_k=max_k)
        chunk_ids = [h.chunk_id for h in hits]
        doc_ids = [h.doc_id for h in hits]

        doc_rank = _first_rank(doc_ids, item["expected_doc_id"])
        doc_ranks.append(doc_rank)
        chunk_rank = None
        if item.get("expected_chunk_id"):
            chunk_rank = _first_rank(chunk_ids, item["expected_chunk_id"])
            chunk_ranks.append(chunk_rank)

        passed = (chunk_rank is not None and chunk_rank <= 3) or (
            item.get("expected_chunk_id") is None and doc_rank is not None and doc_rank <= 3
        )
        print(f"{'PASS' if passed else 'MISS'} hit@3 | {item['query']}")
        print(f"     expected_chunk={item.get('expected_chunk_id')} doc={item['expected_doc_id']}")
        print(f"     retrieved={[f'{c}:{round(h.similarity,3)}' for c, h in zip(chunk_ids, hits)]}")
        print()

    def hit_at(ranks: list[int | None], k: int) -> float:
        if not ranks:
            return 0.0
        return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)

    def mrr(ranks: list[int | None]) -> float:
        if not ranks:
            return 0.0
        return sum((1.0 / r) if r else 0.0 for r in ranks) / len(ranks)

    print("=" * 60)
    if chunk_ranks:
        print("Chunk-level (exact chunk recovery):")
        for k in k_values:
            print(f"  hit@{k}: {hit_at(chunk_ranks, k):.3f}  ({len(chunk_ranks)} queries)")
        print(f"  MRR:   {mrr(chunk_ranks):.3f}")
    print("Doc-level (correct talk recovery):")
    for k in k_values:
        print(f"  hit@{k}: {hit_at(doc_ranks, k):.3f}  ({len(doc_ranks)} queries)")
    print(f"  MRR:   {mrr(doc_ranks):.3f}")


if __name__ == "__main__":
    main()
