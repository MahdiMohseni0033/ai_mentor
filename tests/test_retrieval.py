"""Retrieval integration test against the built index (skips if missing)."""

from CONFIG import CONFIG
from src.retrieval.retriever import RetrievedChunk


def test_returns_results_with_metadata_and_scores(retriever):
    results = retriever.retrieve("How can I become a better leader?")
    assert results, "expected at least one hit"
    for r in results:
        assert isinstance(r, RetrievedChunk)
        assert r.chunk_id and r.doc_id and r.title and r.speaker
        assert -1.0 <= r.similarity <= 1.0
        assert r.text.strip()


def test_default_top_k_is_three(retriever):
    assert CONFIG.retrieval.top_k == 3
    assert len(retriever.retrieve("leadership")) <= 3


def test_top_k_is_configurable(retriever):
    assert len(retriever.retrieve("confidence", top_k=1)) == 1
    assert len(retriever.retrieve("confidence", top_k=5)) <= 5


def test_results_sorted_by_similarity(retriever):
    sims = [r.similarity for r in retriever.retrieve("how to focus", top_k=5)]
    assert sims == sorted(sims, reverse=True)


def test_leadership_query_retrieves_leadership_talk(retriever):
    results = retriever.retrieve("How do great leaders inspire their teams?")
    assert any(r.speaker == "Simon Sinek" for r in results)
