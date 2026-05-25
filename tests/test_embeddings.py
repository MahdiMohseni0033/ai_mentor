"""Embedding model integration test (skips if the model can't load)."""

import math

from CONFIG import CONFIG


def test_query_embedding_shape_and_norm(embedding_model):
    vec = embedding_model.embed_query("How can I become a better leader?")
    assert len(vec) == CONFIG.model.embedding_dim
    # Vectors are L2-normalized, so the norm should be ~1.
    assert math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, abs_tol=1e-2)


def test_document_embeddings_batch(embedding_model):
    vecs = embedding_model.embed_documents(["first passage", "second passage"])
    assert len(vecs) == 2
    assert all(len(v) == CONFIG.model.embedding_dim for v in vecs)


def test_related_texts_are_more_similar(embedding_model):
    q = embedding_model.embed_query("how to lead a team")
    related = embedding_model.embed_documents(["Good leaders inspire their teams."])[0]
    unrelated = embedding_model.embed_documents(["The recipe needs two eggs and flour."])[0]
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    assert dot(q, related) > dot(q, unrelated)
