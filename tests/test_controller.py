"""Controller routing tests using injected fakes (no models / no Ollama).

The decision layer is the LLM turn router. Here we inject a FakeLLM that returns
a canned router JSON for the routing call (``format="json"``) and a stub answer
for the grounded-answer call, so we can assert the controller's handling of each
label deterministically. Fallback-to-rules is covered separately.
"""

import json

import pytest

from src.controller import MentorController
from src.retrieval.retriever import RetrievedChunk


class FakeRetriever:
    def __init__(self):
        self.calls = 0
        self.last_query = None

    def retrieve(self, query, top_k=None):
        self.calls += 1
        self.last_query = query
        return [RetrievedChunk(
            chunk_id="ted_848_chunk_009", doc_id="ted_848",
            title="How great leaders inspire action", speaker="Simon Sinek",
            topic="leadership", source_type="TED", source_url="",
            text="Start with why.", similarity=0.5, distance=0.5,
        )]


class FakeLLM:
    """Returns the canned router decision on the json call; stub answer otherwise.

    Set ``router`` to a dict (the routing decision) or to a string to emit raw
    text for the routing call (e.g. invalid JSON to exercise the fallback). Set
    ``raise_on_router=True`` to simulate the router LLM being unavailable.
    """

    def __init__(self, router=None, raise_on_router=False):
        self.router = router
        self.raise_on_router = raise_on_router
        self.last_messages = None     # last *answer* messages (None if no answer call)
        self.answer_calls = 0

    def chat(self, messages, format=None):
        if format == "json":
            if self.raise_on_router:
                raise RuntimeError("router unavailable")
            return self.router if isinstance(self.router, str) else json.dumps(self.router)
        self.last_messages = messages
        self.answer_calls += 1
        return "stub answer"


def make_controller(router=None, raise_on_router=False):
    retr = FakeRetriever()
    llm = FakeLLM(router=router, raise_on_router=raise_on_router)
    return MentorController(retr, llm), retr, llm


def _decision(label, needs, query="", reason="r"):
    return {"label": label, "needs_retrieval": needs, "search_query": query, "reason": reason}


# --- router-driven paths ---------------------------------------------------

@pytest.mark.parametrize("label", ["emotional", "strategic", "general"])
def test_retrieval_labels_retrieve_with_router_query(label):
    ctrl, retr, llm = make_controller(_decision(label, True, "resolved search query"))
    res = ctrl.respond("How can I become a better leader?")
    assert res.classification.label == label
    assert res.used_retrieval is True
    assert retr.calls == 1
    assert retr.last_query == "resolved search query"   # router's standalone query
    assert llm.answer_calls == 1
    assert res.retrieved


def test_followup_skips_retrieval():
    ctrl, retr, llm = make_controller(_decision("follow_up", False))
    history = [
        {"role": "user", "content": "I get stuck on my business planning."},
        {"role": "assistant", "content": "Long structured advice about planning..."},
    ]
    res = ctrl.respond("could you put that in 2 sentences?", history)
    assert res.classification.label == "follow_up"
    assert res.used_retrieval is False
    assert res.retrieved == []
    assert retr.calls == 0                       # no new retrieval
    assert llm.answer_calls == 1                 # answered from history
    user_turn = llm.last_messages[-1]["content"]
    assert "follow-up about our conversation" in user_turn.lower()
    assert "Context excerpts" not in user_turn


def test_translation_followup_skips_retrieval():
    # Reproduces the French-translation failure: must NOT retrieve.
    ctrl, retr, llm = make_controller(_decision("follow_up", False))
    history = [
        {"role": "user", "content": "I feel exhausted."},
        {"role": "assistant", "content": "A full grounded answer about exhaustion."},
    ]
    res = ctrl.respond("now give it to me in français", history)
    assert res.classification.label == "follow_up"
    assert res.used_retrieval is False
    assert retr.calls == 0


def test_appreciation_skips_retrieval_and_answer_llm():
    ctrl, retr, llm = make_controller(_decision("appreciation", False))
    res = ctrl.respond("that was great", [{"role": "user", "content": "x"},
                                          {"role": "assistant", "content": "y"}])
    assert res.classification.label == "appreciation"
    assert res.used_retrieval is False
    assert res.retrieved == []
    assert retr.calls == 0
    assert llm.answer_calls == 0                 # no answer-LLM call, deterministic reply
    assert "glad it helped" in res.answer.lower()


def test_vague_clarifies():
    ctrl, retr, llm = make_controller(_decision("vague", False))
    res = ctrl.respond("Help")
    assert res.classification.label == "vague"
    assert res.used_retrieval is False
    assert retr.calls == 0 and llm.answer_calls == 0
    assert "tell me a bit more" in res.answer.lower()


def test_off_topic_redirects():
    ctrl, retr, llm = make_controller(_decision("off_topic", False))
    res = ctrl.respond("What is the capital of France?")
    assert res.classification.label == "off_topic"
    assert retr.calls == 0 and llm.answer_calls == 0
    assert "outside what i focus on" in res.answer.lower()


def test_off_topic_continuation_stays_off_topic():
    # Reproduces the "please tell me" failure: a pushy follow-on to an off-topic
    # question must NOT be turned into a retrieval. (Router returns off_topic.)
    ctrl, retr, llm = make_controller(_decision("off_topic", False))
    history = [
        {"role": "user", "content": "who is trump's wife"},
        {"role": "assistant", "content": "That's a bit outside what I focus on..."},
    ]
    res = ctrl.respond("please tell me", history)
    assert res.classification.label == "off_topic"
    assert res.used_retrieval is False
    assert retr.calls == 0


# --- fallback to the rule-based engine -------------------------------------

def test_fallback_when_router_unavailable_still_routes():
    # Router LLM raises -> deterministic rule engine handles the turn.
    ctrl, retr, llm = make_controller(raise_on_router=True)
    res = ctrl.respond("How can I become a better leader?")
    assert res.used_retrieval is True            # strategic -> retrieves
    assert retr.calls == 1
    assert res.classification.reason.lower().startswith("[fallback")


def test_fallback_on_invalid_json_off_topic():
    ctrl, retr, llm = make_controller(router="not json at all")
    res = ctrl.respond("What is the capital of France?")
    assert res.classification.label == "off_topic"
    assert retr.calls == 0 and llm.answer_calls == 0
    assert "[fallback" in res.classification.reason.lower()
