"""Turn-router parsing, coercion, and rule-based fallback (no LLM)."""

import json

import pytest

from src.decision.turn_router import (
    RETRIEVAL_LABELS,
    parse_decision,
    rule_based_decision,
)


def _json(label, needs=None, query="", reason="r"):
    d = {"label": label, "search_query": query, "reason": reason}
    if needs is not None:
        d["needs_retrieval"] = needs
    return json.dumps(d)


def test_parse_valid_retrieval_decision():
    d = parse_decision(_json("emotional", True, "feeling stuck in career"), "I feel stuck", [])
    assert d.label == "emotional"
    assert d.needs_retrieval is True
    assert d.search_query == "feeling stuck in career"


@pytest.mark.parametrize("label", ["follow_up", "vague", "off_topic", "appreciation"])
def test_non_retrieval_labels_never_retrieve(label):
    # Even if the model wrongly says needs_retrieval=true, the label decides.
    d = parse_decision(_json(label, True, "should be cleared"), "q", [])
    assert d.label == label
    assert d.needs_retrieval is False
    assert d.search_query == ""


def test_retrieval_label_with_empty_query_falls_back_to_raw_query():
    d = parse_decision(_json("general", True, ""), "what makes a good leader", [])
    assert d.needs_retrieval is True
    assert d.search_query == "what makes a good leader"


def test_json_embedded_in_prose_is_extracted():
    text = 'Sure! {"label":"strategic","needs_retrieval":true,"search_query":"focus plan","reason":"x"} done'
    d = parse_decision(text, "give me a plan", [])
    assert d.label == "strategic" and d.search_query == "focus plan"


def test_invalid_json_falls_back_to_rules():
    d = parse_decision("totally not json", "What is the capital of France?", [])
    assert d.label == "off_topic"            # rule engine classifies it
    assert d.reason.lower().startswith("[fallback")


def test_unknown_label_falls_back_to_rules():
    d = parse_decision(_json("banana", True, "x"), "I feel lost", [])
    assert d.label == "emotional"            # rules recover a sensible label
    assert "[fallback" in d.reason.lower()


# --- rule-based fallback itself --------------------------------------------

def test_rule_fallback_followup_with_history():
    history = [
        {"role": "user", "content": "How do I focus?"},
        {"role": "assistant", "content": "A long answer."},
    ]
    d = rule_based_decision("summarize that in 2 sentences", history)
    assert d.label == "follow_up" and d.needs_retrieval is False


def test_rule_fallback_vague_no_history_clarifies():
    d = rule_based_decision("Help", [])
    assert d.label == "vague" and d.needs_retrieval is False


def test_rule_fallback_vague_with_context_resolves_to_general():
    history = [
        {"role": "user", "content": "Help"},
        {"role": "assistant", "content": "What is this about?"},
    ]
    d = rule_based_decision("mostly career", history)
    assert d.label == "general" and d.needs_retrieval is True


def test_rule_fallback_labels_match_retrieval_set():
    assert rule_based_decision("Give me a plan to improve focus", []).label in RETRIEVAL_LABELS
