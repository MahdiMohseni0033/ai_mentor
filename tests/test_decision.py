"""The decision engine classifies the five query types correctly."""

import pytest

from src.decision.query_classifier import classify_query, is_followup


@pytest.mark.parametrize("query,expected", [
    ("I feel lost", "emotional"),
    ("I feel stuck in my career and don't know what to do next.", "emotional"),
    ("I'm afraid I'll fail.", "emotional"),
    ("How can I improve my leadership?", "strategic"),
    ("Give me a 30 day plan to improve productivity.", "strategic"),
    ("How do I find my purpose?", "strategic"),
    ("Help", "vague"),
    ("hi", "vague"),
    ("that was great", "appreciation"),
    ("thanks", "appreciation"),
    ("great answer!", "appreciation"),
    ("liked it", "appreciation"),
    ("I liked it", "appreciation"),
    ("I really liked it", "appreciation"),
    ("I loved your team advice", "appreciation"),
    ("that was useful", "appreciation"),
    ("ok thanks", "appreciation"),
    ("What is the capital of France?", "off_topic"),
    ("Tell me a recipe for pasta.", "off_topic"),
])
def test_labels(query, expected):
    assert classify_query(query).label == expected


def test_returns_label_and_reason():
    result = classify_query("I feel anxious about my job")
    assert result.label == "emotional"
    assert result.reason                      # non-empty explanation
    assert "label" in result.to_dict() and "reason" in result.to_dict()


def test_emotional_beats_strategic_when_both_present():
    # Has both a feeling and a plan-word; empathy should win.
    assert classify_query("I feel overwhelmed and need a plan").label == "emotional"


def test_general_on_topic_question():
    assert classify_query("Tell me about leadership and teamwork").label == "general"


@pytest.mark.parametrize("query,expected", [
    ("I would like to improve my leadership", "strategic"),
    ("How can I be liked as a manager?", "strategic"),
    ("liked it, now tell me about leadership", "general"),
])
def test_appreciation_does_not_swallow_real_requests(query, expected):
    assert classify_query(query).label == expected


@pytest.mark.parametrize("query", [
    "could you tell all of your context in 2 small sentence?",
    "can you summarize that?",
    "make it shorter",
    "explain it simpler",
    "what did you just say, in two sentences?",
    "rephrase your answer please",
    "shorter",
])
def test_is_followup_true(query):
    assert is_followup(query) is True


@pytest.mark.parametrize("query", [
    "How can I become a better leader?",
    "I feel stuck in my career",
    "Give me a plan to improve focus",
    "mostly career",
    "that was great",
    "liked it",
    "summarize the key traits of great leaders",   # 'summarize' but a fresh, longer ask
])
def test_is_followup_false(query):
    # Note: 'summarize ...' as a long fresh question should NOT be a follow-up
    # for content reasons here; the controller also requires history to fire.
    assert is_followup(query) is False
