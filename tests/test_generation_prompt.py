"""The prompt builder includes everything the LLM needs to stay grounded.

No real LLM call here — we only assert on the assembled messages.
"""

import pytest

from CONFIG import CONFIG
from src.decision.query_classifier import Classification
from src.generation.mentor_response import (
    SYSTEM_PROMPT,
    appreciation_response,
    build_messages,
    clarifying_response,
    off_topic_response,
)
from src.retrieval.retriever import RetrievedChunk
from src.utils.prompts import load_prompt


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="ted_848_chunk_009", doc_id="ted_848",
        title="How great leaders inspire action", speaker="Simon Sinek",
        topic="leadership", source_type="TED", source_url="http://x",
        text="Great leaders start with why.", similarity=0.51, distance=0.49,
    )


def test_system_prompt_has_grounding_rules():
    assert "context" in SYSTEM_PROMPT.lower()
    assert "fabricate" in SYSTEM_PROMPT.lower() or "invent" in SYSTEM_PROMPT.lower()


def test_system_prompt_is_loaded_from_prompts_dir():
    # The persona lives in prompts/system_prompt.md and is loaded at import.
    assert (CONFIG.paths.prompts_dir / "system_prompt.md").exists()
    assert SYSTEM_PROMPT == load_prompt("system_prompt")
    assert SYSTEM_PROMPT.strip()


def test_load_prompt_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_prompt("does_not_exist")


def test_messages_contain_all_required_parts():
    cls = Classification("strategic", "Asks for a plan.")
    history = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]
    messages = build_messages("How do I lead better?", cls, history, [_chunk()])

    assert messages[0]["role"] == "system"
    # History is preserved between system and the final user turn.
    assert {"role": "user", "content": "earlier question"} in messages

    user_turn = messages[-1]["content"]
    assert "How do I lead better?" in user_turn          # the query
    assert "strategic" in user_turn                        # query type
    assert "Asks for a plan." in user_turn                 # classification reason
    assert "Simon Sinek" in user_turn                      # source metadata
    assert "ted_848_chunk_009" in user_turn                # chunk id
    assert "Great leaders start with why." in user_turn    # chunk text


def test_messages_handle_no_context():
    cls = Classification("general", "On-topic.")
    messages = build_messages("hello?", cls, [], [])
    assert "no relevant context" in messages[-1]["content"].lower()


def test_templated_responses_mention_supported_topics():
    assert "career" in clarifying_response().lower()
    assert "leader" in off_topic_response().lower()


def test_appreciation_response_is_short_and_social():
    response = appreciation_response()
    assert "glad" in response.lower()
    assert "Reflection" not in response
