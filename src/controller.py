"""MentorController: the single service layer that orchestrates a turn.

Flow per query:
    route (LLM, context-aware) -> (retrieve | answer-from-history | clarify
                                   | redirect | acknowledge)

The decision of *how to handle the turn* is made by an LLM turn router
(``src/decision/turn_router.py``) that reads the conversation and the latest
message and returns a label + whether to search the knowledge base + a clean
standalone search query. This is what makes routing context-aware (e.g.
"translate that to French" is answered from history, not re-retrieved; a pushy
follow-on to an off-topic question is still refused). The router degrades to a
deterministic rule-based engine if the LLM is unavailable.

Conversation memory is passed in by the caller (Streamlit session_state / CLI)
and trimmed here — there is no separate memory agent, by design.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from CONFIG import CONFIG
from src.decision.query_classifier import Classification
from src.decision.turn_router import route_turn
from src.generation.llm_client import LLMClient
from src.generation.mentor_response import (
    appreciation_response,
    build_followup_messages,
    build_messages,
    clarifying_response,
    off_topic_response,
)
from src.retrieval.retriever import RetrievedChunk, Retriever


@dataclass
class MentorResult:
    answer: str
    classification: Classification
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    used_retrieval: bool = False


class MentorController:
    def __init__(self, retriever: Retriever, llm: LLMClient | None = None):
        self.retriever = retriever
        self.llm = llm or LLMClient()

    def respond(self, query: str, history: list[dict] | None = None) -> MentorResult:
        history = history or []
        trimmed = history[-CONFIG.app.conversation_memory_size :]

        # One LLM call decides how to handle the turn (rules fall back if it fails).
        decision = route_turn(self.llm, query, trimmed)
        classification = decision.classification()

        # emotional / strategic / general -> grounded RAG answer.
        if decision.needs_retrieval:
            chunks = self.retriever.retrieve(decision.search_query)
            messages = build_messages(query, classification, trimmed, chunks)
            answer = self.llm.chat(messages)
            return MentorResult(answer, classification, chunks, used_retrieval=True)

        # Conversational / meta follow-up ("summarize that", "say it in French")
        # — answered from history with no new retrieval.
        if decision.label == "follow_up":
            answer = self.llm.chat(build_followup_messages(query, trimmed))
            return MentorResult(answer, classification, used_retrieval=False)

        # The remaining labels are handled by deterministic templates — faster and
        # safer than letting the model improvise off-topic or over-answer a "thanks".
        if decision.label == "off_topic":
            return MentorResult(off_topic_response(), classification)
        if decision.label == "appreciation":
            return MentorResult(appreciation_response(), classification)
        # vague (and any unexpected non-retrieval label) -> ask for a little more.
        return MentorResult(clarifying_response(), classification)
