"""Prompt construction and persona for the mentor.

`build_messages` is a pure function (no LLM call) so it can be unit-tested:
it assembles the system persona, recent conversation history, the retrieved
context with source metadata, and per-category response guidance.

Vague and off-topic queries are answered with deterministic templates (no
retrieval, no LLM) — faster and safer for a demo than letting the model guess.
Short acknowledgement turns are handled the same way.
"""

from __future__ import annotations

from CONFIG import SUPPORTED_TOPICS
from src.decision.query_classifier import Classification
from src.retrieval.retriever import RetrievedChunk
from src.utils.prompts import load_prompt

# Mentor persona + grounding rules — edit in prompts/system_prompt.md.
SYSTEM_PROMPT = load_prompt("system_prompt")

# Per-category guidance appended to the user turn.
RESPONSE_GUIDANCE = {
    "emotional": (
        "This person is sharing a feeling. Begin with one or two sentences of "
        "genuine empathy. Then offer one grounded insight from the context, a "
        "little practical guidance, and one small next step."
    ),
    "strategic": (
        "This is a request for a plan. Give structured, actionable advice as a "
        "short numbered list grounded in the context, then end with one small "
        "next step they can take today."
    ),
    "general": (
        "Answer as a mentor: a clear, grounded insight, brief practical guidance, "
        "and one small next step."
    ),
}

RESPONSE_FORMAT = (
    "If I asked for a specific length, format, or style (e.g. \"in 2 sentences\", "
    "\"shorter\", \"as bullets\"), follow that exactly and skip the section headings.\n"
    "Otherwise, structure your reply with these short sections (use these exact "
    "headings):\n"
    "**Reflection** - one or two sentences.\n"
    "**Mentor insight** - the core idea, grounded in the context.\n"
    "**Practical steps** - 2-4 bullet points or numbered steps.\n"
    "**One small next step** - a single concrete action.\n"
    "Keep the whole reply under ~250 words. Do not add a sources section; the app "
    "shows sources separately."
)


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a labeled context block for the prompt."""
    if not chunks:
        return "(no relevant context retrieved)"
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(
            f"[{i}] Speaker: {c.speaker} | Talk: \"{c.title}\" | "
            f"Topic: {c.topic} | chunk_id: {c.chunk_id}\n{c.text}"
        )
    return "\n\n".join(blocks)


def build_messages(
    query: str,
    classification: Classification,
    history: list[dict],
    chunks: list[RetrievedChunk],
) -> list[dict]:
    """Assemble the full message list sent to the LLM."""
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Recent conversation history (already trimmed by the controller).
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})

    guidance = RESPONSE_GUIDANCE.get(classification.label, RESPONSE_GUIDANCE["general"])
    user_content = (
        f"Context excerpts from TED talks:\n{format_context(chunks)}\n\n"
        f"User query: {query}\n"
        f"Detected query type: {classification.label} "
        f"({classification.reason})\n\n"
        f"{guidance}\n\n{RESPONSE_FORMAT}"
    )
    messages.append({"role": "user", "content": user_content})
    return messages


def build_followup_messages(query: str, history: list[dict]) -> list[dict]:
    """Messages for a conversational follow-up — answered from history, no new
    retrieval (e.g. "summarize that in 2 sentences", "explain it simpler")."""
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({
        "role": "user",
        "content": (
            f"{query}\n\n"
            "(This is a follow-up about our conversation above. Answer from what "
            "was already said — do not introduce new sources or the usual section "
            "headings. Honor any length or format I asked for.)"
        ),
    })
    return messages


def clarifying_response() -> str:
    """Deterministic reply for a vague query with no useful history."""
    topics = ", ".join(SUPPORTED_TOPICS[:-1]) + f", or {SUPPORTED_TOPICS[-1]}"
    return (
        "I'd love to help. Could you tell me a bit more about what's on your mind? "
        f"For example, is this mainly about {topics}?"
    )


def off_topic_response() -> str:
    """Deterministic polite redirect for an off-topic query."""
    topics = ", ".join(SUPPORTED_TOPICS[:-1]) + f", and {SUPPORTED_TOPICS[-1]}"
    return (
        "That's a bit outside what I focus on. I'm a mentor for "
        f"{topics}. If any of those connect to what you're after, ask away — "
        "for instance, how to grow as a leader or find more focus at work."
    )


def appreciation_response() -> str:
    """Deterministic reply for thanks/praise; avoids unnecessary RAG output."""
    return "You're welcome. I'm glad it helped."
