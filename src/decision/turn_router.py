"""LLM turn router — the primary, context-aware decision layer.

Given the conversation so far and the user's latest message, one LLM call decides
*how to handle the turn* and returns a single JSON object:

    {label, needs_retrieval, search_query, reason}

``label`` is one of the same seven labels the rest of the system understands
(emotional / strategic / general / follow_up / vague / off_topic / appreciation),
so the controller, UI, and tests need no new vocabulary. The router is what makes
routing context-aware — e.g. "translate that to French" after an answer is a
``follow_up`` (no retrieval), and "please tell me" after an off-topic question
stays ``off_topic`` (the guardrail beats the continuation), instead of being
blindly re-embedded and searched.

Robustness: the prompt lives in ``prompts/router_prompt.md`` (editable). On any
LLM failure or unparseable / invalid reply we fall back to the deterministic
rule-based engine (``query_classifier``). Rules are the safety net, not the
primary path — when the fallback fires it is flagged in ``reason`` so regressions
are visible in the UI.

One decision + at most one retrieval per turn (no multi-hop loop, by design);
across a session retrieval can still fire on as many turns as genuinely need it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from src.decision.query_classifier import (
    GREETING_HELP,
    Classification,
    classify_query,
    is_followup,
)
from src.utils.prompts import load_prompt

# Labels whose handling requires a fresh knowledge-base search.
RETRIEVAL_LABELS = {"emotional", "strategic", "general"}
# All labels the router (and the rest of the system) understands.
VALID_LABELS = RETRIEVAL_LABELS | {"follow_up", "vague", "off_topic", "appreciation"}

ROUTER_SYSTEM = load_prompt("router_prompt")


@dataclass
class TurnDecision:
    label: str
    needs_retrieval: bool
    search_query: str
    reason: str

    def classification(self) -> Classification:
        """Adapt to the Classification the controller/UI already speak."""
        return Classification(self.label, self.reason)


def _format_conversation(history: list[dict], max_chars: int = 600) -> str:
    if not history:
        return "(none)"
    lines = []
    for turn in history:
        who = "User" if turn["role"] == "user" else "Mentor"
        text = " ".join(turn["content"].split())
        if len(text) > max_chars:
            text = text[:max_chars] + " ..."
        lines.append(f"{who}: {text}")
    return "\n".join(lines)


def _coerce(data: dict, query: str) -> TurnDecision | None:
    """Validate/normalize a parsed router reply; None if it is unusable."""
    label = str(data.get("label", "")).strip().lower()
    if label not in VALID_LABELS:
        return None
    # The label is the source of truth for whether we retrieve; ignore an
    # inconsistent needs_retrieval from the model.
    needs = label in RETRIEVAL_LABELS
    search_query = str(data.get("search_query") or "").strip()
    if needs and not search_query:
        search_query = query
    if not needs:
        search_query = ""
    reason = str(data.get("reason", "")).strip() or f"Routed as {label}."
    return TurnDecision(label, needs, search_query, reason)


def parse_decision(text: str, query: str, history: list[dict]) -> TurnDecision:
    """Parse the router's JSON; fall back to the rule-based engine on any problem."""
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        decision = _coerce(json.loads(text[start:end]), query)
    except (ValueError, json.JSONDecodeError, TypeError):
        decision = None
    if decision is None:
        return rule_based_decision(query, history, note="router reply unparseable")
    return decision


def route_turn(llm, query: str, history: list[dict]) -> TurnDecision:
    """Ask the LLM how to handle this turn; degrade to rules if it can't answer."""
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM},
        {
            "role": "user",
            "content": (
                f"<conversation>\n{_format_conversation(history)}\n</conversation>\n\n"
                f"<latest_message>\n{query}\n</latest_message>\n\n"
                "Return ONLY the JSON object."
            ),
        },
    ]
    try:
        reply = llm.chat(messages, format="json")
    except Exception:  # noqa: BLE001 - any LLM/transport failure -> safe fallback
        return rule_based_decision(query, history, note="router LLM unavailable")
    return parse_decision(reply, query, history)


def rule_based_decision(
    query: str, history: list[dict], note: str = ""
) -> TurnDecision:
    """Deterministic fallback mirroring the original keyword router.

    Used only when the LLM router is unavailable or returns an invalid reply.
    ``note`` is surfaced in the reason so a fallback is visible in the UI.
    """
    prefix = f"[fallback: {note}] " if note else "[fallback] "

    if history and is_followup(query):
        return TurnDecision(
            "follow_up", False, "",
            prefix + "Refers to our conversation — answering from history.",
        )

    cls = classify_query(query)
    label = cls.label

    if label == "vague":
        prev = next(
            (t["content"] for t in reversed(history) if t["role"] == "user"), ""
        )
        if prev:
            # Short on-topic follow-up resolved with earlier context. A bare
            # greeting/"help" prior turn carries no signal, so don't fold it in.
            sq = query if prev.lower().strip() in GREETING_HELP else f"{prev} {query}"
            return TurnDecision(
                "general", True, sq,
                prefix + "Short follow-up resolved using earlier context.",
            )
        return TurnDecision("vague", False, "", prefix + cls.reason)

    if label in RETRIEVAL_LABELS:
        return TurnDecision(label, True, query, prefix + cls.reason)

    # appreciation / off_topic
    return TurnDecision(label, False, "", prefix + cls.reason)
