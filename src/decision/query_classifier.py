"""Rule-based decision engine and lightweight pre-RAG router.

Classifies a user query and returns the label plus a short, human-readable
reason. Rule-based on purpose: it is transparent, testable, and fast — exactly
what the assignment values over model complexity.

  emotional  -> lead with empathy
  strategic  -> structured, actionable advice
  vague      -> ask one clarifying question (unless history gives context)
  general    -> normal grounded mentor answer
  off_topic  -> politely redirect to supported topics
  appreciation -> lightweight social acknowledgement, no retrieval
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Words that signal an emotional state (empathy should come first).
EMOTIONAL_TERMS = {
    "feel", "feeling", "feelings", "felt", "stuck", "lost", "afraid", "fear",
    "scared", "anxious", "anxiety", "overwhelmed", "depressed", "sad", "hopeless",
    "lonely", "exhausted", "struggling", "struggle", "hurt", "ashamed", "shame",
    "worthless", "insecure", "nervous", "worried", "worry", "miserable",
    "frustrated", "frustration", "doubt", "impostor", "imposter", "burnout",
    "burned", "burnt", "cry", "crying", "panic", "hate",
}

# Phrases that signal an emotional state.
EMOTIONAL_PHRASES = {
    "give up", "giving up", "can't cope", "cant cope", "hate myself",
    "not good enough", "fall apart", "falling apart", "no idea what to do",
}

# Words/phrases that signal a request for a plan or strategy.
STRATEGIC_PHRASES = {
    "how to", "how can i", "how do i", "how should i", "what should i",
    "step by step", "step-by-step", "action plan", "30 day", "30-day", "7 day",
    "7-day", "become a better", "become better", "improve my", "build a",
    "develop a", "give me a plan", "plan to", "strategy for", "strategy to",
    "ways to", "tips for", "tips to", "framework", "roadmap", "routine",
}
STRATEGIC_TERMS = {
    "plan", "steps", "strategy", "improve", "build", "develop", "goal", "goals",
    "achieve", "habit", "habits", "routine", "framework", "roadmap", "boost",
    "increase", "optimize",
}

# Topics the mentor actually covers — used to keep on-topic queries on-topic.
TOPIC_TERMS = {
    "career", "job", "jobs", "work", "working", "leader", "leadership", "lead",
    "manager", "management", "managing", "team", "boss", "promotion", "confidence",
    "confident", "purpose", "meaning", "motivation", "motivated", "productivity",
    "productive", "focus", "procrastinate", "procrastination", "procrastinating",
    "habit", "growth", "mindset", "success", "succeed", "fail", "failure",
    "failing", "vulnerability", "vulnerable", "stress", "communication",
    "communicate", "listen", "decision", "decisions", "passion", "grit",
    "perseverance", "introvert", "introverts", "happiness", "fulfillment",
    "fulfilment", "purposeful", "growth-mindset", "burnout", "self-doubt",
}

# Signals that a query is general-knowledge / unrelated to mentoring.
OFF_TOPIC_TERMS = {
    "capital", "weather", "recipe", "cook", "cooking", "score", "football",
    "soccer", "movie", "lyrics", "translate", "translation", "python",
    "javascript", "java", "code", "equation", "calculate", "president",
    "population", "currency", "bitcoin", "stock", "planet", "distance",
    "temperature", "wifi", "install", "download", "phone", "laptop", "recipe",
}
_TRIVIA_OPENER = re.compile(r"^(what|who|when|where|which)\s+(is|are|was|were|did)\b")

GREETING_HELP = {
    "help", "help me", "i need help", "advice", "idk", "i don't know",
    "i dont know", "not sure", "hi", "hello", "hey", "yo", "?", "...",
}

# Short social turns should close the conversational loop, not trigger retrieval.
APPRECIATION_PHRASES = {
    "thanks", "thank you", "thank u", "thx", "appreciate it",
    "thanks a lot", "thank you so much", "many thanks", "ok thanks",
    "okay thanks", "yes thanks",
    "that was great", "this was great", "that is great", "this is great",
    "that was helpful", "this was helpful", "that helps", "this helps",
    "it helped", "that helped", "very helpful", "super helpful",
    "that was useful", "this was useful", "it was useful",
    "good answer", "great answer", "nice answer", "clear answer",
    "makes sense", "that makes sense", "it makes sense",
    "got it", "i got it", "understood", "perfect", "awesome", "nice",
    "cool", "great", "good", "excellent", "exactly",
    "liked it", "i liked it", "i liked that", "i liked this",
    "liked that", "liked this", "i liked the answer",
    "i liked your answer", "i liked your response", "i liked your advice",
    "love it", "loved it", "i love it", "i loved it", "i loved that",
}
APPRECIATION_TERMS = {
    "thanks", "thank", "appreciate", "liked", "love", "loved", "helpful",
    "useful", "great", "good", "nice", "excellent", "awesome", "perfect",
    "cool", "clear", "helped", "agree", "understood",
}
APPRECIATION_REFERENTS = {
    "it", "that", "this", "answer", "response", "reply", "advice", "guidance",
    "explanation",
}
REQUEST_TERMS = {
    "how", "what", "why", "when", "where", "who", "which", "tell", "explain",
    "give", "plan", "steps", "strategy", "advice", "help", "about", "more",
    "another", "next",
}
REQUEST_PHRASES = {
    "can you", "could you", "would you", "should i", "what should",
    "how can", "how do", "tell me", "give me", "help me", "i need",
    "i would like", "i'd like", "like to",
}
CONTRAST_OR_CONTINUE = {"but", "however", "though", "although", "now", "also"}

# Signals that a query operates on the *ongoing conversation* (summarize/rephrase
# the previous answer, refer back to it) rather than asking something new. These
# only count when there is conversation history (handled by the controller), and
# such turns are answered from history WITHOUT new retrieval.

# Explicit references to the previous answer — always a follow-up.
FOLLOWUP_BACKREFS = {
    "you said", "you mentioned", "you told", "you just", "you wrote",
    "your answer", "your response", "your reply", "your context",
    "your previous", "your last", "previous answer", "last answer",
    "the above", "above answer", "what you said", "what you wrote",
}
# Meta phrases that act on the prior answer.
FOLLOWUP_PHRASES = {
    "make it short", "too long", "in brief", "be brief", "more concise",
    "in simple terms", "say it again", "repeat that", "repeat it",
    "explain that", "explain it", "expand on",
}
# Meta words that act on prior content — but only when the query is short, so a
# real content question ("summarize the key traits of great leaders") is not
# misread as meta.
FOLLOWUP_WORDS = {
    "summarize", "summarise", "summary", "tldr", "shorter", "briefly",
    "concise", "rephrase", "reword", "rewrite", "simpler", "elaborate",
    "expand", "repeat",
}
# "in 2 sentences", "in three words", "in one line", ...
_FOLLOWUP_LENGTH = re.compile(
    r"\bin (one|two|three|four|five|\d+)\b.{0,15}(sentence|word|line|bullet|point)"
)


def is_followup(query: str) -> bool:
    """True if the query refers to / acts on the prior conversation.

    The caller should only treat this as a follow-up when history exists.
    """
    text = query.lower().strip()
    words = _words(text)
    if any(p in text for p in FOLLOWUP_BACKREFS):
        return True
    if any(p in text for p in FOLLOWUP_PHRASES):
        return True
    if _FOLLOWUP_LENGTH.search(text):
        return True
    # Bare meta commands like "shorter" / "can you summarize that" — short only.
    if len(words) <= 6 and (FOLLOWUP_WORDS & set(words)):
        return True
    return False


@dataclass
class Classification:
    label: str
    reason: str

    def to_dict(self) -> dict:
        return {"label": self.label, "reason": self.reason}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z'\-]*", text.lower())


def _matched(terms: set[str], word_set: set[str]) -> list[str]:
    return sorted(terms & word_set)


def _phrases_in(phrases: set[str], text: str) -> list[str]:
    return sorted(p for p in phrases if p in text)


def is_appreciation(query: str) -> bool:
    """Backward-compatible alias for the broader social acknowledgement check."""
    return is_social_acknowledgement(query)


def is_social_acknowledgement(query: str) -> bool:
    """Broader appreciation detector for short, non-request acknowledgement."""
    text = query.lower().strip()
    if "?" in text:
        return False
    text = re.sub(r"[.!?,;:]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    words = _words(text)
    word_set = set(words)

    if len(words) > 10:
        return False
    if text in APPRECIATION_PHRASES:
        return True
    if any(p in text for p in REQUEST_PHRASES):
        return False
    if word_set & CONTRAST_OR_CONTINUE:
        return False
    if {"liked", "loved"} & word_set and "your" in word_set:
        return True
    if word_set & REQUEST_TERMS:
        return False
    return bool((word_set & APPRECIATION_TERMS) and (word_set & APPRECIATION_REFERENTS))


def classify_query(query: str) -> Classification:
    text = query.lower().strip()
    words = _words(text)
    word_set = set(words)

    emo = _matched(EMOTIONAL_TERMS, word_set) + _phrases_in(EMOTIONAL_PHRASES, text)
    strat = _matched(STRATEGIC_TERMS, word_set) + _phrases_in(STRATEGIC_PHRASES, text)
    topic = _matched(TOPIC_TERMS, word_set)
    off = _matched(OFF_TOPIC_TERMS, word_set)

    has_emo, has_strat, has_topic = bool(emo), bool(strat), bool(topic)

    # 0) Conversational acknowledgement: "thanks", "that was great", etc.
    if is_social_acknowledgement(text) and not has_emo:
        return Classification(
            "appreciation",
            "Short acknowledgement or thanks — no new retrieval needed.",
        )

    # 1) Off-topic: an unrelated factual question with no mentoring signal.
    if not has_emo and not has_strat and not has_topic:
        if off:
            return Classification(
                "off_topic",
                f"Mentions unrelated topics ({', '.join(off)}) and no career/"
                "leadership/psychology terms.",
            )
        if _TRIVIA_OPENER.match(text):
            return Classification(
                "off_topic",
                "Looks like a general-knowledge question unrelated to mentoring.",
            )

    # 2) Vague: greetings/bare help, or very short with no clear signal.
    if text in GREETING_HELP or (
        len(words) <= 3 and not has_emo and not has_strat
    ):
        return Classification(
            "vague",
            "Too short or generic to act on — needs a clarifying question.",
        )

    # 3) Emotional beats strategic: lead with empathy when feelings are present.
    if has_emo:
        return Classification(
            "emotional",
            f"Contains emotional terms ({', '.join(emo)}).",
        )

    # 4) Strategic: asking for a plan / how-to / improvement.
    if has_strat:
        return Classification(
            "strategic",
            f"Asks for a plan or strategy ({', '.join(strat)}).",
        )

    # 5) General on-topic question.
    return Classification(
        "general",
        f"On-topic mentoring question ({', '.join(topic) or 'general'}).",
    )
