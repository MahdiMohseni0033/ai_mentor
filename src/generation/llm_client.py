"""Minimal client for the Ollama chat API.

Kept stateless — the conversation memory lives in the controller / Streamlit
session, not here.
qwen3.6 is a "thinking" model; we default `think=False` for fast, clean demo
replies and strip any stray <think> blocks defensively.
"""

from __future__ import annotations

import re

import requests

from CONFIG import CONFIG

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class LLMUnavailableError(RuntimeError):
    """Raised when the local LLM endpoint cannot be reached."""


class LLMClient:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        think: bool | None = None,
        num_predict: int | None = None,
        temperature: float | None = None,
        timeout_s: int | None = None,
    ):
        self.model = model or CONFIG.model.llm_model
        self.base_url = (base_url or CONFIG.model.llm_base_url).rstrip("/")
        self.think = CONFIG.model.llm_think if think is None else think
        self.num_predict = num_predict or CONFIG.model.llm_num_predict
        self.temperature = CONFIG.model.llm_temperature if temperature is None else temperature
        self.timeout_s = timeout_s or CONFIG.model.llm_timeout_s

    def chat(self, messages: list[dict], format: str | None = None) -> str:
        """Send a list of {role, content} messages, return the reply text.

        Pass format="json" to ask Ollama to constrain the output to valid JSON.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": self.think,
            "options": {
                "num_predict": self.num_predict,
                "temperature": self.temperature,
            },
        }
        if format:
            payload["format"] = format
        try:
            response = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise LLMUnavailableError(
                f"Could not reach the local LLM at {self.base_url}. "
                f"Is Ollama running with '{self.model}'?  ({exc})"
            ) from exc

        content = response.json().get("message", {}).get("content", "")
        return _THINK_BLOCK.sub("", content).strip()

    def health_check(self) -> tuple[bool, str]:
        """Return (ok, detail). Tries a tiny generation."""
        try:
            reply = LLMClient(
                model=self.model, base_url=self.base_url, think=False, num_predict=16
            ).chat([{"role": "user", "content": "Reply with the single word: OK"}])
            return (bool(reply), f"reply={reply!r}")
        except LLMUnavailableError as exc:
            return (False, str(exc))
