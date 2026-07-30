"""Shared LLM interface so the agent can swap backends (local / Groq / ...)."""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    name: str

    #: Set by each backend on every completion. Reasoning models (gpt-oss,
    #: qwen3, o-series, gpt-5) spend the completion budget on hidden reasoning
    #: first and then return content="" with finish_reason="length" — which is
    #: indistinguishable from a model that answered nothing unless the caller
    #: can see this. Part of the interface so callers can read it on any
    #: backend; backends that cannot report it leave it None.
    last_finish_reason: str | None = None

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 256) -> str:
        """Return the model's raw text output for a chat (system, user) pair."""
        raise NotImplementedError
