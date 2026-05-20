"""Shared LLM interface so the agent can swap backends (local / Groq / ...)."""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    name: str

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 256) -> str:
        """Return the model's raw text output for a chat (system, user) pair."""
        raise NotImplementedError
