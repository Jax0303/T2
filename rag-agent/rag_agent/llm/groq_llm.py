"""Groq cloud LLM backend (free tier).

Models tried in the bench:
  - llama-3.3-70b-versatile   (recommended; strongest free option)
  - llama-3.1-8b-instant       (fast fallback)

Requires ``GROQ_API_KEY`` in the environment.
"""
from __future__ import annotations

import logging
import os
import time

from .base import BaseLLM

logger = logging.getLogger(__name__)


class GroqLLM(BaseLLM):
    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        api_key: str | None = None,
        temperature: float = 0.0,
        request_timeout: float = 60.0,
        retry_on_429: int = 3,
    ) -> None:
        from groq import Groq

        self.name = f"groq:{model_name}"
        self.model_name = model_name
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.retry_on_429 = retry_on_429
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys"
            )
        self.client = Groq(api_key=key, timeout=request_timeout)

    def complete(self, system: str, user: str, max_tokens: int = 256) -> str:
        last_err: Exception | None = None
        for attempt in range(1 + self.retry_on_429):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # broad — Groq SDK error hierarchy varies by version
                msg = str(e)
                last_err = e
                if "429" in msg or "rate" in msg.lower():
                    wait = 2 ** attempt
                    logger.warning("Groq rate-limited, retrying in %ds (attempt %d)", wait, attempt + 1)
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError(f"Groq retries exhausted: {last_err}")
