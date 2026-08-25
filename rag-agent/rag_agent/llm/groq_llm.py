"""Groq cloud LLM backend (free tier).

Models tried in the bench:
  - llama-3.3-70b-versatile   (recommended; strongest free option)
  - llama-3.1-8b-instant       (fast fallback)

Requires ``GROQ_API_KEY`` in the environment.
"""
from __future__ import annotations

import logging
import os
import re
import time

from .base import BaseLLM

logger = logging.getLogger(__name__)

_RETRY_HINT = re.compile(r"try again in\s+([0-9hms.]+)", re.I)
# ms before m before s: "500ms" is half a second, not 500 minutes
_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_UNIT_S = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def retry_after(msg: str, attempt: int, cap: float = 900.0) -> float:
    """How long to wait after a 429.

    The free tier limits both tokens-per-minute and tokens-per-day, so a long
    run meets 429s as a matter of course rather than as a fault. Groq says how
    long to wait ("Please try again in 3.6825s"); honouring that beats a blind
    2**attempt, which either oversleeps or -- the bug this fixes -- exhausts its
    retries while the limit window is still open and kills a multi-hour run.
    Falls back to exponential when the hint is absent, and pads by a second
    because the window has to actually roll over.

    The cap is 15 minutes, not one, because the daily window asks for waits in
    minutes: capping at 60s meant eight retries covered eight minutes of a
    ten-minute wait and the run died anyway. A wait longer than the cap is a
    quota that will not clear inside this run, and should surface as the error
    it is rather than as an hours-long silent sleep.
    """
    hint = _RETRY_HINT.search(msg)
    if hint:
        # the daily limit answers in mixed units -- "10m29.424s" -- which a
        # single number-plus-unit read as "10" and rounded down to nothing
        parts = _DURATION.findall(hint.group(1))
        if parts:
            secs = sum(float(n) * _UNIT_S[u.lower()] for n, u in parts)
            return min(cap, secs + 1.0)
    return min(cap, 2.0 ** attempt)


class GroqLLM(BaseLLM):
    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        api_key: str | None = None,
        temperature: float = 0.0,
        request_timeout: float = 60.0,
        retry_on_429: int = 8,
        reasoning_effort: str | None = None,
    ) -> None:
        from groq import Groq

        # gpt-oss defaults to reasoning_effort="medium" and spends the completion
        # budget on hidden reasoning first: content comes back "" at
        # finish_reason="length" and the caller pays again for the wider retry.
        # On the free tier's 200k tokens/day that bought ~30 of 175 queries
        # before the daily quota ended the run. "low" still does the arithmetic.
        self.reasoning_effort = reasoning_effort or (
            "low" if "gpt-oss" in model_name else None)
        # part of the name, so guard_resume refuses to join records written at a
        # different effort -- it is a different reader
        self.name = f"groq:{model_name}" + (
            f"?reasoning_effort={self.reasoning_effort}"
            if self.reasoning_effort else "")
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
        self.last_finish_reason: str | None = None  # see BaseLLM

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
                    **({"reasoning_effort": self.reasoning_effort}
                       if self.reasoning_effort else {}),
                )
                self.last_finish_reason = resp.choices[0].finish_reason
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # broad — Groq SDK error hierarchy varies by version
                msg = str(e)
                last_err = e
                if "429" in msg or "rate" in msg.lower():
                    wait = retry_after(msg, attempt)
                    logger.warning("Groq rate-limited, retrying in %.1fs (attempt %d)",
                                   wait, attempt + 1)
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError(f"Groq retries exhausted: {last_err}")
