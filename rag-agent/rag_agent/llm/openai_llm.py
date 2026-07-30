"""OpenAI cloud LLM backend (paid).

Used as a SECOND solver family alongside Groq, so a serialization result
(S1 vs S2) can be shown to hold independently of the solver LLM. Never mix two
backends inside one records file — the arms stop being comparable.

Models exercised in the bench:
  - gpt-4.1-mini   (recommended; non-reasoning, works at codegen_max_tokens=160)
  - gpt-4o-mini    (cheaper, weaker at codegen)
  - gpt-5-mini     (reasoning: spends the completion budget on hidden reasoning
                    first, so it needs ~1024 like Groq's gpt-oss models)

Talks to the REST endpoint over stdlib ``urllib`` rather than the ``openai``
SDK: this environment's Python is PEP-668 externally-managed, and the chat
completions contract used here is small and stable.

Requires ``OPENAI_API_KEY`` in the environment.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request

from .base import BaseLLM

logger = logging.getLogger(__name__)

_ENDPOINT = "https://api.openai.com/v1/chat/completions"


class OpenAILLM(BaseLLM):
    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        api_key: str | None = None,
        temperature: float = 0.0,
        request_timeout: float = 120.0,
        retry_on_429: int = 8,
    ) -> None:
        self.name = f"openai:{model_name}"
        self.model_name = model_name
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.retry_on_429 = retry_on_429
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Create one at "
                "https://platform.openai.com/api-keys"
            )
        self._key = key
        self.last_finish_reason: str | None = None  # see BaseLLM

    def complete(self, system: str, user: str, max_tokens: int = 256) -> str:
        # Newer OpenAI models reject `max_tokens`; `max_completion_tokens` is the
        # accepted spelling across both the 4.x and reasoning families.
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": max_tokens,
        }
        # Reasoning models accept only the default temperature; sending 0.0 is a
        # 400. Non-reasoning models keep the deterministic setting.
        if not self._is_reasoning():
            payload["temperature"] = self.temperature
        # Loop-invariant: the prompt carries the serialized table and runs to
        # tens of KB, so encode it once instead of per retry.
        data = json.dumps(payload).encode()
        headers = {"Authorization": f"Bearer {self._key}",
                   "Content-Type": "application/json"}

        last_err: Exception | None = None
        for attempt in range(1 + self.retry_on_429):
            # urlopen mutates the Request it is handed (unredirected headers,
            # full_url on redirect), so build a fresh one around the shared body.
            req = urllib.request.Request(_ENDPOINT, data=data, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.request_timeout) as fh:
                    body = json.load(fh)
                choice = body["choices"][0]
                self.last_finish_reason = choice.get("finish_reason")
                # .strip() to match GroqLLM: answerer.py gates its empty-completion
                # retry on `not raw`, so a whitespace-only body must read as empty
                # here too or the two backends stop being interchangeable.
                return (choice["message"].get("content") or "").strip()
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:300]
                last_err = RuntimeError(f"HTTP {exc.code}: {detail}")
                # 429 = rate limit, 5xx = transient. A daily/credit exhaustion
                # also arrives as 429; the caller decides whether to stop.
                if not (exc.code == 429 or exc.code >= 500):
                    raise last_err from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError,
                    json.JSONDecodeError) as exc:  # timeout / connection reset
                last_err = exc
            # Deliberately narrow: a malformed response body (KeyError on
            # "choices") never succeeds on retry, so it propagates rather than
            # burning the whole backoff ladder before surfacing.
            if attempt < self.retry_on_429:  # no point sleeping after the last try
                wait = min(2.0 * (2 ** attempt), 60.0)
                logger.warning("openai call failed (%s) -> retry in %.0fs", last_err, wait)
                time.sleep(wait)
        raise RuntimeError(f"OpenAI call failed after retries: {last_err}")

    def _is_reasoning(self) -> bool:
        m = self.model_name
        return m.startswith(("o1", "o3", "o4", "gpt-5"))
