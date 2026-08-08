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

_DEFAULT_BASE = "https://api.openai.com/v1"

# Providers that speak the OpenAI chat-completions contract, so the same client
# reaches all of them by changing only the base URL and the key. Set
# OPENAI_COMPAT_BASE_URL (+ the matching key) to use one.
#   Anthropic      https://api.anthropic.com/v1     (its OpenAI-compat layer;
#                  the native endpoint is /v1/messages and is NOT this contract)
#   Google Gemini  https://generativelanguage.googleapis.com/v1beta/openai
#   DeepSeek       https://api.deepseek.com/v1
#   Together       https://api.together.xyz/v1


class OpenAILLM(BaseLLM):
    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        api_key: str | None = None,
        temperature: float = 0.0,
        request_timeout: float = 120.0,
        retry_on_429: int = 8,
        base_url: str | None = None,
    ) -> None:
        base = (base_url or os.environ.get("OPENAI_COMPAT_BASE_URL")
                or _DEFAULT_BASE).rstrip("/")
        self._endpoint = f"{base}/chat/completions"
        self._is_openai = base == _DEFAULT_BASE
        # The provider is part of what produced a number, so it goes in the name
        # that result files record -- "openai:gpt-4o" and a Gemini run must not
        # be indistinguishable afterwards.
        host = "openai" if self._is_openai else base.split("//")[-1].split("/")[0]
        self.name = f"{host}:{model_name}"
        self.model_name = model_name
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.retry_on_429 = retry_on_429
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Create one at "
                "https://platform.openai.com/api-keys, or point "
                "OPENAI_COMPAT_BASE_URL at another OpenAI-compatible provider "
                "and put that provider's key in OPENAI_API_KEY."
            )
        self._key = key
        # Cleared the first time the provider rejects `temperature`; read it to
        # find out whether a run was actually deterministic.
        self._send_temperature = True
        self.last_finish_reason: str | None = None  # see BaseLLM

    def complete(self, system: str, user: str, max_tokens: int = 256) -> str:
        # Newer OpenAI models reject `max_tokens`; `max_completion_tokens` is the
        # accepted spelling across both the 4.x and reasoning families. The
        # compatible providers went the other way and kept `max_tokens`, so the
        # spelling follows the host rather than the model.
        cap = "max_completion_tokens" if self._is_openai else "max_tokens"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            cap: max_tokens,
        }
        # Reasoning models accept only the default temperature; sending 0.0 is a
        # 400. Non-reasoning models keep the deterministic setting.
        if not self._is_reasoning() and self._send_temperature:
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
            req = urllib.request.Request(self._endpoint, data=data, headers=headers)
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
                # Some models (Claude 5 via the compat layer) reject `temperature`
                # outright rather than ignoring it. Drop it for the rest of this
                # object's life and retry once -- the alternative is maintaining a
                # per-provider list of which models still accept it. This LOSES the
                # temperature=0 determinism guarantee, so it is logged at warning
                # level and recorded on the object for result files to read.
                if (exc.code == 400 and "temperature" in detail
                        and self._send_temperature):
                    self._send_temperature = False
                    logger.warning("%s rejects temperature; retrying without it "
                                   "(sampling is now the provider default, NOT "
                                   "temperature=%s)", self.name, self.temperature)
                    payload.pop("temperature", None)
                    data = json.dumps(payload).encode()
                    continue
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
