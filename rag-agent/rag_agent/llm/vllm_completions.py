"""vLLM OpenAI 호환 서버(/v1/completions)로 보내는 로컬 리더 — 2026-10-23 비교군 재실행용.

왜 completions 인가: 프롬프트를 여기서 토크나이저의 채팅 템플릿으로 직접 만들어(LocalQwenLLM._prompt 와 같은 메시지 구성,
enable_thinking=False) 원문 그대로 보내므로 HF 한 건씩 경로와 프롬프트가 글자 단위로 같다. 서버 쪽 템플릿 설정(생각 모드 기본값 등)에
좌우되지 않는다. greedy(temperature 0), `stop` 문자열은 서버가 생성 중 멈추고 돌려주는 글에는 포함되지 않는다(OpenAI 의미).

spec: "vllm:Qwen/Qwen3.5-4B" (factory). 서버 주소는 VLLM_BASE_URL(기본 http://localhost:8000/v1).
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional

from .base import BaseLLM


class VLLMCompletionsLLM(BaseLLM):
    def __init__(self, model_name: str = "Qwen/Qwen3.5-4B", base_url: Optional[str] = None,
                 request_timeout: float = 600.0, retries: int = 6, **_ignored) -> None:
        from transformers import AutoTokenizer
        self.model_name = model_name
        self.name = f"vllm:{model_name}"
        self.base = (base_url or os.environ.get("VLLM_BASE_URL") or "http://localhost:8000/v1").rstrip("/")
        self.request_timeout, self.retries = request_timeout, retries
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.n_calls = 0
        self.last_usage: dict = {}
        self.last_finish_reason: Optional[str] = None
        self.last_generation: dict = {}
        self._ctx = 0

    # ---- 프롬프트: LocalQwenLLM._prompt 와 같은 구성(system 메시지 항상 포함, 비생각) ----
    def _prompt(self, system: str, user: str, thinking: bool = False) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)

    def n_prompt_tokens(self, system: str, user: str) -> int:
        return len(self.tokenizer(self._prompt(system, user))["input_ids"])

    @property
    def context_limit(self) -> int:
        if not self._ctx:
            try:
                req = urllib.request.Request(f"{self.base}/models")
                with urllib.request.urlopen(req, timeout=30) as fh:
                    d = json.load(fh)
                for m in d.get("data", []):
                    if m.get("id") == self.model_name and m.get("max_model_len"):
                        self._ctx = int(m["max_model_len"])
            except Exception:
                self._ctx = 0
        return self._ctx

    def metadata(self) -> dict:
        return {"model_name": self.model_name, "endpoint": f"{self.base}/completions", "engine": "vllm-server",
                "context_limit": self.context_limit, "enable_thinking": False, "decoding": "greedy (temperature 0)"}

    # ---- 호출 ----
    def complete_prompt(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0, top_p: float = 0.95,
                        stop: Optional[list] = None) -> str:
        """렌더링된 프롬프트 원문을 그대로 보낸다(Yu 비교군처럼 도구 설명이 든 채팅 템플릿을 호출자가 만들 때)."""
        payload = {"model": self.model_name, "prompt": prompt, "max_tokens": int(max_tokens), "temperature": float(temperature or 0.0),
                   "seed": 42}
        if temperature and temperature > 0:
            payload["top_p"] = top_p
        if stop:
            payload["stop"] = list(stop)
        data = json.dumps(payload).encode()
        last = None
        for attempt in range(1 + self.retries):
            req = urllib.request.Request(f"{self.base}/completions", data=data, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.request_timeout) as fh:
                    body = json.load(fh)
                ch = body["choices"][0]
                self.n_calls += 1
                self.last_usage = body.get("usage") or {}
                self.last_finish_reason = ch.get("finish_reason")
                self.last_generation = {"new_tokens": (self.last_usage or {}).get("completion_tokens"), "finish_reason": self.last_finish_reason}
                return (ch.get("text") or "").strip()
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:300]
                last = RuntimeError(f"HTTP {exc.code}: {detail}")
                if exc.code == 400:
                    raise last from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
                last = exc
            if attempt < self.retries:
                time.sleep(min(2.0 * (2 ** attempt), 30.0))
        raise RuntimeError(f"vLLM call failed after retries: {last}")

    def complete(self, system: str, user: str, max_tokens: int = 256, temperature: float = 0.0, top_p: float = 0.95,
                 top_k: Optional[int] = None, min_p: Optional[float] = None, thinking: bool = False,
                 stop: Optional[list] = None) -> str:
        return self.complete_prompt(self._prompt(system, user, thinking), max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop)

    def complete_batch(self, system: str, users: list, max_tokens: int = 256) -> list:
        return [self.complete(system, u, max_tokens=max_tokens) for u in users]
