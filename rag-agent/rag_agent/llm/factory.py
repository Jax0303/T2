"""LLM factory: ``build_llm('groq:llama-3.3-70b-versatile')`` etc."""
from __future__ import annotations

import os
from pathlib import Path

from .base import BaseLLM
from .groq_llm import GroqLLM
from .local_qwen import LocalQwenLLM
from .openai_llm import OpenAILLM


def _load_dotenv() -> None:
    """Minimal .env loader (no dependency): rag-agent/.env then repo-root/.env.

    Lives here rather than in one script because the backends read their keys
    out of the environment, and a script that forgets to load .env does not
    fail until it constructs the LLM -- which is *after* reconstructing and
    indexing every table. `setdefault`, so a real environment variable always
    wins over the file.
    """
    here = Path(__file__).resolve().parents[2]
    for env in (here / ".env", here.parent / ".env"):
        if env.is_file():
            for line in env.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def build_llm(spec: str, **kwargs) -> BaseLLM:
    """``spec`` examples:

      "local:Qwen/Qwen2.5-7B-Instruct"
      "local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit"
      "groq:llama-3.3-70b-versatile"
      "groq:llama-3.1-8b-instant"
      "openai:gpt-4.1-mini"
    """
    _load_dotenv()
    # Split off an optional "?k=v&k=v" query string (e.g.
    # "local:Qwen/...?quantization=8bit&dtype=float16"). These become keyword
    # arguments to the backend; "none" maps to Python None. Only applied to the
    # local backend, whose LocalQwenLLM accepts quantization/dtype — a hosted
    # backend would reject them. Without this the whole "Model?..." string was
    # passed as the HF repo id and raised "Repo id must use alphanumeric chars".
    base, _, query = spec.partition("?")
    opts = {}
    for kv in query.split("&"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            opts[k.strip()] = None if v.strip() == "none" else v.strip()
    backend, _, model = base.partition(":")
    if backend == "local":
        return LocalQwenLLM(model_name=model or "Qwen/Qwen2.5-7B-Instruct",
                            **{**opts, **kwargs})
    if backend == "groq":
        return GroqLLM(model_name=model or "llama-3.3-70b-versatile", **kwargs)
    if backend == "openai":
        return OpenAILLM(model_name=model or "gpt-4.1-mini", **kwargs)
    raise ValueError(
        f"Unknown LLM backend: {backend!r} (expected local:, groq: or openai:)")
