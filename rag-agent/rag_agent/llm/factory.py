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


def _split_spec(model: str) -> tuple[str, dict]:
    """``"Qwen/Qwen2.5-7B-Instruct?quantization=8bit"`` -> name plus kwargs.

    The suffix has been in this docstring since the backends were written but
    was never parsed, so ``--reader local:...?quantization=8bit`` silently ran
    at the 4-bit default. That matters now: the arithmetic population is where
    the local reader fails, and whether the cause is the quantization or the
    model size is one run apart -- but only if the flag reaches the loader.
    """
    name, _, query = model.partition("?")
    kw = {}
    for part in query.split("&"):
        if not part:
            continue
        k, _, v = part.partition("=")
        v = v.strip()
        kw[k.strip()] = (None if v == "none" else
                         int(v) if v.lstrip("-").isdigit() else v)
    return name, kw


def build_llm(spec: str, **kwargs) -> BaseLLM:
    """``spec`` examples:

      "local:Qwen/Qwen2.5-7B-Instruct"
      "local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit"
      "groq:llama-3.3-70b-versatile"
      "groq:llama-3.1-8b-instant"
      "openai:gpt-4.1-mini"
    """
    _load_dotenv()
    backend, _, model = spec.partition(":")
    model, spec_kw = _split_spec(model)
    kwargs = {**spec_kw, **kwargs}          # an explicit kwarg beats the spec
    if backend == "local":
        return LocalQwenLLM(model_name=model or "Qwen/Qwen2.5-7B-Instruct",
                            **kwargs)
    if backend == "groq":
        return GroqLLM(model_name=model or "llama-3.3-70b-versatile", **kwargs)
    if backend == "openai":
        return OpenAILLM(model_name=model or "gpt-4.1-mini", **kwargs)
    raise ValueError(
        f"Unknown LLM backend: {backend!r} (expected local:, groq: or openai:)")
