"""LLM factory: ``build_llm('groq:llama-3.3-70b-versatile')`` etc."""
from __future__ import annotations

from .base import BaseLLM
from .groq_llm import GroqLLM
from .local_qwen import LocalQwenLLM
from .openai_llm import OpenAILLM


def build_llm(spec: str, **kwargs) -> BaseLLM:
    """``spec`` examples:

      "local:Qwen/Qwen2.5-7B-Instruct"
      "local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit"
      "groq:llama-3.3-70b-versatile"
      "groq:llama-3.1-8b-instant"
      "openai:gpt-4.1-mini"
    """
    backend, _, model = spec.partition(":")
    if backend == "local":
        return LocalQwenLLM(model_name=model or "Qwen/Qwen2.5-7B-Instruct", **kwargs)
    if backend == "groq":
        return GroqLLM(model_name=model or "llama-3.3-70b-versatile", **kwargs)
    if backend == "openai":
        return OpenAILLM(model_name=model or "gpt-4.1-mini", **kwargs)
    raise ValueError(
        f"Unknown LLM backend: {backend!r} (expected local:, groq: or openai:)")
