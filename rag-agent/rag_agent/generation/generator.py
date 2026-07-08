"""Answer generation: direct vs codegen, consuming the component-4 context."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..fallback import ContextBundle
from .codegen import CodeResult, run_codegen
from .prompts import (
    CODEGEN_SYS,
    DIRECT_SYS,
    build_codegen_user,
    build_direct_user,
    cells_from_chunks,
)


@dataclass
class AnswerResult:
    answer: Optional[str]
    mode: str                       # "direct" | "codegen"
    raw: str = ""                   # raw LLM output
    code: str = ""                  # codegen path only
    exec_ok: bool = False           # codegen path only
    error: str = ""
    used_fallback: bool = False
    n_cells: int = 0

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "mode": self.mode,
            "exec_ok": self.exec_ok,
            "error": self.error,
            "used_fallback": self.used_fallback,
            "n_cells": self.n_cells,
        }


class Answerer:
    """Generates an answer from an assembled context with a chosen reasoning path."""

    def __init__(self, llm, max_tokens: int = 256, codegen_timeout: float = 5.0) -> None:
        self.llm = llm
        self.max_tokens = max_tokens
        self.codegen_timeout = codegen_timeout

    def answer_direct(self, query: str, bundle: ContextBundle) -> AnswerResult:
        raw = self.llm.complete(
            system=DIRECT_SYS,
            user=build_direct_user(query, bundle.text),
            max_tokens=self.max_tokens,
        )
        ans = (raw or "").strip().splitlines()[0].strip() if raw else ""
        return AnswerResult(
            answer=ans or None,
            mode="direct",
            raw=raw or "",
            used_fallback=bundle.used_fallback,
            n_cells=bundle.n_chunks,
        )

    def answer_codegen(self, query: str, bundle: ContextBundle) -> AnswerResult:
        cells = cells_from_chunks(bundle.chunks)
        raw = self.llm.complete(
            system=CODEGEN_SYS,
            user=build_codegen_user(query, cells),
            max_tokens=self.max_tokens,
        )
        res: CodeResult = run_codegen(raw or "", cells, timeout=self.codegen_timeout)
        return AnswerResult(
            answer=res.value,
            mode="codegen",
            raw=raw or "",
            code=(raw or "").strip(),
            exec_ok=res.ok,
            error=res.error,
            used_fallback=bundle.used_fallback,
            n_cells=bundle.n_chunks,
        )

    def answer(self, query: str, bundle: ContextBundle, mode: str = "direct") -> AnswerResult:
        if mode == "direct":
            return self.answer_direct(query, bundle)
        if mode == "codegen":
            return self.answer_codegen(query, bundle)
        raise ValueError(f"mode must be 'direct' or 'codegen', got {mode!r}")
