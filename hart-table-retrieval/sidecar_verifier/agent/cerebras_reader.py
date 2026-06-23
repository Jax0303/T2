"""Cerebras-hosted reader. Same surface as ``GroqAnswerer``.

Cerebras Cloud serves Llama-3.3-70B over an OpenAI-compatible endpoint at
https://api.cerebras.ai/v1. Free tier starts with $10 credit (~5k+ requests).
Requires ``CEREBRAS_API_KEY``.
"""
from __future__ import annotations

import os
import time
from typing import Optional

from openai import OpenAI

from .answerer import AnswerResult, _format_table_for_llm
from .groq_reader import CodeResult, _clean_answer, _df_schema_for_llm
from .subtable import SubtableResult, render_subtable_for_llm
from ..store.table_store import TableRecord


# Cerebras hosts a different model lineup than Groq. As of writing, the closest
# strong open-source reader is `gpt-oss-120b` (Llama-3.3-70B is not available).
_DEFAULT_MODEL = "gpt-oss-120b"


class CerebrasAnswerer:
    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: Optional[str] = None,
        # gpt-oss-120b is a reasoning model: 50–200 reasoning tokens may precede
        # the visible answer. Keep this generous.
        max_tokens: int = 1024,
        temperature: float = 0.0,
        retries: int = 4,
    ) -> None:
        key = api_key or os.environ.get("CEREBRAS_API_KEY")
        if not key:
            raise RuntimeError(
                "CEREBRAS_API_KEY not set. Get one free at https://cloud.cerebras.ai "
                "and `export CEREBRAS_API_KEY=...`"
            )
        self.client = OpenAI(base_url="https://api.cerebras.ai/v1", api_key=key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retries = retries

    def _chat(self, system: str, user: str) -> str:
        last_err = None
        for attempt in range(self.retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                # Honor Retry-After header when present (rate-limit / 429).
                ra = None
                resp = getattr(e, "response", None)
                if resp is not None:
                    try:
                        ra = float(resp.headers.get("retry-after", "") or "")
                    except (ValueError, TypeError):
                        ra = None
                sleep_s = ra if ra and ra > 0 else 2 ** attempt
                time.sleep(min(sleep_s, 60))
        raise RuntimeError(f"Cerebras chat failed after {self.retries} retries: {last_err}")

    def answer_full(self, query: str, rec: TableRecord) -> AnswerResult:
        table_block = _format_table_for_llm(rec, max_rows=80)
        system = (
            "You are a precise table QA assistant. Answer ONLY from the table below. "
            "If the answer is a number, output just the number (no units, no commas). "
            "If multiple numbers, separate with ', '. "
            "If not answerable from the table, output 'N/A'."
        )
        user = f"Table:\n{table_block}\n\nQuestion: {query}\n\nAnswer:"
        raw = self._chat(system, user)
        return AnswerResult(answer=_clean_answer(raw), raw_output=raw, table_id=rec.table_id)

    def answer_subtable(self, query: str, sub: SubtableResult,
                        table_id: str = "", title: str = "") -> AnswerResult:
        sub_block = render_subtable_for_llm(sub, title=title)
        system = (
            "You are a precise table QA assistant. The sub-table below has been "
            "pre-filtered to contain only the rows and columns relevant to the question. "
            "Answer ONLY from this sub-table. "
            "If the answer is a number, output just the number (no units, no commas). "
            "If multiple numbers, separate with ', '. "
            "If not answerable, output 'N/A'."
        )
        user = f"Sub-table:\n{sub_block}\n\nQuestion: {query}\n\nAnswer:"
        raw = self._chat(system, user)
        return AnswerResult(answer=_clean_answer(raw), raw_output=raw, table_id=table_id)

    def code_for_query(self, query: str, rec: TableRecord) -> CodeResult:
        schema = _df_schema_for_llm(rec)
        system = (
            "You write a single pandas EXPRESSION (not a statement) that computes "
            "the answer to a question about a DataFrame `df`. "
            "Do NOT include imports, assignments, print, or anything other than the expression. "
            "The expression must evaluate to the answer: a scalar number, string, or list. "
            "If the answer cannot be computed, return the expression: 'N/A'."
        )
        user = (
            f"{schema}\n\n"
            f"Question: {query}\n\n"
            "Pandas expression:"
        )
        raw = self._chat(system, user)
        code = raw.strip()
        if code.startswith("```"):
            code = "\n".join(ln for ln in code.splitlines() if not ln.startswith("```"))
        non_blank = [ln.strip() for ln in code.splitlines() if ln.strip()]
        if non_blank:
            code = non_blank[-1]
        return CodeResult(code=code, raw_output=raw, table_id=rec.table_id)

    def answer(self, query: str, rec: TableRecord) -> AnswerResult:
        return self.answer_full(query, rec)
