"""Groq-hosted reader for HiTab answers.

Three modes, one wrapper:
- ``answer_full``       (route α): full table markdown → free-form answer
- ``answer_subtable``   (route β): pre-extracted sub-table → answer
- ``code_for_query``    (route γ): generate a single pandas expression
                                   that produces the answer

Requires ``GROQ_API_KEY`` to be set (free tier: https://console.groq.com).
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Optional

from groq import Groq

from .answerer import AnswerResult, _format_table_for_llm
from .subtable import SubtableResult, render_subtable_for_llm
from ..store.table_store import TableRecord


_DEFAULT_MODEL = "llama-3.3-70b-versatile"


@dataclass
class CodeResult:
    code: str           # the raw pandas expression returned by the model
    raw_output: str
    table_id: str


def _clean_answer(text: str) -> str:
    text = text.strip()
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), text)
    first_line = re.sub(r"^(answer|the answer is|=|:)\s*", "", first_line, flags=re.IGNORECASE)
    return first_line


def _df_schema_for_llm(rec: TableRecord, n_sample: int = 3) -> str:
    df = rec.df
    cols = [" / ".join(rec.col_header_path(c)) or f"col_{c}" for c in range(df.shape[1])]
    schema_lines = [
        f"DataFrame `df` shape: ({df.shape[0]} rows, {df.shape[1]} cols)",
        "Column names (index → label):",
    ]
    for i, name in enumerate(cols):
        try:
            sample = df.iloc[:n_sample, i].tolist()
        except Exception:
            sample = []
        schema_lines.append(f"  df.columns[{i}] = {name!r}   sample: {sample}")
    schema_lines.append("\nRow index (first few left-header paths):")
    for r in range(min(n_sample, df.shape[0])):
        schema_lines.append(f"  df.index[{r}] = {' / '.join(rec.row_header_path(r)) or f'row_{r}'!r}")
    return "\n".join(schema_lines)


class GroqAnswerer:
    """Groq Llama-3.3-70B reader.

    Same surface area as ``LocalLLMAnswerer`` plus two extra modes.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        retries: int = 3,
    ) -> None:
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Get one free at https://console.groq.com "
                "and `export GROQ_API_KEY=...`"
            )
        self.client = Groq(api_key=key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retries = retries

    # ---- core chat helper with retry ----
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
                # exponential backoff for rate-limit / transient errors
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Groq chat failed after {self.retries} retries: {last_err}")

    # ---- route alpha: full table dump ----
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

    # ---- route beta: pre-extracted sub-table ----
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

    # ---- route gamma: code generation ----
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
        # Strip code fences / leading "df ..."-like prefixes
        code = raw.strip()
        if code.startswith("```"):
            code = "\n".join(ln for ln in code.splitlines() if not ln.startswith("```"))
        # Take the last non-blank line (some models prefix explanation)
        non_blank = [ln.strip() for ln in code.splitlines() if ln.strip()]
        if non_blank:
            code = non_blank[-1]
        return CodeResult(code=code, raw_output=raw, table_id=rec.table_id)

    # ---- back-compat ----
    def answer(self, query: str, rec: TableRecord) -> AnswerResult:
        return self.answer_full(query, rec)
