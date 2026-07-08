"""Synthetic-question generation for condition C3 (QGpT-style).

Two providers:

  - ``TemplateSynth``  deterministic, LLM-free. Fills lookup templates from
    sampled (row, column) pairs. Seeded by a hash of the table_id so output
    is stable regardless of corpus order. Good for offline smoke runs and
    as a lower bound on what LLM-generated questions buy.
  - ``LLMSynth``       wraps rag_agent.llm.factory.build_llm; generates N
    questions from a partial table. Results are cached to a jsonl file so
    the (slow, rate-limited) generation runs once per corpus.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

from rag_agent.prep.conditions import PrepTable, _cell_str, _markdown


class TemplateSynth:
    """Deterministic template questions from sampled cells."""

    def __init__(self, n_questions: int = 5):
        self.n_questions = n_questions

    def __call__(self, table: PrepTable) -> List[str]:
        seed = int(hashlib.sha1(table.table_id.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        out: List[str] = []

        if table.hierarchical and table.row_paths:
            # hierarchical: ask by header path
            cells = []
            for r in range(min(len(table.rows), len(table.row_paths))):
                for c in range(min(len(table.columns), len(table.rows[r]))):
                    if _cell_str(table.rows[r][c]):
                        cells.append((r, c))
            rng.shuffle(cells)
            for r, c in cells:
                row_p = " ".join(_cell_str(s) for s in table.row_paths[r])
                col_p = " ".join(
                    _cell_str(s)
                    for s in (table.col_paths[c] if c < len(table.col_paths) else [])
                )
                if not row_p or not col_p:
                    continue
                out.append(f"What is the {col_p} for {row_p}?")
                if len(out) >= self.n_questions:
                    break
            return out

        # flat: first column is the row key (WikiSQL convention)
        if not table.columns or not table.rows:
            return out
        key_col = 0
        value_cols = [c for c in range(1, len(table.columns))] or [0]
        rows = [r for r in table.rows if _cell_str(r[key_col])]
        rng.shuffle(rows)
        for row in rows:
            c = rng.choice(value_cols)
            key = _cell_str(row[key_col])
            col = _cell_str(table.columns[c])
            val = _cell_str(row[c]) if c < len(row) else ""
            if not key or not col:
                continue
            out.append(f"What is the {col} of {key}?")
            if val and len(out) < self.n_questions:
                out.append(f"Which {_cell_str(table.columns[key_col])} has {col} {val}?")
            if len(out) >= self.n_questions:
                break
        return out[: self.n_questions]


_SYNTH_PROMPT = """You write search queries for a table-retrieval index.
Given the table below, write {n} short, natural questions that this table
(and only a table like this) can answer. One question per line, no numbering.

{table}"""


class LLMSynth:
    """LLM-backed synthetic questions with a jsonl cache."""

    def __init__(self, llm_spec: str, n_questions: int = 5,
                 cache_path: Optional[str] = None, max_rows: int = 8):
        from rag_agent.llm.factory import build_llm

        self.llm = build_llm(llm_spec)
        self.n_questions = n_questions
        self.max_rows = max_rows
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: Dict[str, List[str]] = {}
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._cache[rec["table_id"]] = rec["questions"]

    def __call__(self, table: PrepTable) -> List[str]:
        if table.table_id in self._cache:
            return self._cache[table.table_id]
        partial = _markdown(table, max_rows=self.max_rows)
        raw = self.llm.complete(
            system="You generate retrieval queries for tables.",
            user=_SYNTH_PROMPT.format(n=self.n_questions, table=partial),
            max_tokens=300,
        )
        questions = []
        for line in raw.splitlines():
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
            if line.endswith("?") and len(line) > 10:
                questions.append(line)
        questions = questions[: self.n_questions]
        self._cache[table.table_id] = questions
        if self.cache_path:
            with open(self.cache_path, "a") as f:
                f.write(json.dumps(
                    {"table_id": table.table_id, "questions": questions},
                    ensure_ascii=False) + "\n")
        return questions


def build_synth(spec: str, n_questions: int = 5, cache_path: Optional[str] = None):
    """``template`` or ``llm:<llm_spec>`` (e.g. llm:local:Qwen/Qwen2.5-7B-Instruct)."""
    if spec == "template":
        return TemplateSynth(n_questions=n_questions)
    if spec.startswith("llm:"):
        return LLMSynth(spec[4:], n_questions=n_questions, cache_path=cache_path)
    raise ValueError(f"unknown synth spec {spec!r}")
