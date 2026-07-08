# SPDX-License-Identifier: MIT
"""Two-phase cell extraction via query decomposition.

Addresses the core bottleneck identified in v3.1 experiments: when a query
requires 3+ cells from a hierarchical table, a single-shot LLM call fails
to select the correct cells simultaneously.

**Phase 1 — Decompose**: ask the LLM to break the complex query into atomic
sub-questions (each targeting ONE cell) and an overall combining formula.

**Phase 2 — Locate**: for each sub-question, ask the LLM to identify exactly
one (row_header, col_header) pair in the table.

This yields the same ``ExtractedPlan`` as the original single-shot extractor,
so the downstream symbolic evaluator (``symbolic_eval.py``) works unchanged.

References:
  - PAL (Gao et al., 2023): LLM decomposes, program executes
  - Least-to-Most Prompting (Zhou et al., 2023): break into sub-problems
  - DATER (Ye et al., 2023): decompose-and-reason over tables
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional

from ..llm.base import BaseLLM
from ..stores.original_store import OriginalTable
from .cell_extractor import ExtractedCell, ExtractedPlan


def _format_table_compact(t: OriginalTable, max_rows: int = 30) -> str:
    cols_block = []
    for c in range(t.n_cols):
        p = t.col_path(c)
        cols_block.append(f"  col[{c}]: {' > '.join(p)}" if p else f"  col[{c}]: (blank)")
    rows_block = []
    for r in range(min(t.n_rows, max_rows)):
        p = t.row_path(r)
        rh = " > ".join(p) if p else f"row_{r}"
        vals = []
        for c in range(min(t.n_cols, 12)):
            v = t.cell(r, c)
            vals.append("" if v is None else str(v))
        rows_block.append(f"  row[{r}] ({rh}): " + " | ".join(vals))
    note = "" if t.n_rows <= max_rows else f"\n  (...{t.n_rows - max_rows} more rows truncated)"
    return (
        f"Title: {t.title}\n"
        f"Columns (header paths):\n" + "\n".join(cols_block) + "\n"
        f"Data:\n" + "\n".join(rows_block) + note
    )


# ---------------------------------------------------------------------------
# Phase 1: Decompose
# ---------------------------------------------------------------------------

_DECOMPOSE_SYSTEM = """\
You break a complex numerical table question into atomic sub-questions.
Each sub-question asks for exactly ONE numeric value from the table.
Then you specify the arithmetic formula that combines them into the final answer.

Output VALID JSON with this exact schema:
{
  "sub_questions": [
    {"var": "x1", "question": "What is the value of ... ?"},
    {"var": "x2", "question": "What is the value of ... ?"}
  ],
  "formula": "x1 + x2"
}

Rules:
- Each sub-question must target exactly ONE cell in the table.
- Variable names must be x1, x2, x3, ... in order.
- The formula uses only + - * / ( ) and the variable names.
- For "average of k values", output (x1+x2+...+xk)/k.
- For "difference", output x1 - x2.
- For "sum" or "total", output x1+x2+...+xk.
- If the question only needs ONE cell, output one sub-question and formula "x1".
- No commentary. JSON only.
"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(s: str) -> Optional[dict]:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.MULTILINE).strip()
    m = _JSON_RE.search(s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


@dataclass
class DecomposeResult:
    sub_questions: List[dict]  # [{"var": "x1", "question": "..."}, ...]
    formula: str
    raw_output: str
    ok: bool


def _decompose(llm: BaseLLM, query: str, table: OriginalTable) -> DecomposeResult:
    table_block = _format_table_compact(table)
    user = (
        f"Table:\n{table_block}\n\n"
        f"Question: {query}\n\n"
        "Break this into atomic sub-questions. Output JSON only."
    )
    raw = llm.complete(system=_DECOMPOSE_SYSTEM, user=user, max_tokens=500)
    obj = _parse_json(raw)
    if obj is None:
        return DecomposeResult([], "", raw, ok=False)

    subs = []
    for sq in obj.get("sub_questions", []) or []:
        if not isinstance(sq, dict):
            continue
        subs.append({
            "var": str(sq.get("var", "")).strip(),
            "question": str(sq.get("question", "")).strip(),
        })
    formula = str(obj.get("formula", "")).strip()
    return DecomposeResult(subs, formula, raw, ok=bool(subs and formula))


# ---------------------------------------------------------------------------
# Phase 2: Locate one cell per sub-question
# ---------------------------------------------------------------------------

_LOCATE_SYSTEM = """\
You find exactly ONE cell in a table that answers a specific question.
Use the table's row and column header paths (NOT Excel-style cell IDs).

Output VALID JSON with this exact schema:
{"row_header": "...", "col_header": "..."}

Rules:
- Pick header strings that uniquely identify the cell.
- Use the LEAF segment of the row/col path (case-insensitive).
- Match headers EXACTLY as they appear in the table. Do not paraphrase.
- No commentary. JSON only.
"""


def _locate_cell(
    llm: BaseLLM,
    sub_question: str,
    var_name: str,
    table: OriginalTable,
) -> Optional[ExtractedCell]:
    table_block = _format_table_compact(table)
    user = (
        f"Table:\n{table_block}\n\n"
        f"Question: {sub_question}\n\n"
        "Find the ONE cell that answers this. Output JSON only."
    )
    raw = llm.complete(system=_LOCATE_SYSTEM, user=user, max_tokens=150)
    obj = _parse_json(raw)
    if obj is None:
        return None
    rh = str(obj.get("row_header", "")).strip()
    ch = str(obj.get("col_header", "")).strip()
    if not rh and not ch:
        return None
    return ExtractedCell(var=var_name, row_header=rh, col_header=ch)


# ---------------------------------------------------------------------------
# Public API: drop-in replacement for extract_plan()
# ---------------------------------------------------------------------------

def extract_plan_decomposed(
    llm: BaseLLM,
    query: str,
    table: OriginalTable,
) -> ExtractedPlan:
    """Two-phase extraction: decompose → locate each cell independently."""
    decomp = _decompose(llm, query, table)
    if not decomp.ok:
        return ExtractedPlan(
            cells=[], expression="",
            raw_llm_output=decomp.raw_output, parse_ok=False,
        )

    cells: List[ExtractedCell] = []
    all_raw = [decomp.raw_output]

    for sq in decomp.sub_questions:
        cell = _locate_cell(llm, sq["question"], sq["var"], table)
        if cell is None:
            return ExtractedPlan(
                cells=cells, expression=decomp.formula,
                raw_llm_output="\n---\n".join(all_raw),
                parse_ok=False,
            )
        cells.append(cell)

    return ExtractedPlan(
        cells=cells,
        expression=decomp.formula,
        raw_llm_output="\n---\n".join(all_raw),
        parse_ok=True,
    )
