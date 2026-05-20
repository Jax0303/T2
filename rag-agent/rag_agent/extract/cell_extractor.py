"""LLM-driven cell extraction.

The LLM emits a small JSON object naming the cells it needs by HEADER PATH
(not Excel ref) and an algebraic expression over named variables. This keeps
the LLM in its sweet spot (semantic header matching) and pushes arithmetic
to deterministic code.

Schema:

    {
      "cells": [
        {"var": "x1", "row_header": "total", "col_header": "2017 actual"},
        {"var": "x2", "row_header": "covid", "col_header": "2017 actual"}
      ],
      "expression": "x1 - x2"
    }

The agent uses the OriginalStore's header-path resolver to map each cell to
a numeric value, then evaluates ``expression`` in a sandboxed AST evaluator.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional

from ..llm.base import BaseLLM
from ..stores.original_store import OriginalTable


@dataclass
class ExtractedCell:
    var: str
    row_header: str
    col_header: str


@dataclass
class ExtractedPlan:
    cells: List[ExtractedCell]
    expression: str
    raw_llm_output: str = ""
    parse_ok: bool = True


# Format the table for the LLM as a compact header listing + first few rows.
def _format_table_for_extractor(t: OriginalTable, max_rows: int = 30) -> str:
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
        f"Columns (top-header paths):\n" + "\n".join(cols_block) + "\n"
        f"Data (left-header path → values; first {min(max_rows, t.n_rows)} rows):\n"
        + "\n".join(rows_block) + note
    )


SYS_PROMPT = """\
You map a numerical table-QA question to a set of cell references and an
arithmetic expression. Use the table's row and column header paths (NOT
Excel-style cell IDs). Output VALID JSON with this exact schema:

{
  "cells": [ {"var":"x1", "row_header":"...", "col_header":"..."}, ... ],
  "expression": "x1 + x2 - x3"
}

Rules:
- Use only operators + - * / ( ) and the variable names you declared.
- For ratios/percentages, output the raw ratio (e.g. x1/x2), not x1/x2*100,
  UNLESS the question explicitly asks for a percentage.
- For "difference" output x1 - x2 (do NOT take the absolute value here).
- For "average" of k cells, output (x1+x2+...+xk)/k.
- For "sum" or "total" output x1+x2+...+xk.
- For argmax/argmin style questions (which is highest?), DO NOT use this tool;
  return {"cells": [], "expression": ""} so the caller falls back to the LLM reader.
- Pick header strings that uniquely identify the cell. Use the LEAF segment of
  the row/col path you observed in the table above (case-insensitive).
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


def extract_plan(llm: BaseLLM, query: str, table: OriginalTable) -> ExtractedPlan:
    table_block = _format_table_for_extractor(table)
    user = (
        f"Table:\n{table_block}\n\n"
        f"Question: {query}\n\n"
        "Output JSON only."
    )
    raw = llm.complete(system=SYS_PROMPT, user=user, max_tokens=400)
    obj = _parse_json(raw)
    if obj is None:
        return ExtractedPlan(cells=[], expression="", raw_llm_output=raw, parse_ok=False)
    cells = []
    for c in obj.get("cells", []) or []:
        if not isinstance(c, dict):
            continue
        cells.append(ExtractedCell(
            var=str(c.get("var", "")).strip(),
            row_header=str(c.get("row_header", "")).strip(),
            col_header=str(c.get("col_header", "")).strip(),
        ))
    expr = str(obj.get("expression", "")).strip()
    return ExtractedPlan(cells=cells, expression=expr, raw_llm_output=raw, parse_ok=True)
