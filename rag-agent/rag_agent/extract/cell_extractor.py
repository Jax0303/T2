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

The agent uses the parsed table's header-path resolver to map each cell to
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
- You may also use these functions: max(), min(), abs(), int().
  Example: "max(x1, x2, x3)" or "int(x1 / x2)".
- For ratios/percentages ("what percentage of X is Y?"), you MUST have
  SEPARATE cells for the numerator(s) AND the denominator. Output x1/x2,
  NOT (x1+x2)/len.  The denominator is typically a "total" row.
- For "difference" output x1 - x2 (do NOT take the absolute value here).
- For "average" of k cells, output (x1+x2+...+xk)/k.
- For "sum" or "total" output x1+x2+...+xk.
- For argmax/argmin style questions (which is highest?), DO NOT use this tool;
  return {"cells": [], "expression": ""} so the caller falls back to the LLM reader.
- Pick header strings that uniquely identify the cell. COPY the EXACT path
  text from the row/col headers shown above (case-insensitive). Do NOT
  paraphrase or abbreviate header text.
- No commentary. JSON only.

--- FEW-SHOT EXAMPLES ---

Example 1 — Ratio/percentage ("what was the percentage of X among Y?"):
Suppose the table has row "theft" at row[5] and "total" at row[20],
column "female accused" at col[1].
Q: "What was the percentage of theft among female accused?"
Correct output:
{"cells":[{"var":"x1","row_header":"theft","col_header":"female accused"},{"var":"x2","row_header":"total","col_header":"female accused"}],"expression":"x1 / x2"}
WRONG: {"cells":[{"var":"x1","row_header":"theft","col_header":"female accused"}],"expression":"x1"}  (missing denominator)
WRONG: {"cells":[...], "expression":"(x1 + x2) / 2"}  (averaging instead of dividing)

Example 2 — Multi-row sum ("percentage of A, B and C consisting of Z"):
Suppose rows are "southern asia" at row[15], "southeast asia" at row[16],
"east asia" at row[17], column "economic class" at col[0].
Q: "What is the percentage of southern asia, southeast asia and east asia consisting of economic immigrants?"
Correct output:
{"cells":[{"var":"x1","row_header":"southern asia","col_header":"economic class"},{"var":"x2","row_header":"southeast asia","col_header":"economic class"},{"var":"x3","row_header":"east asia","col_header":"economic class"}],"expression":"(x1 + x2 + x3)"}

Example 3 — Division with named denominator:
Q: "How many times more X than Y?"
Correct output:
{"cells":[{"var":"x1","row_header":"X","col_header":"value"},{"var":"x2","row_header":"Y","col_header":"value"}],"expression":"x1 / x2"}
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
