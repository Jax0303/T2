"""Serialization conditions for the pre-retrieval preprocessing experiment.

The independent variable is *what information is added to the indexed text*,
not the markup format (the pilot showed format alone is not the bottleneck
for BGE-small on flat tables). Conditions are CUMULATIVE:

  C0  raw          markdown table only (header + truncated rows)
  C1  +metadata    C0 + page/section title, caption (TARGET: titles dominate)
  C2  +schema      C1 + per-column name/type/example-values description
  C2h +schema-hier C1 + root-to-leaf header *paths* (hierarchical tables only;
                   the flattening used by HiTab / API-assisted codegen)
  C3  +synthetic   C2 (or C2h) + synthetic questions (QGpT-style)

A `PrepTable` is the dataset-neutral table representation; both
OpenWikiTable (flat) and HiTab (hierarchical, via OriginalTable) map onto it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

CONDITIONS = ["C0", "C1", "C2", "C2h", "C3"]

_NUM_RE = re.compile(r"^-?[\d,]+\.?\d*%?$")


@dataclass
class PrepTable:
    """Dataset-neutral table for serialization.

    For flat tables ``columns`` holds plain header names and ``col_paths`` /
    ``row_paths`` stay empty. For hierarchical tables ``columns`` holds the
    leaf header names while ``col_paths`` / ``row_paths`` hold the full
    root-to-leaf paths (one list of segments per column / row).
    """

    table_id: str
    columns: List[str]
    rows: List[List[object]]
    page_title: str = ""
    section_title: str = ""
    caption: str = ""
    col_paths: List[List[str]] = field(default_factory=list)
    row_paths: List[List[str]] = field(default_factory=list)

    @property
    def hierarchical(self) -> bool:
        return any(len(p) > 1 for p in self.col_paths) or any(
            len(p) > 1 for p in self.row_paths
        )


def _cell_str(v: object) -> str:
    if v is None:
        return ""
    return str(v).replace("\n", " ").strip()


def _markdown(table: PrepTable, max_rows: int) -> str:
    header = " | ".join(_cell_str(c) for c in table.columns)
    lines = [header, " | ".join("---" for _ in table.columns)]
    for r in table.rows[:max_rows]:
        lines.append(" | ".join(_cell_str(v) for v in r))
    return "\n".join(lines)


def _metadata_block(table: PrepTable) -> str:
    parts = []
    if table.page_title:
        parts.append(f"Title: {table.page_title}")
    if table.section_title:
        parts.append(f"Section: {table.section_title}")
    if table.caption:
        parts.append(f"Caption: {table.caption}")
    return "\n".join(parts)


def _infer_type(values: List[str]) -> str:
    non_empty = [v for v in values if v]
    if not non_empty:
        return "text"
    numeric = sum(1 for v in non_empty if _NUM_RE.match(v.replace(" ", "")))
    return "number" if numeric / len(non_empty) >= 0.7 else "text"


def _schema_block(table: PrepTable, n_examples: int = 3) -> str:
    """Flat schema description: column name (type), example values."""
    lines = [f"Columns ({len(table.columns)} columns, {len(table.rows)} rows):"]
    for ci, col in enumerate(table.columns):
        values = [_cell_str(r[ci]) for r in table.rows if ci < len(r)]
        ctype = _infer_type(values)
        seen: List[str] = []
        for v in values:
            if v and v not in seen:
                seen.append(v)
            if len(seen) >= n_examples:
                break
        ex = "; ".join(seen)
        lines.append(f"- {_cell_str(col)} ({ctype}): e.g. {ex}" if ex
                     else f"- {_cell_str(col)} ({ctype})")
    return "\n".join(lines)


def _schema_block_hier(table: PrepTable, max_row_paths: int = 30) -> str:
    """Hierarchical schema description: root-to-leaf header paths.

    This is the C2 variant for hierarchical tables — instead of leaf names
    it spells out every column path and (up to ``max_row_paths``) row paths,
    so a query phrased against any level of the header tree can match.
    """
    lines = [f"Column header paths ({len(table.columns)} columns):"]
    seen_cols: List[str] = []
    for ci in range(len(table.columns)):
        path = table.col_paths[ci] if ci < len(table.col_paths) else []
        text = " > ".join(_cell_str(s) for s in path) or _cell_str(table.columns[ci])
        if text and text not in seen_cols:
            seen_cols.append(text)
    lines += [f"- {t}" for t in seen_cols]
    if table.row_paths:
        lines.append("Row header paths:")
        seen_rows: List[str] = []
        for path in table.row_paths:
            text = " > ".join(_cell_str(s) for s in path)
            if text and text not in seen_rows:
                seen_rows.append(text)
            if len(seen_rows) >= max_row_paths:
                break
        lines += [f"- {t}" for t in seen_rows]
    return "\n".join(lines)


def serialize(
    table: PrepTable,
    condition: str,
    max_rows: int = 20,
    synth_provider: Optional[Callable[[PrepTable], List[str]]] = None,
) -> str:
    """Render ``table`` under a cumulative preprocessing condition.

    ``synth_provider`` is required for C3; it maps a table to a list of
    synthetic questions (template-based or LLM-generated, see prep.synth).
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected {CONDITIONS}")

    blocks: List[str] = []
    if condition in ("C1", "C2", "C2h", "C3"):
        meta = _metadata_block(table)
        if meta:
            blocks.append(meta)
    if condition == "C2":
        blocks.append(_schema_block(table))
    elif condition == "C2h":
        blocks.append(_schema_block_hier(table))
    elif condition == "C3":
        blocks.append(_schema_block_hier(table) if table.hierarchical
                      else _schema_block(table))
        if synth_provider is None:
            raise ValueError("C3 requires a synth_provider")
        questions = synth_provider(table)
        if questions:
            blocks.append("Questions answerable from this table:\n"
                          + "\n".join(f"- {q}" for q in questions))
    blocks.append(_markdown(table, max_rows=max_rows))
    return "\n\n".join(blocks)


# ---- dataset adapters -------------------------------------------------------

def from_openwikitable(rec: Dict) -> PrepTable:
    """Adapter for one record of the normalized OpenWikiTable corpus jsonl."""
    return PrepTable(
        table_id=str(rec["table_id"]),
        columns=[str(c) for c in rec.get("header", [])],
        rows=rec.get("rows", []),
        page_title=rec.get("page_title", "") or "",
        section_title=rec.get("section_title", "") or "",
        caption=rec.get("caption", "") or "",
    )


def from_hitab(original_table) -> PrepTable:
    """Adapter from stores.original_store.OriginalTable (hierarchical)."""
    n_cols = original_table.n_cols
    columns = []
    for c in range(n_cols):
        path = original_table.col_path(c)
        columns.append(path[-1] if path else f"col_{c}")
    return PrepTable(
        table_id=original_table.table_id,
        columns=columns,
        rows=original_table.data,
        page_title=original_table.title,
        col_paths=[original_table.col_path(c) for c in range(n_cols)],
        row_paths=[original_table.row_path(r) for r in range(original_table.n_rows)],
    )
