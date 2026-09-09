"""Lossless views of the fields already present in retrieved cell sentences.

No question, annotation, gold address, or additional table cell enters render().
The corpus dictionary only disambiguates the grammar of an existing sentence.
A sentence admitting more than one visible-field decomposition stays verbatim.
"""
from __future__ import annotations
from collections import Counter, OrderedDict
from dataclasses import dataclass
import json

@dataclass(frozen=True)
class VisibleCell:
    title: str
    row: str
    col: str
    value: str

def build_visible_lookup(data_dir, tids, template, page_titles):
    from rag_agent.bench.hitab_grid import load_table
    from rag_agent.serialization.base import fmt_value, join_path
    from rag_agent.serialization.caption import with_page_title
    from rag_agent.serialization.templates import render, STRUCTURAL_COMPACT, MT2NET, _path, MT2NET_ROW_SEP, MT2NET_COL_SEP, MT2NET_LEAF_FIRST
    options = {}
    for tid in sorted(tids):
        tab = load_table(tid, data_dir)
        t = tab.table
        title = with_page_title(tab.title, page_titles.get(tid))
        for i in range(t.n_rows):
            for j in range(t.n_cols):
                v = t.data[i][j]
                if not str(v).strip():
                    continue
                rp, cp = t.row_path(i), t.col_path(j)
                mode = STRUCTURAL_COMPACT if template == "s3c" else MT2NET
                text = render(mode, title, rp, cp, v)
                # Compact untitled sentences do not expose the axis boundary.
                # Preserve them, rather than importing that hidden boundary.
                if template == "s3c" and not title:
                    options.setdefault(text, set()).add(None)
                    continue
                if template == "s3c":
                    fields = VisibleCell(fmt_value(title), join_path(rp), join_path(cp), fmt_value(v))
                else:
                    fields = VisibleCell("", _path(rp, MT2NET_ROW_SEP, MT2NET_LEAF_FIRST),
                                         _path(cp, MT2NET_COL_SEP, MT2NET_LEAF_FIRST), fmt_value(v))
                options.setdefault(text, set()).add(fields)
    return {s: next(iter(v)) if len(v) == 1 and None not in v else None for s, v in options.items()}

def _escape(s):
    # Markdown only: preserve the displayed characters, do not rewrite numbers.
    return str(s).replace("&", "&amp;").replace("|", "&#124;").replace("\n", "&#10;").replace("\r", "&#13;").replace("<", "&lt;").replace(">", "&gt;")

def partition(context, lookup):
    groups = OrderedDict()
    fallback = []
    for rank, line in enumerate(context, 1):
        cell = lookup.get(line)
        if cell is None:
            fallback.append((rank, line))
        else:
            groups.setdefault(cell.title, []).append((rank, cell, line))
    return groups, fallback

def render_context(context, lookup, layout):
    if layout == "original":
        return "\n".join(context), {"n_input_lines": len(context), "n_preserved_lines": len(context), "fallback": 0}
    if layout not in {"sparse", "grouped"}:
        raise ValueError(layout)
    groups, fallback = partition(context, lookup)
    seen = []
    blocks = []
    for title, items in groups.items():
        heading = "Table: " + json.dumps(title, ensure_ascii=False) if title else "Retrieved cells (table titles were not supplied)"
        if layout == "grouped":
            lines = [heading]
            for rank, cell, original in items:
                prefix = "In the table '" + title + "', "
                body = original[len(prefix):] if title and original.startswith(prefix) else original
                lines.append(f"[#{rank}] {body}")
                seen.append(rank)
            blocks.append("\n".join(lines))
            continue
        rows = list(dict.fromkeys(c.row for _, c, _ in items))
        cols = list(dict.fromkeys(c.col for _, c, _ in items))
        buckets = {}
        for rank, c, _ in items:
            buckets.setdefault((c.row, c.col), []).append((rank, c.value))
        lines = [heading, "— means not retrieved. [#n] is the original retrieval rank.",
                 "| Row header path | " + " | ".join(_escape(c) or "(no column header)" for c in cols) + " |",
                 "| --- | " + " | ".join("---" for _ in cols) + " |"]
        for row in rows:
            values = []
            for col in cols:
                entries = buckets.get((row, col), [])
                if entries:
                    values.append("<br>".join(f"{_escape(v)} [#{rank}]" for rank, v in entries))
                    seen.extend(rank for rank, _ in entries)
                else:
                    values.append("—")
            lines.append("| " + (_escape(row) or "(no row header)") + " | " + " | ".join(values) + " |")
        blocks.append("\n".join(lines))
    if fallback:
        blocks.append("Unconverted retrieved cells:\n" + "\n".join(f"[#{r}] {s}" for r, s in fallback))
        seen.extend(r for r, _ in fallback)
    assert sorted(seen) == list(range(1, len(context) + 1)), "a retrieved line was lost or duplicated"
    return "\n\n".join(blocks), {"n_input_lines": len(context), "n_preserved_lines": len(seen), "fallback": len(fallback), "groups": len(groups)}
