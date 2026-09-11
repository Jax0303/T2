#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Retrieval accuracy on HiTab: per query, was the retrieval right or wrong?

One number, defined the way the dataset lets it be defined. HiTab says which
cells each answer is read from, so a query is not scored on "how much of the
gold was covered" or "how deep the first hit was" — it is scored PASS or FAIL,
and the accuracy is the pass rate over the whole split:

    accuracy = (# queries whose retrieved context contains the answer evidence)
               / (# queries)

The cell budget is a stopping threshold. The last unit is kept whole, so larger
units may exceed it. Actual delivered cells and unit texts are recorded together.
This is an index-representation comparison, not a reproduction of full published
systems. Two gold shapes exist, both from the annotation:

  all  the answer is read off specific data cells (``answer_formulas``) — every
       one of them must be in the context.
  any  the answer IS a header ("which province had the highest rate"), which no
       cell index holds as a unit; the header rides in the sentence of every
       cell it labels, so any one of those cells serves. Reported as its own
       row, never pooled into the headline, because its bar is lower.

Queries whose gold cannot be resolved are EXCLUDED BY NAME with a reason and
counted in the summary. They are not dropped into the denominator's shadow.

  PYTHONPATH=. .venv/bin/python scripts/retrieval_accuracy.py --arm hybrid
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                                    # noqa: E402

from rag_agent.bench import hitab_grid as hg                          # noqa: E402
from rag_agent.data.loader import load_samples                        # noqa: E402
from rag_agent.retrieve.encoders import default_encoder               # noqa: E402
from rag_agent.retrieve.hybrid_index import _minmax, _tokenize        # noqa: E402
from rag_agent.retrieve.sparse_bm25 import SparseBM25                 # noqa: E402
from rag_agent.serialization.caption import (caption_sentence,        # noqa: E402
                                             with_page_title)
from rag_agent.serialization import tablerag_unit as trag             # noqa: E402
from rag_agent.serialization.templates import (MT2NET, STRUCTURAL,    # noqa: E402
                                               STRUCTURAL_COMPACT)
from rag_agent.eval.artifacts import (Selection, digest, evidence_fields, file_digest,
                                     provenance, validate_retrieval, write_pair)

TEMPLATES = {"s3c": STRUCTURAL_COMPACT, "s3": STRUCTURAL, "mt2net": MT2NET}
# Two ablations of the index unit itself, not templates: they answer "what does
# the header path buy, and what does the table's own label buy" by removing one
# at a time. Byte-identical to point3_reconstruction_cost.cell_text(..., "flat")
# and (..., "S2"), so they stay comparable with the older numbers on disk.
ABLATIONS = ("flat", "s2")
# cell/row/table are index-unit sizes of OUR sentence. chunk and tablerag are
# representation adaptations and ignore --template. Neither `chunk` nor
# `tablerag` is a reproduction of a published end-to-end system.
UNITS = ("cell", "row", "table", "chunk", "tablerag", "trag_hetero", "rowcol")
PAGE_TITLES = ROOT / "results/tableconf/totto_page_titles.json"


def cell_unit(title, row_path, col_path, value, template: str) -> str:
    """One index unit's text under ``template``.

    ``flat`` keeps only the LEAF labels -- the hierarchy is gone, and so is the
    table's own label. ``s2`` keeps the full header path but no table label.
    Together they separate the two things the deployed unit carries.
    """
    if template == "flat":
        lab = " ".join(x for x in ((row_path[-1] if row_path else ""),
                                   (col_path[-1] if col_path else "")) if x)
        return f"{lab}: {value}" if lab else str(value)
    if template == "s2":
        path = " > ".join([*row_path, *col_path])
        return f"{path}: {value}" if path else str(value)
    return caption_sentence(title, row_path, col_path, value=value,
                            template=TEMPLATES[template])


def raw_lines(tab, t):
    """The sheet as a reader of the .xlsx sees it — every raw row, headers included.

    ``excel_to_markdown`` walks openpyxl rows, so a hierarchical table's whole
    multi-level header block arrives as ordinary text at the top of the render.
    Rendering only the parsed leaf labels would delete header levels the baseline
    actually receives, and the header is the axis this table measures — the cut
    has to be theirs, not ours.

    Returns ``[(markdown line, [(i, j) …])]``; header lines carry no data cell.
    ``row_map`` / ``col_map`` are the raw→parsed index maps the grid already
    carries, so a line's cells are exact rather than matched by text.
    """
    out = []
    for r, row in enumerate(tab.raw.get("texts") or []):
        cells = [str(v) for v in row]
        if not any(c.strip() for c in cells):
            continue
        i = tab.row_map.get(r)
        got = []
        if i is not None:
            for c in range(len(cells)):
                j = tab.col_map.get(c)
                if j is not None and str(t.data[i][j]).strip():
                    got.append((i, j))
        out.append(("| " + " | ".join(cells) + " |", got))
    return out


def _pack(lines, head, size, overlap, per_chunk_head):
    """Greedy merge of rendered lines into chunks, carrying `overlap` chars.

    Custom row-preserving packing, not LangChain's recursive splitter.
    ``per_chunk_head`` repeats the header block in every chunk. The Huawei
    adaptation uses the separate native splitter in serialization/chunks.py.
    """
    out, buf, cells, size_now = [], [], [], 0
    for line, cs in lines:
        if buf and size_now + len(line) + 1 > size:
            out.append(("\n".join([head, *[b for b, _ in buf]]) if per_chunk_head
                        else "\n".join(b for b, _ in buf), cells))
            keep, tot = [], 0
            for b in reversed(buf):
                if tot >= overlap:
                    break
                keep.insert(0, b)
                tot += len(b[0]) + 1
            buf, cells, size_now = keep, [c for _b, cs2 in keep for c in cs2], tot
        buf.append((line, cs))
        cells = cells + cs
        size_now += len(line) + 1
    if buf:
        out.append(("\n".join([head, *[b for b, _ in buf]]) if per_chunk_head
                    else "\n".join(b for b, _ in buf), cells))
    return out


def markdown_chunks(tab, t, title, chunk_chars: int):
    """A general-purpose RAG stack's view: markdown, split on a character budget.

    Custom row-preserving packing with a repeated header block. The size is a
    stopping threshold, not a hard character cap or a library default.
    """
    lines = raw_lines(tab, t)
    # 헤더 블록은 셀을 배달하지 않는 줄의 **선두 연속**이다. 전체 개수를 세면
    # 표 중간의 섹션 구분 행(빈 줄)까지 더해져 head_n 이 부풀고, 그만큼 진짜
    # 데이터 줄이 잘려 나간다 -- 40표 표본에서 셀 153개가 사라졌다.
    head_n = 0
    while head_n < len(lines) and not lines[head_n][1]:
        head_n += 1
    head = "\n".join([f"# {title}"] + [l for l, _cs in lines[:head_n]])
    return _pack(lines[head_n:], head, chunk_chars, 0, True)


def trag_hetero_chunks(tab, t, name: str, chunk_chars: int, overlap: int,
                      tokenizer=None):
    """Huawei TableRAG's retrieval leg (arXiv 2506.10380, EMNLP 2025).

    ``online_inference/tools/retriever.py`` renders each workbook to markdown
    (``utils/tool_utils.py: excel_to_markdown`` — ``"Table name: {name}"`` then
    one pipe row per sheet row), splits it with ``RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)``, and prepends ``"File name: {key}"`` to
    every chunk. That is the whole unit.

    The ``{"table_name", "column_list"}`` schema (``src/data_persistent.py:
    generate_schema_info``) never reaches the embedding index — it goes to
    ``NL2SQL_USER_PROMPT`` and feeds SQL against their MySQL copy. The SQL leg
    has no counterpart here (our reader gets retrieved text and no database), so
    this arm adapts their chunk representation, without their BGE-M3/reranker
    or other retrieval orchestration. It does not reproduce the retrieval leg.

    The table name rides in the FIRST chunk only, because it is prepended before
    the split. That is theirs, not a handicap we added. ``name`` is the table's
    TITLE rather than its ``table_id``: their ``table_name`` comes from the
    workbook's file name, which carries meaning in their corpus while
    ``0_1_nsf21326-tab001`` carries none.

    NOTE the paper and the code disagree on the size. Section 5.1.2 says "1000
    tokens, with a 200-token overlap"; the code counts CHARACTERS, which is
    LangChain's default unit. Token mode requires an explicit tokenizer;
    the historical 2,400-character run was not a 1,000-token measurement.
    """
    # The HiTab renderer keeps blank columns to preserve alignment; it is an
    # explicit adaptation of their Excel renderer. Character splitting uses
    # the real LangChain splitter, not our row-preserving generic _pack.
    from rag_agent.serialization.chunks import table_chunks
    return list(table_chunks(tab, t, name, chunk_chars, overlap, tokenizer))


def tablerag_units(tab, t, mode: str):
    """TableRAG's corpus for one table, with the cells each doc delivers.

    Text comes from the port (``rag_agent/serialization/tablerag_unit.py``) so
    it stays byte-identical to the published recipe. The cell mapping is ours
    and it is the charitable one: a numeric column's summary carries that
    column's min and max, so it delivers those two cells; a deduplicated
    categorical doc names a (column, value) pair, so it delivers every cell in
    the column holding that value. TableRAG itself keeps no row identity — the
    reader is handed the *value*, not the address — so this is the most
    evidence any honest reading can credit it with.
    """
    out = []
    numeric = [c for c in range(t.n_cols) if trag._is_numeric_column(t, c)]
    for c in numeric:
        vals = {}
        for r in range(t.n_rows):
            v = trag.fmt_value(t.cell(r, c))
            if v:
                try:
                    vals[float(v.replace(",", ""))] = r
                except ValueError:
                    pass
        rows = {vals[min(vals)], vals[max(vals)]} if vals else set()
        out.append((trag.schema_doc(t, c, mode),
                    [(r, c) for r in sorted(rows)]))
    seen: dict = {}
    for c in range(t.n_cols):
        if c in numeric:
            continue
        for r in range(t.n_rows):
            if not trag.fmt_value(t.cell(r, c)):
                continue
            seen.setdefault(trag.cell_doc(t, r, c, mode), []).append((r, c))
    # The stub column. `build_cell_corpus` walks a FLAT read, where the row
    # labels are an ordinary categorical column and therefore get their own
    # `{"column_name", "cell_value"}` docs. Our parse keeps headers out of
    # `t.data`, so omitting them would be a handicap we invented -- and not a
    # small one: such a doc carries no data cell, so it costs zero of the cell
    # budget while still letting the retriever find the table by a row label.
    stub = ""
    for row in (tab.raw.get("texts") or []):
        if row and str(row[0]).strip():
            stub = str(row[0]).strip()
            break
    for r in range(t.n_rows):
        lab = trag.fmt_value(t.row_path(r)[-1]) if t.row_path(r) else ""
        if lab:
            seen.setdefault(f'{{"column_name": "{stub}", "cell_value": "{lab}"}}',
                            [])
    # `build_schema_corpus`, the half the earlier port left out. One doc per
    # column: numeric columns already have theirs above, categorical columns get
    # `cell_examples` (TableRAG's top-3 by frequency). They deliver no cell of
    # their own, so they cost nothing of the budget -- including them can only
    # help TableRAG find the table, never hurt it.
    for c in range(t.n_cols):
        if c in numeric:
            continue
        vals = Counter(trag.fmt_value(t.cell(r, c)) for r in range(t.n_rows)
                       if trag.fmt_value(t.cell(r, c)))
        if not vals:
            continue
        ex = ", ".join(f'"{v}"' for v, _ in vals.most_common(3))
        seen.setdefault(f'{{"column_name": "{trag._col_name(t, c, mode)}", '
                        f'"dtype": "object", "cell_examples": [{ex}]}}', [])
    out += sorted(seen.items(), key=lambda kv: -len(kv[1]))
    return out


def line_text(t, cells, title: str, template: str, row_text: str,
              include_row_label: bool = True) -> str:
    """One row's / one column's index unit text.

    ``sentence`` joins this repo's own cell sentences -- every cell carries the
    table title and both header paths. That is NOT the published row unit; it is
    an ablation of OUR unit's granularity, kept because the results already on
    disk were produced with it.

    ``values`` is the published one. TableRAG's ``build_row_corpus`` /
    ``build_column_corpus`` (Chen et al., NeurIPS 2024) write
    ``'|'.join(str(cell) for cell in row)`` -- bare values, no header PATH. That
    absence is the baseline's defining property; putting our header path back in
    erases the axis the table is measuring.

    A row DOES carry its own label, because ``df.iterrows()`` walks a flat read
    where the stub column is an ordinary column: the row's leaf label is cell 0
    of ``row`` and lands in the joined string. A column does not carry its name,
    because ``df.items()`` joins only the Series values. That asymmetry is
    theirs. The label column itself would be one more column unit delivering no
    data cell; it is omitted, which spends one fewer slot than they would.

    The title is prefixed once even in ``values`` mode. Their retriever is built
    per table (``init_retriever(table_id, df)``) so it never has to identify one;
    ours searches 538 tables at once and a bare ``"52.1|60.6"`` addresses none of
    them. Every other arm gets the same courtesy, and without it this arm would
    be measuring our corpus scale instead of their index unit.
    """
    if row_text == "values":
        rows = {i for i, _j in cells}
        lead = ([t.row_path(next(iter(rows)))[-1]]
                if include_row_label and len(rows) == 1 and t.row_path(next(iter(rows))) else [])
        vals = "|".join(lead + [str(t.data[i][j]) for i, j in cells])
        return f"{title} | {vals}" if title else vals
    return " | ".join(
        cell_unit(title if k == 0 else "", t.row_path(i), t.col_path(j),
                  t.data[i][j], template)
        for k, (i, j) in enumerate(cells))


def build_corpus(data_dir: str, tids, template: str, unit: str, page_titles: dict,
                 chunk_chars: int = 1000, trag_mode: str = "leaf",
                 row_text: str = "sentence", chunk_overlap: int = 200,
                 chunk_tokenizer=None):
    """(texts, cell_sets, is_row, unit_tids, grid) — one index unit per entry, the cells
    it delivers, (``rowcol`` only) whether it is a row unit or a column unit, and
    the table it came from. ``unit_tids`` is what ``--corpus gold`` masks on: a
    unit carrying no cell (TableRAG's stub docs) still belongs to one table."""
    texts, covers, is_row, unit_tids, grid = [], [], [], [], {}
    for tid in tids:
        tab = hg.load_table(tid, data_dir)
        if tab is None:
            continue
        before = len(texts)
        t = tab.table
        title = with_page_title(tab.title, page_titles.get(tid))
        live = [(i, j) for i in range(t.n_rows) for j in range(t.n_cols)
                if str(t.data[i][j]).strip()]
        if unit == "cell":
            for i, j in live:
                texts.append(cell_unit(title, t.row_path(i), t.col_path(j),
                                       t.data[i][j], template))
                covers.append(frozenset([(tid, i, j)]))
        elif unit == "row":
            for i in range(t.n_rows):
                cs = [(i, j) for (a, j) in live if a == i]
                if not cs:
                    continue
                texts.append(line_text(t, cs, title, template, row_text))
                covers.append(frozenset((tid, i, j) for _i, j in cs))
        elif unit == "chunk":
            for txt, cs in markdown_chunks(tab, t, title, chunk_chars):
                texts.append(txt)
                covers.append(frozenset((tid, i, j) for i, j in cs))
        elif unit == "trag_hetero":
            for txt, cs in trag_hetero_chunks(tab, t, title, chunk_chars,
                                             chunk_overlap, chunk_tokenizer):
                texts.append(txt)
                covers.append(frozenset((tid, i, j) for i, j in cs))
        elif unit == "tablerag":
            for txt, cs in tablerag_units(tab, t, trag_mode):
                texts.append(txt)
                covers.append(frozenset((tid, i, j) for i, j in cs))
        elif unit == "rowcol":
            grid[tid] = (t, title)
            # RowColRetrieval (TableRAG, Chen et al. NeurIPS 2024, §4.2; methodology
            # from Sui et al., TAP4LM). Rows AND columns are encoded separately; the
            # context is the SUB-TABLE their top-K lists intersect, which the official
            # `Retriever.sample_rows_and_columns` writes as `df.iloc[row_ids, col_ids]`.
            # Cell text mirrors `build_row_corpus` / `build_column_corpus`: the line's
            # values joined, no header path -- that absence is the baseline, not a
            # handicap. The title rides on the first cell so the unit is findable at
            # corpus scale, the same courtesy every other arm gets.
            for i in range(t.n_rows):
                cs = [(i, j) for (a, j) in live if a == i]
                if not cs:
                    continue
                texts.append(line_text(t, cs, title, template, row_text))
                covers.append(frozenset((tid, i, j) for _i, j in cs))
                is_row.append(True)
            for j in range(t.n_cols):
                cs = [(i, j) for (i, b) in live if b == j]
                if not cs:
                    continue
                texts.append(line_text(t, cs, title, template, row_text, include_row_label=False))
                covers.append(frozenset((tid, i, j) for i, _j in cs))
                is_row.append(False)
        else:                                     # whole table as one unit
            texts.append(" | ".join(
                cell_unit(title if k == 0 else "", t.row_path(i),
                          t.col_path(j), t.data[i][j], template)
                for k, (i, j) in enumerate(live)))
            covers.append(frozenset((tid, i, j) for i, j in live))
        unit_tids.extend([tid] * (len(texts) - before))
    return texts, covers, is_row, unit_tids, grid


def budget_select(order, covers, texts, budget: int, dump: int, max_units: int = 0):
    """Top units until the context holds ``budget`` DISTINCT cells.

    The budget is what the reader receives, so a unit that repeats a cell an
    earlier unit already delivered does not spend budget for it. Counting the
    repeats instead costs an overlapping arm twice: it stops early on cells the
    reader never gained, and its reported context size overstates what was in
    the prompt. ``trag_hetero`` carries 200 characters of overlap and delivers
    15,510 of the split's 67,664 cells more than once, so this is not a corner
    case. ``rowcol_select`` counts the distinct set already; this is that rule.
    """
    if budget <= 0 or max_units < 0 or dump < 0:
        raise ValueError("budget must be positive; dump/max_units must be nonnegative")
    got, units = set(), []
    for n, p in enumerate(order):
        if len(got) >= budget or (max_units and n >= max_units):
            break
        got |= covers[p]
        units.append({"index_unit": int(p), "text": texts[p], "cells": sorted(covers[p])})
    # dump enables export; it never truncates the evidence. --max-units changes
    # selection explicitly when a cap on retrieved units is wanted.
    return Selection(got, units, bool(dump))


def subtable_units(got, grid, template="s3c", row_text="values",
                   context_format="markdown", header_mode="leaf"):
    """A sub-table keeps its column labels, as upstream df.to_markdown does.

    Legacy row-only values remain an explicit historical ablation. The default
    leaf labels preserve the index representation; full paths are opt-in.
    """
    if context_format == "markdown":
        from types import SimpleNamespace
        tabs = {tid: SimpleNamespace(table=t, title=title) for tid, (t, title) in grid.items()}
        return subtable_context(got, tabs, "", {}, header_mode=header_mode)
    if context_format != "legacy_values":
        raise ValueError("unknown RowCol context format")
    units = []
    for tid in sorted({c[0] for c in got}):
        table, title = grid[tid]
        for i in sorted({c[1] for c in got if c[0] == tid}):
            cells = sorted((r, c) for t, r, c in got if t == tid and r == i)
            units.append({"table_id": tid,
                          "text": line_text(table, cells, title, template, row_text),
                          "cells": [(tid, r, c) for r, c in cells]})
    return units


def subtable_lines(got, grid, template="s3c", row_text="values", **kwargs):
    return [u["text"] for u in subtable_units(got, grid, template, row_text, **kwargs)]


def rowcol_select(order, covers, texts, is_row, budget: int, dump: int, grid=None,
                  template: str = "s3c", row_text: str = "values", cap: int = 200,
                  render_context=None, context_format="markdown", header_mode="leaf"):
    """RowColRetrieval's sub-table: top-K rows INTERSECT top-K columns.

    ``sample_rows_and_columns`` returns ``df.iloc[row_ids, col_ids]``, so a cell
    is delivered only when BOTH its row unit and its column unit were retrieved.
    K is one number for both lists in the paper, so it grows on both together
    here, and the cell budget decides where it stops -- the same rule every other
    arm gets, since a fixed K would hand this arm a different amount of context.
    Intersecting across a corpus is safe because ``covers`` carries the table id:
    a row of one table and a column of another share no cell.
    """
    if budget <= 0 or dump < 0 or cap <= 0:
        raise ValueError("invalid RowCol budget, dump or cap")
    rows = [p for p in order if is_row[p]]
    cols = [p for p in order if not is_row[p]]
    R, C, got, k = set(), set(), set(), 0
    while k < min(cap, max(len(rows), len(cols))):
        if k < len(rows):
            R |= covers[rows[k]]
        if k < len(cols):
            C |= covers[cols[k]]
        k += 1
        got = R & C
        if len(got) >= budget:
            break
    if render_context is None and grid is not None:
        render_context = lambda cs: subtable_units(cs, grid, template, row_text,
                                                 context_format, header_mode)
    if dump and render_context is None:
        raise ValueError("RowCol export requires a sub-table renderer")
    units = render_context(got) if render_context is not None else []
    if render_context is not None:
        rendered = set().union(*(set(map(tuple, u["cells"])) for u in units))
        if rendered != got:
            raise ValueError("RowCol renderer changed the selected intersection")
    return Selection(got, units, bool(dump))


def subtable_context(cells, tabs, data_dir, page_titles, header_mode="path"):
    """Render exactly the selected intersection with its row/column labels.

    Full paths disambiguate hierarchical HiTab headers. This is a documented
    HiTab adaptation, not the original flat-table RowColRetrieval system.
    """
    if header_mode not in {"leaf", "path"}:
        raise ValueError("unknown RowCol header mode")
    def label(path):
        return " > ".join(path if header_mode == "path" else path[-1:])
    def escape(value):
        return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>").replace("\r", "")
    units = []
    for tid in sorted({c[0] for c in cells}):
        tab = tabs.get(tid) or hg.load_table(tid, data_dir)
        if tab is None:
            raise ValueError(f"cannot render missing table {tid}")
        tabs[tid] = tab
        table = tab.table
        selected = {c for c in cells if c[0] == tid}
        rows = sorted({c[1] for c in selected})
        cols = sorted({c[2] for c in selected})
        title = with_page_title(tab.title, page_titles.get(tid))
        text = [f"# {escape(title)}",
                "| row | " + " | ".join(escape(label(table.col_path(j))) for j in cols) + " |",
                "| --- | " + " | ".join("---" for _ in cols) + " |"]
        for i in rows:
            values = [escape(table.data[i][j]) if (tid, i, j) in selected else "" for j in cols]
            text.append("| " + escape(label(table.row_path(i))) + " | " + " | ".join(values) + " |")
        units.append({"table_id": tid, "text": "\n".join(text), "cells": sorted(selected)})
    return units


def load_queries(data_dir: str, split: str, tabs):
    """Every question in the split, with its gold target and why it is excluded."""
    out, seen = [], set()
    for s in load_samples(data_dir, split):
        if s["id"] in seen:
            raise ValueError(f"duplicate query_id in dataset: {s['id']}")
        seen.add(s["id"])
        tid = s["table_id"]
        tab = tabs.get(tid) or hg.load_table(tid, data_dir)
        tabs[tid] = tab
        if tab is None:
            out.append({"query_id": s["id"], "question": s["question"],
                        "answer": s.get("answer"), "table_id": tid,
                        "gold": set(), "mode": "all", "excluded": "table_missing"})
            continue
        gold, mode, why = hg.gold_target(s, tab)
        out.append({"query_id": s["id"], "question": s["question"],
                    "answer": s.get("answer"), "table_id": tid,
                    "aggregation": (s.get("aggregation") or [None])[0]
                    if isinstance(s.get("aggregation"), list) else s.get("aggregation"),
                    "gold": gold, "mode": mode, "excluded": why})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--corpus", default="all", choices=["all", "split", "gold"],
                    help="all = every table in the store (the real haystack); "
                         "split = only tables this split's questions touch, "
                         "gold = build over the split but search ONLY inside each "
                         "query's own table (TableRAG's per-table setting)")
    ap.add_argument("--template", default="s3c",
                    choices=list(TEMPLATES) + list(ABLATIONS))
    ap.add_argument("--unit", default="cell", choices=UNITS)
    ap.add_argument("--chunk-chars", type=int, default=1000,
                    help="character chunk size; generic chunk preserves whole rows")
    ap.add_argument("--chunk-tokens", type=int, default=0,
                    help="true token windows for trag_hetero; requires --chunk-tokenizer")
    ap.add_argument("--chunk-overlap", type=int, default=200,
                    help="overlap in the same unit as the chunk size")
    ap.add_argument("--chunk-tokenizer", default="")
    ap.add_argument("--chunk-tokenizer-revision", default=None)
    ap.add_argument("--row-text", default="sentence", choices=["sentence", "values"],
                    help="--unit row/rowcol: 'values' is the published unit "
                         "(TableRAG build_row_corpus: bare values, no header "
                         "path); 'sentence' joins OUR cell sentences and is an "
                         "ablation of our own granularity, not a baseline.")
    ap.add_argument("--tablerag-colmode", default="leaf", choices=["leaf", "path"],
                    help="--unit tablerag: 'leaf' is what a plain read of a "
                         "hierarchical table yields, 'path' hands TableRAG the "
                         "joined header path. Report both.")
    ap.add_argument("--embed-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--embed-revision", default=None)
    ap.add_argument("--embed-overflow", choices=["error", "truncate"], default="error",
                    help="refuse hidden encoder truncation unless explicitly allowed")
    ap.add_argument("--alpha", type=float, default=0.7,
                    help="dense weight. 0 = BM25 only, 1 = dense only. The "
                         "default was fixed on the dev split in an earlier "
                         "phase and is not re-tuned here.")
    ap.add_argument("--budget", type=int, default=20,
                    help="stop after this many distinct cells; the last unit stays whole")
    ap.add_argument("--no-query-prefix", action="store_true",
                    help="encode the question with no instruction prefix — what "
                         "the pipeline did before the fix, kept so the cost of "
                         "that bug is a measured number rather than a claim")
    ap.add_argument("--max-units", type=int, default=0,
                    help="also stop after N index units (TableRAG's native "
                         "top_k=5 counts documents, not cells). 0 = no unit cap")
    ap.add_argument("--rowcol-max-pairs", type=int, default=200,
                    help="maximum ranked row/column pairs considered by RowCol")
    ap.add_argument("--rowcol-context", choices=["markdown", "legacy_values"], default="markdown")
    ap.add_argument("--rowcol-header", choices=["leaf", "path"], default="leaf")
    ap.add_argument("--cache-dir", default=".cache/retrieval_accuracy")
    ap.add_argument("--dump-context", type=int, default=1,
                    help="0 = no reader export; any positive value exports ALL selected units")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default="results/retrieval_accuracy")
    a = ap.parse_args()
    if a.budget <= 0 or a.max_units < 0 or a.dump_context < 0 or not 0 <= a.alpha <= 1:
        ap.error("invalid budget, max-units, dump-context or alpha")
    if a.unit == "rowcol" and a.max_units:
        ap.error("--max-units is not a RowCol row/column-pair cap")
    if a.rowcol_max_pairs <= 0:
        ap.error("--rowcol-max-pairs must be positive")
    if a.chunk_tokens and (a.unit != "trag_hetero" or not a.chunk_tokenizer):
        ap.error("--chunk-tokens requires --unit trag_hetero and --chunk-tokenizer")
    if not a.chunk_tokens and a.chunk_tokenizer:
        ap.error("--chunk-tokenizer requires --chunk-tokens")
    size = a.chunk_tokens or a.chunk_chars
    if size <= 0 or (a.unit == "trag_hetero" and not 0 <= a.chunk_overlap < size):
        ap.error("invalid chunk size/overlap")
    tag = a.tag or f"{a.split}_{a.corpus}_{a.unit}_{a.template}_a{a.alpha}_k{a.budget}_v2"
    if Path(tag).name != tag or any(c in tag for c in ("/", "\\")):
        ap.error("tag must be a filename, not a path")
    out = Path(a.out_dir)
    if any((out / f"{tag}{suffix}").exists() for suffix in (".json", "_records.jsonl")):
        ap.error("output exists; use a new tag so legacy evidence is preserved")
    tokenizer = None
    if a.chunk_tokens:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(a.chunk_tokenizer,
                                                 revision=a.chunk_tokenizer_revision)

    t0 = time.time()
    page_titles = json.loads(PAGE_TITLES.read_text()) if PAGE_TITLES.exists() else {}
    tabs: dict = {}
    queries = load_queries(a.data_dir, a.split, tabs)
    tids = (hg.table_ids(a.data_dir) if a.corpus == "all"
            else sorted({q["table_id"] for q in queries}))
    texts, covers, is_row, unit_tids, grid = build_corpus(a.data_dir, tids, a.template, a.unit,
                                         page_titles, size, a.tablerag_colmode,
                                         a.row_text, a.chunk_overlap, tokenizer)
    unit_tid_arr = np.array(unit_tids)
    assert len(unit_tid_arr) == len(texts), "unit->table map lost a unit"
    print(f"[corpus] {len(tids)} tables / {len(texts)} {a.unit} units "
          f"({time.time() - t0:.0f}s)", flush=True)

    t0 = time.time()
    bm = SparseBM25(_tokenize(t) for t in texts)
    print(f"[bm25] {len(bm.vocab)} terms in {time.time() - t0:.0f}s", flush=True)

    enc = default_encoder(model_name=a.embed_model, revision=a.embed_revision) if a.alpha > 0 else None
    if enc is not None and a.no_query_prefix:
        enc.query_prefix = ""
    input_audit = None
    if enc is not None:
        input_audit = {"documents": enc.audit_inputs(texts, overflow=a.embed_overflow),
                       "queries": enc.audit_inputs([q["question"] for q in queries],
                                                   query=True, overflow=a.embed_overflow)}
    emb = None
    if a.alpha > 0:
        cache = Path(a.cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        key = digest({"texts": texts, "encoder": enc.metadata(),
                      "overflow": a.embed_overflow})[:24]
        f = cache / f"{enc.name.replace('/', '_')}_{len(texts)}_{key}.npy"
        if f.exists():
            emb = np.load(f)
            if emb.ndim != 2 or len(emb) != len(texts) or not np.isfinite(emb).all():
                raise ValueError(f"invalid embedding cache: {f}")
            print(f"[dense] cache hit {f.name}", flush=True)
        else:
            t0 = time.time()
            emb = enc.encode(texts)
            if emb.ndim != 2 or len(emb) != len(texts) or not np.isfinite(emb).all():
                raise ValueError("encoder produced invalid document vectors")
            np.save(f, emb)
            print(f"[dense] encoded {len(texts)} in {time.time() - t0:.0f}s", flush=True)

    recs, t0 = [], time.time()
    for k, q in enumerate(queries, 1):
        if q["excluded"]:
            recs.append({"query_id": q["query_id"], "excluded": q["excluded"],
                         "table_id": q["table_id"], "mode": q["mode"]})
            continue
        # `--corpus gold` reproduces TableRAG's setting: `init_retriever(table_id,
        # df)` builds one index per table and searches only inside it. Masking the
        # candidate set is that, for the dense leg exactly (cosine is per-document);
        # the sparse leg keeps corpus-wide IDF, which alpha=1.0 removes entirely.
        sel = np.flatnonzero(unit_tid_arr == q["table_id"]) if a.corpus == "gold" else None
        s = bm.get_scores(_tokenize(q["question"]))
        if emb is not None:
            d = emb @ enc.encode_query([q["question"]])[0].astype(np.float32)
            if sel is not None:
                s, d = s[sel], d[sel]
            s = a.alpha * _minmax(d) + (1 - a.alpha) * _minmax(s) if a.alpha < 1 else d
        elif sel is not None:
            s = s[sel]
        order = np.argsort(-s, kind="stable")
        if sel is not None:
            order = sel[order]
        if a.unit == "rowcol":
            selected = rowcol_select(order, covers, texts, is_row,
                                     a.budget, a.dump_context, grid, a.template, a.row_text,
                                     cap=a.rowcol_max_pairs, context_format=a.rowcol_context,
                                     header_mode=a.rowcol_header)
        else:
            selected = budget_select(order, covers, texts,
                                     a.budget, a.dump_context, a.max_units)
        got, n_cells, ctx = selected
        gold = q["gold"]
        hit = (gold <= got) if q["mode"] == "all" else bool(gold & got)
        # gold 를 처음 배달한 단위가 순위 몇 번째인가. 판정에는 쓰지 않는다 --
        # 진단값이다(`CLAUDE.md` §0.1: 주지표는 질의 단위 정확도 하나). 예산 안에
        # 들어왔는데도 리더가 틀리는 몫이 이 순위와 붙어 있어서 따로 센다.
        grank = next((i for i, p2 in enumerate(order[:500], 1) if covers[p2] & gold),
                     None)
        r = {"query_id": q["query_id"], "table_id": q["table_id"],
             "mode": q["mode"], "m": len(gold), "correct": int(hit),
             "aggregation": q.get("aggregation"), "cells_in_context": n_cells,
             "gold_rank": grank,
             "gold_table_in_context": int(any(c[0] == q["table_id"] for c in got))}
        if a.dump_context:
            r["question"] = q["question"]
            r["answer"] = q["answer"]
            r.update(evidence_fields(selected, gold))
            validate_retrieval(r)
        recs.append(r)
        if k % 200 == 0:
            print(f"  {k}/{len(queries)}  {time.time() - t0:.0f}s", flush=True)

    scored = [r for r in recs if "correct" in r]
    excl = Counter(r["excluded"] for r in recs if "excluded" in r)
    by_mode = defaultdict(list)
    for r in scored:
        by_mode[r["mode"]].append(r["correct"])

    def acc(v):
        return round(sum(v) / len(v), 4) if v else None

    from rag_agent.data.loader import _find_data_root
    data_root = _find_data_root(a.data_dir)
    raw_dir, hmt_dir = hg._dirs(a.data_dir)
    table_files = {f"{kind}/{tid}": file_digest(directory / f"{tid}.json")
                   if (directory / f"{tid}.json").exists() else None
                   for tid in tids for kind, directory in (("raw", raw_dir), ("hmt", hmt_dir))}
    summary = {
        "context_version": 2 if a.dump_context else None,
        "provenance": provenance(ROOT), "arguments": vars(a),
        "comparison_scope": "retrieval_hint_adaptation" if a.unit == "tablerag" else "index_representation_adaptation",
        "dataset": {"split_sha256": file_digest(data_root / "data" / f"{a.split}_samples.jsonl"),
                    "tables_sha256": digest(table_files),
                    "page_titles_sha256": file_digest(PAGE_TITLES) if PAGE_TITLES.exists() else None},
        "corpus_text_sha256": digest(texts),
        "encoder_details": enc.metadata() if enc else None,
        "embedding_input_audit": input_audit,
        "chunk_measure": "tokens" if a.chunk_tokens else "characters",
        "chunk_size": size, "chunk_overlap": a.chunk_overlap if a.unit == "trag_hetero" else 0,
        "chunk_tokenizer": a.chunk_tokenizer or None,
        "chunk_tokenizer_revision": (getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
                                     or a.chunk_tokenizer_revision),
        "splitter": ("token_offsets" if tokenizer else "RecursiveCharacterTextSplitter")
                    if a.unit == "trag_hetero" else "row_pack" if a.unit == "chunk" else None,
        "budget_policy": "whole_unit_stop_after_distinct_cells",
        "cells_delivered_mean": sum(r["cells_in_context"] for r in scored) / max(len(scored), 1),
        "cells_over_budget": sum(r["cells_in_context"] > a.budget for r in scored),
        "split": a.split, "corpus": a.corpus, "n_tables": len(tids),
        "n_units": len(texts), "unit": a.unit, "template": a.template,
        # 단위를 정하는 인자도 적는다. 없으면 t_trag_hetero(1,000자)와
        # t_trag_hetero_tok(2,400자)처럼 같은 unit 인 두 행을 결과 파일만 보고
        # 구별할 수 없고, 어떤 명령이 이 파일을 만들었는지 복원되지 않는다.
        "chunk_chars": a.chunk_chars, "row_text": a.row_text,
        "tablerag_colmode": a.tablerag_colmode,
        "max_units": a.max_units,
        "rowcol_max_pairs": a.rowcol_max_pairs if a.unit == "rowcol" else None,
        "rowcol_context": a.rowcol_context if a.unit == "rowcol" else None,
        "rowcol_header": a.rowcol_header if a.unit == "rowcol" else None,
        "encoder": enc.name if enc else "none (bm25 only)", "alpha": a.alpha,
        "query_prefix": (enc.query_prefix if enc else ""), "budget_cells": a.budget,
        "n_queries_in_split": len(queries), "n_scored": len(scored),
        "n_excluded": sum(excl.values()), "excluded_by_reason": dict(excl),
        "accuracy_all_mode": acc(by_mode["all"]), "n_all_mode": len(by_mode["all"]),
        "accuracy_any_mode": acc(by_mode["any"]), "n_any_mode": len(by_mode["any"]),
        "accuracy_scored": acc([r["correct"] for r in scored]),
        "accuracy_over_full_split": round(
            sum(r["correct"] for r in scored) / len(queries), 4),
        "gold_table_in_context": round(
            sum(r["gold_table_in_context"] for r in scored) / max(len(scored), 1), 4),
    }
    # Retrieval summaries historically use tag.json, while reader outputs use
    # one shared stem. Keep the retrieval path convention and hash its records.
    records_path = out / f"{tag}_records.jsonl"
    write_pair(records_path, recs, summary, summary_path=out / f"{tag}.json")
    print(json.dumps(summary, indent=2))
    print(f"wrote -> {out / tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
