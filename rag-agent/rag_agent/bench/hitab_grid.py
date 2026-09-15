# SPDX-License-Identifier: MIT
"""HiTab addressing: one deterministic map from the raw grid to the data matrix.

HiTab ships every table twice. ``data/tables/raw/{id}.json`` is the spreadsheet
as exported — a ``texts`` grid with the header block still in it, plus header
trees whose nodes carry ``(row_index, column_index)`` INTO that grid.
``data/tables/hmt/{id}.json`` is the parsed form — a header-free ``data``
matrix, plus the same trees with a ``line_idx`` naming the data row/column each
node owns. Every gold annotation (``linked_cells``, ``reference_cells_map``) is
in RAW coordinates; the index unit is a data cell. Something has to join them.

What this module replaces (measured on the test split, 1,584 questions / 538
tables):

* ``align()`` in ``scripts/tree_reconstruct_hitab_raw.py`` guessed the join by
  matching cell VALUES and refused any table under a 0.90 match rate. That
  dropped 124 of 538 tables — never indexed, their 314 questions deleted from
  the population.
* ``_coord_offset`` in ``rag_agent/bench/hitab.py`` searched a 6x6 grid of
  (header_rows, header_cols) offsets for one that made the gold values line up.
* ``_coords_of`` read gold from ``quantity_link`` only, and only where the value
  parsed as a number — so the 351 questions whose answer is a label rather than
  a number resolved to no gold cell at all and were dropped.

None of the guessing is necessary. The two trees are the same tree, so walking
them together maps ``row_index -> line_idx`` exactly; and ``answer_formulas``
plus ``reference_cells_map`` name the answer cells outright.

Verified on the test split (``tests/test_hitab_grid.py``, and the counts in
``analysis/pipeline_audit.py``): the walk maps raw rows for 538/538 tables and
raw columns for 505/538 -- the other 33 lose a column to a merged header cell,
which :meth:`HitabTable._fill_gaps` recovers by pairing the data band in order
where that agrees with the walk. 1,681 of the 1,682 mappable ``quantity_link``
entries land on a data cell holding the annotated value, and 995 of 996
single-reference gold cells hold the answer HiTab publishes. Scoreable
questions go from 987/1,584 under the old rules to 1,581/1,584.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..stores.original_store import OriginalTable, _to_float, build_original_table

# `answer_formulas` are Excel refs over the raw grid with the sheet's own 1-based
# row numbering and a 3-row preamble. `reference_cells_map` states the mapping
# per query; on the test split all 1,882 entries agree on (row-3, col+0), so the
# constant is used where the map is silent and the map wins where it is not.
REF_ROW_OFFSET = 3

_REF = re.compile(r"\$?([A-Z]{1,2})\$?(\d+)")
_RANGE = re.compile(r"\$?([A-Z]{1,2})\$?(\d+):\$?([A-Z]{1,2})\$?(\d+)")
_COORD = re.compile(r"\d+")


def norm_value(v) -> str:
    """Compare two renderings of one cell. Numbers by value, text by fold."""
    f = _to_float(v)
    if f is not None:
        return f"{f:g}"
    return " ".join(str(v).strip().lower().split())


def col_index(letters: str) -> int:
    n = 0
    for ch in letters.upper():
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def parse_coord(coord) -> Optional[Tuple[int, int]]:
    """``"(17, 1)"`` -> ``(17, 1)``, raw-grid (row, col)."""
    nums = _COORD.findall(str(coord))
    return (int(nums[0]), int(nums[1])) if len(nums) >= 2 else None


# --------------------------------------------------------------------------
# raw grid <-> data matrix
# --------------------------------------------------------------------------

def _children(node: dict) -> List[dict]:
    ch = node.get("children_dict")
    if ch is None:
        ch = node.get("children") or []
    if isinstance(ch, dict):
        ch = list(ch.values())
    return [c for c in ch if isinstance(c, dict)]


def _raw_text(texts, node: dict) -> str:
    r, c = node.get("row_index", -1), node.get("column_index", -1)
    if r >= 0 and c >= 0 and r < len(texts) and c < len(texts[r]):
        return norm_value(texts[r][c])
    return ""


def axis_map(hmt_root: dict, raw_root: dict, texts, axis: str) -> Dict[int, int]:
    """``{raw line -> data line}`` for one axis, by walking both trees together.

    The trees have the same shape wherever the raw export is clean. Where a
    merged header cell makes the raw tree drop a child the parsed tree keeps,
    positional pairing would silently shift every sibling after it, so children
    are paired by header TEXT with a monotone pointer and an unmatched parsed
    child simply goes unmapped rather than stealing its neighbour's line.
    """
    out: Dict[int, int] = {}
    key = "column_index" if axis == "top" else "row_index"

    def walk(hn: dict, rn: dict) -> None:
        li = hn.get("line_idx")
        line = rn.get(key, -1)
        if li is not None and line >= 0:
            out[int(line)] = int(li)
        hc, rc = _children(hn), _children(rn)
        if len(hc) == len(rc):
            pairs = zip(hc, rc)
        else:
            pairs, k = [], 0
            for h in hc:
                want = norm_value(h.get("value") or h.get("name") or "")
                j = k
                while j < len(rc) and _raw_text(texts, rc[j]) != want:
                    j += 1
                if j < len(rc):
                    pairs.append((h, rc[j]))
                    k = j + 1
        for h, r in pairs:
            walk(h, r)

    walk(hmt_root, raw_root)
    return out


class HitabTable:
    """One HiTab table, addressable in both coordinate systems."""

    __slots__ = ("table_id", "raw", "hmt", "table", "row_map", "col_map", "title")

    def __init__(self, table_id: str, raw: dict, hmt: dict):
        self.table_id = table_id
        self.raw, self.hmt = raw, hmt
        hmt = dict(hmt, table_id=table_id)
        self.table: OriginalTable = build_original_table(hmt)
        texts = raw.get("texts") or []
        self.row_map = axis_map(self.hmt["left_root"], raw["left_root"], texts, "left")
        self.col_map = axis_map(self.hmt["top_root"], raw["top_root"], texts, "top")
        # A header cell merged across two columns appears once in the raw tree
        # and twice in the parsed one, so the walk leaves the covered column
        # unmapped -- and a gold cell in it then looks unresolvable (32 of the
        # test split's 1,584 questions). Both axes are monotone, so when the
        # data band has exactly as many raw lines as the parsed table has
        # lines, order pairs them; only then, and only for what the walk missed.
        self._fill_gaps(texts, raw)
        self.title = str(raw.get("title") or self.table.title or "").strip()

    def _fill_gaps(self, texts, raw) -> None:
        H = int(raw.get("top_header_rows_num") or 0)
        W = int(raw.get("left_header_columns_num") or 0)
        n_raw_rows = len(texts)
        n_raw_cols = max((len(r) for r in texts), default=0)
        for m, lo, hi, n_lines in ((self.row_map, H, n_raw_rows, self.table.n_rows),
                                   (self.col_map, W, n_raw_cols, self.table.n_cols)):
            if len(m) == n_lines:
                continue
            band = list(range(lo, hi))
            if len(band) != n_lines:
                continue                      # not a clean 1-1 band; leave it unmapped
            paired = dict(zip(band, range(n_lines)))
            if all(m[k] == v for k, v in paired.items() if k in m):
                m.update(paired)              # only where it AGREES with the walk

    def to_data(self, rc: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Raw-grid (row, col) -> data-matrix (i, j); ``None`` for a header cell."""
        i, j = self.row_map.get(rc[0]), self.col_map.get(rc[1])
        return None if i is None or j is None else (i, j)


@lru_cache(maxsize=None)
def _dirs(data_dir: str) -> Tuple[Path, Path]:
    base = Path(data_dir) / "data" / "tables"
    return base / "raw", base / "hmt"


def table_ids(data_dir: str = "data/hitab") -> List[str]:
    """Every table in the store — the corpus a corpus-wide retriever searches."""
    raw_dir, hmt_dir = _dirs(data_dir)
    return sorted(p.stem for p in hmt_dir.glob("*.json")
                  if (raw_dir / p.name).exists())


def load_table(table_id: str, data_dir: str = "data/hitab") -> Optional[HitabTable]:
    raw_dir, hmt_dir = _dirs(data_dir)
    fr, fh = raw_dir / f"{table_id}.json", hmt_dir / f"{table_id}.json"
    if not (fr.exists() and fh.exists()):
        return None
    return HitabTable(table_id, json.loads(fr.read_text()), json.loads(fh.read_text()))


# --------------------------------------------------------------------------
# gold cells
# --------------------------------------------------------------------------

def formula_refs(sample: dict) -> Tuple[List[Tuple[int, int]], Optional[str]]:
    """Raw-grid coords the query's ``answer_formulas`` reference.

    Returns ``(coords, error)``. ``error`` is set when a formula cannot be read,
    which is a label defect and has to be reported, not silently swallowed.
    """
    rmap = {k.upper().replace("$", ""): parse_coord(v)
            for k, v in (sample.get("reference_cells_map") or {}).items()}

    def rc(letters: str, row: str) -> Tuple[int, int]:
        hit = rmap.get(f"{letters.upper()}{row}")
        return hit if hit else (int(row) - REF_ROW_OFFSET, col_index(letters))

    coords: List[Tuple[int, int]] = []
    for f in sample.get("answer_formulas") or []:
        expr = re.sub(r'&"[^"]*"', "", str(f)).strip().lstrip("=")
        seen = expr
        for m in _RANGE.finditer(expr):
            r1, c1 = rc(m.group(1), m.group(2))
            r2, c2 = rc(m.group(3), m.group(4))
            coords += [(r, c) for r in range(min(r1, r2), max(r1, r2) + 1)
                       for c in range(min(c1, c2), max(c1, c2) + 1)]
            seen = seen.replace(m.group(0), " ")
        for m in _REF.finditer(seen):
            coords.append(rc(m.group(1), m.group(2)))
        left = _REF.sub(" ", _RANGE.sub(" ", expr))
        if re.search(r"[A-Za-z]", left.replace("SUM", " ").replace("AVERAGE", " ")
                     .replace("COUNTA", " ").replace("COUNT", " ")
                     .replace("MAX", " ").replace("MIN", " ").replace("ABS", " ")):
            return coords, f"unreadable_formula:{f}"
    return coords, None if coords else "no_formula"


def linked_refs(sample: dict) -> List[Tuple[int, int]]:
    """Raw-grid coords from ``quantity_link`` — every entry, numeric or not."""
    ql = (sample.get("linked_cells") or {}).get("quantity_link") or {}
    out = []
    for bucket in ql.values():
        if isinstance(bucket, dict):
            out += [c for c in (parse_coord(k) for k in bucket) if c]
    return out


def gold_cells(sample: dict, tab: HitabTable) -> Tuple[set, Optional[str]]:
    """``({(table_id, i, j), ...}, exclusion_reason)`` for one query.

    Gold is what the dataset says the answer is read from: the cells
    ``answer_formulas`` references, mapped through ``reference_cells_map``.
    ``quantity_link`` fills in where a query ships no formula. A query whose
    gold does not land on an indexable data cell is EXCLUDED with a reason
    rather than scored on the part of its gold that happened to resolve —
    dropping the unresolvable half of a gold set silently inflates every
    all-cells-retrieved metric computed from it.
    """
    coords, err = formula_refs(sample)
    if not coords:
        coords = linked_refs(sample)
        err = None if coords else "no_gold_annotation"
    if err:
        return set(), err
    cells, header = set(), 0
    for rc in coords:
        d = tab.to_data(rc)
        if d is None:
            header += 1
        else:
            cells.add((tab.table_id, d[0], d[1]))
    if header:
        return cells, "gold_is_header_cell"
    if not cells:
        return set(), "gold_unmappable"
    if any(not str(tab.table.data[i][j]).strip() for _t, i, j in cells):
        return cells, "gold_cell_is_empty"
    return cells, None


# --------------------------------------------------------------------------
# header-answer queries
# --------------------------------------------------------------------------

def _find_node(root: dict, raw_root: dict, texts, rc: Tuple[int, int], axis: str):
    """The (parsed, raw) node pair sitting at raw coordinate ``rc``, if any."""
    key = "column_index" if axis == "top" else "row_index"
    found = []

    def walk(hn, rn):
        if (rn.get("row_index", -1), rn.get("column_index", -1)) == rc:
            found.append((hn, rn))
        hc, rc_ = _children(hn), _children(rn)
        if len(hc) == len(rc_):
            pairs = list(zip(hc, rc_))
        else:
            pairs, k = [], 0
            for h in hc:
                want = norm_value(h.get("value") or h.get("name") or "")
                j = k
                while j < len(rc_) and _raw_text(texts, rc_[j]) != want:
                    j += 1
                if j < len(rc_):
                    pairs.append((h, rc_[j]))
                    k = j + 1
        for h, r in pairs:
            walk(h, r)

    walk(root, raw_root)
    _ = key
    return found[0] if found else None


def header_scope(tab: HitabTable, rc: Tuple[int, int]) -> set:
    """Data cells the header cell at raw ``rc`` labels.

    HiTab annotates "which region had the higher rate" with the ANSWER HEADER
    (``=A7``), never with the numbers being compared — so for those queries the
    thing to retrieve is not a cell but a header, and a cell index holds no such
    unit. A header is nevertheless present in every cell sentence of the row or
    column it governs, so the query is served the moment ANY of those cells is
    retrieved. That is the scope this returns.
    """
    texts = tab.raw.get("texts") or []
    out = set()
    for axis, root_key in (("left", "left_root"), ("top", "top_root")):
        hit = _find_node(tab.hmt[root_key], tab.raw[root_key], texts, rc, axis)
        if not hit:
            continue
        lines = []

        def collect(n):
            if n.get("line_idx") is not None:
                lines.append(int(n["line_idx"]))
            for c in _children(n):
                collect(c)

        collect(hit[0])
        t = tab.table
        for li in lines:
            if axis == "left" and 0 <= li < t.n_rows:
                out |= {(tab.table_id, li, j) for j in range(t.n_cols)
                        if str(t.data[li][j]).strip()}
            elif axis == "top" and 0 <= li < t.n_cols:
                out |= {(tab.table_id, i, li) for i in range(t.n_rows)
                        if str(t.data[i][li]).strip()}
    return out


def gold_target(sample: dict, tab: HitabTable) -> Tuple[set, str, Optional[str]]:
    """``(cells, mode, exclusion_reason)`` — what retrieval has to bring back.

    ``mode="all"``  every cell must be retrieved (the answer is read off them).
    ``mode="any"``  one cell of the scope suffices (the answer IS a header, and
                    every cell under it carries that header in its sentence).
    """
    coords, err = formula_refs(sample)
    if not coords:
        coords = linked_refs(sample)
        err = None if coords else "no_gold_annotation"
    if err and err != "no_formula":
        return set(), "all", err
    data, headers = set(), []
    for rc in coords:
        d = tab.to_data(rc)
        (data.add((tab.table_id, d[0], d[1])) if d else headers.append(rc))
    if data:
        if any(not str(tab.table.data[i][j]).strip() for _t, i, j in data):
            return data, "all", "gold_cell_is_empty"
        return data, "all", None
    scope = set()
    for rc in headers:
        scope |= header_scope(tab, rc)
    if not scope:
        return set(), "any", "gold_unmappable"
    return scope, "any", None


def answer_gold(sample: dict, tab: HitabTable):
    """``(gold, carriers, source, exclusion_reason, anomaly)`` — Strict Recall 의 gold.

    ``linked_cells`` 의 ``[ANSWER]`` 좌표가 먼저다 (quantity_link 는 데이터 셀, entity_link 는
    헤더 셀). 좌표가 없거나 읽히지 않을 때만 ``answer_formulas`` + ``reference_cells_map``.
    두 집합을 합치지 않는다 — 공식 참조가 ``[ANSWER]`` 와 다르면 ``anomaly`` 로 돌려준다.

    데이터 셀은 ``(table_id, i, j)``, 헤더 셀은 ``(table_id, "header", r, c)`` (원 격자 좌표).
    ``carriers[헤더]`` 는 그 헤더를 문장 경로에 싣는 데이터 셀(:func:`header_scope`)이다.
    """
    lc = sample.get("linked_cells") or {}
    buckets = [(lc.get("quantity_link") or {}).get("[ANSWER]"),
               *(v.get("[ANSWER]") for v in (lc.get("entity_link") or {}).values()
                 if isinstance(v, dict))]
    coords = [parse_coord(k) for b in buckets if isinstance(b, dict) for k in b]
    refs, err = formula_refs(sample)
    source, anomaly = "[ANSWER]", None
    if not coords or None in coords:
        source, coords = "answer_formulas", refs
        if err or not coords:
            return set(), {}, source, err or "no_gold_annotation", None
    elif set(refs) != set(coords):
        anomaly = {"answer_cells": sorted(set(coords)), "formula_cells": sorted(set(refs)),
                   "formula_error": err, "answer_formulas": sample.get("answer_formulas")}
    gold, carriers = set(), {}
    for rc in dict.fromkeys(coords):
        d = tab.to_data(rc)
        if d is not None:
            if not str(tab.table.data[d[0]][d[1]]).strip():
                return set(), {}, source, "gold_cell_is_empty", anomaly
            gold.add((tab.table_id, *d))
            continue
        scope = header_scope(tab, rc)
        if not scope:
            return set(), {}, source, "gold_unmappable", anomaly
        h = (tab.table_id, "header", *rc)
        gold.add(h)
        carriers[h] = frozenset(scope)
    return gold, carriers, source, None, anomaly
