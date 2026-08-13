# SPDX-License-Identifier: MIT
"""TreeThinker header reconstruction — RealHiTBench's prompt, used at index time.

RealHiTBench (Zhang et al., ACL 2025) ships TreeThinker as a *two-turn prompt*,
not an algorithm: turn 1 (``Generate_Tree``) hands an LLM the whole table and
asks for the header tuples plus a tree, turn 2 asks the question with that tree
still in the conversation. There is no parser in their repo — the tree is free
text consumed by the next LLM turn (``inference/inference_llm_tree.py:81-97``).

That shape cannot be dropped into a retrieval index, so this module keeps turn 1
and throws away turn 2:

    raw grid --render--> LaTeX --LLM(Generate_Tree)--> tuples --> (row_path, col_path)

The tuple format is the one their own prompt specifies — ``T(t1, t2, t3, t4)``
with ``t1`` in ``{R,C}{level}``, ``t2``/``t3`` the start and end line the header
spans, ``t4`` the label — so the parse is reading their stated contract, not
imposing ours. Output is ``List[List[str]]`` indexed by DATA line, the same shape
:func:`~.header_grid.reconstruct_col_paths` returns, so it drops into the scorer
and the serializer at the same seam.

Cost: one LLM call per table, at index time only, cached on disk by grid hash.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

Grid = List[List[str]]

# Verbatim from RealHiTBench inference/answer_prompt_tree.py (Generate_Tree),
# minus the trailing "[TABLE]" marker which we replace with the table itself.
GENERATE_TREE = """
You are tasked with performing detailed table analysis. Your task is to generate a hierarchical tree structure for the top-row and left-column headers based on a LaTeX syntax complex table.
(1). Task Description
[Reasoning Steps]
Your thought process is as follows:

- Understand the Table Structure: Provide a comprehensive description of the table, including the various levels of row and column headers and their corresponding meanings. Construct two distinct hierarchical trees: one for the row headers and one for the column headers. Each tree should accurately represent the levels and relationships of the headers.

- Traverse the Table: Analyze each row and column header to extract its content, indentation, and positions in the table. Identify merged cells and indentation, as they often indicate hierarchical relationships. Determine the parent-child relationships based on these visual cues and arrange the data under the correct parent node in both row and column header trees.

- Validate the Hierarchical Relationships: Iterate through both the row header tree and column header tree. Verify that the parent-child relationships are accurate and that the nodes are correctly placed within their respective hierarchies.

(2). Node Definition
You will be provided with a table in LaTeX format. The table may contain complex structures, such as merged or nested cells. Your task is to encode each node of table header as a tuple T(t1, t2, t3, t4)
The first element t1 indicates it represents row header (R) or column header (C), along with its corresponding level.
The second element t2 and third element t3 represent its start and end positions, while the fourth element t4 contains the value from the table. For example, a tuple (R0, 1, 2, City) indicates that it is a row header (R) at level 0, spanning from row 1 to row 2, with the value City.
Please Convert the table headers to list L=[T1, T2, ...]

(3). Tree Generate
1. Divide the tuples list L into groups based on their levels, such that all tuples with the same level are grouped together. Add a special ROOT node for rows and columns, each with a level of "-1".
2. For each tuple A in L. If the start and end positions of A are equal, mark A as a leaf node.
3. Otherwise, compare its T2 and T3 values with every closest higher-level and same flag tuple B. If tuple A is within the range of tuple B, then B is the parent-header of A.
4. Repeat steps 2 and 3 iteratively until all tuples in L are linked to their respective parent nodes (Tuples without parent node are linked to the ROOT node), forming a hierarchical Table-Header Tree H.

(4) Next, we will provide a table for you to analyze the hierarchical structure for the table and please organize the table header tuples as a tree, which can help you better understand the table structure.
Your should clearly and comprehensively understanding the content of the table, including the structure of the table, the meaning and formatting of each row and column header (Note: There is usually summative cell in the table, such as all, combine, total, sum, average, mean, etc. Please pay careful attention to the flag information in the row header and column header, this information can help you to skip many operations.)

Please check the constructed tree structure carefully and make sure that you have not missed any information in the contents of the table.

Let's get started!
"""

# The list L is what we parse. Ask for it last and alone so it survives a short
# completion budget — the prose reasoning the prompt invites is not needed here,
# and an answer truncated mid-prose loses the tuples entirely.
_TUPLE_INSTRUCTION = """

Output ONLY the list L, one tuple per line, in the exact form
(R0, 1, 2, City)
with no prose, no tree drawing, and no markdown fences. Emit every header tuple
for BOTH axes. Row positions are row indices of the table as given; column
positions are column indices.
"""

# (R0, 1, 2, City) — label may itself contain commas/parens, so take everything
# up to the final ")" on the line.
_TUPLE_RE = re.compile(
    r"\(\s*([RrCc])\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(.*?)\s*\)\s*$",
    re.MULTILINE,
)


def render_latex(grid: Grid, max_rows: int = 0,
                 merged_regions: Optional[Sequence[dict]] = None) -> str:
    """Render the raw occupancy grid as a LaTeX tabular.

    ``merged_regions`` matters more than it looks. TreeThinker's prompt tells the
    model to "identify merged cells and indentation, as they often indicate
    hierarchical relationships" — those two cues are the entire basis on which it
    is asked to infer parents. An occupancy grid has already expanded merges into
    blanks, and HiTab's export stripped indentation (0 of 1,500 raw tables retain
    leading whitespace), so rendering the bare grid hands the model a table with
    BOTH of its stated cues deleted. Passing the regions restores the first one as
    ``\\multicolumn`` / ``\\multirow`` spans.
    """
    rows = grid[:max_rows] if max_rows else grid
    n_cols = max((len(r) for r in rows), default=0)

    # (row, col) -> (rowspan, colspan) for anchors; covered cells are dropped.
    span: Dict[Tuple[int, int], Tuple[int, int]] = {}
    covered = set()
    for m in merged_regions or []:
        r0, r1 = int(m["first_row"]), int(m["last_row"])
        c0, c1 = int(m["first_column"]), int(m["last_column"])
        if r1 < r0 or c1 < c0:
            continue
        span[(r0, c0)] = (r1 - r0 + 1, c1 - c0 + 1)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if (r, c) != (r0, c0):
                    covered.add((r, c))

    def esc(x) -> str:
        return str(x).replace("&", r"\&").replace("%", r"\%").replace("\n", " ")

    body = []
    for r, row in enumerate(rows):
        cells = []
        for c in range(n_cols):
            if (r, c) in covered:
                continue
            val = esc(row[c]) if c < len(row) else ""
            rs, cs = span.get((r, c), (1, 1))
            if rs > 1:
                val = r"\multirow{%d}{*}{%s}" % (rs, val)
            if cs > 1:
                val = r"\multicolumn{%d}{c}{%s}" % (cs, val)
            cells.append(val)
        body.append(" & ".join(cells) + r" \\")
    return ("\\begin{tabular}{" + "l" * n_cols + "}\n"
            + "\n".join(body) + "\n\\end{tabular}")


def parse_tuples(text: str) -> List[Tuple[str, int, int, int, str]]:
    """Extract ``(axis, level, start, end, label)`` from a Generate_Tree reply."""
    out = []
    for axis, level, start, end, label in _TUPLE_RE.findall(text or ""):
        label = label.strip().strip('"').strip("'")
        if not label:
            continue
        s, e = int(start), int(end)
        out.append((axis.upper(), int(level), min(s, e), max(s, e), label))
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _paths_at(ts: Sequence[Tuple[str, int, int, int, str]], base: int,
              n_data_lines: int) -> List[List[str]]:
    paths: List[List[str]] = []
    for i in range(n_data_lines):
        line = i + base
        by_level: Dict[int, str] = {}
        for _, level, s, e, label in ts:
            if s <= line <= e and level not in by_level:
                by_level[level] = label
        paths.append([by_level[l] for l in sorted(by_level)])
    return paths


def calibrate_offset(ts: Sequence[Tuple[str, int, int, int, str]],
                     leaf_labels: Sequence[str], n_data_lines: int,
                     search: range = range(-2, 3)) -> Tuple[int, float]:
    """Pick the span origin by anchoring the model's leaves to the real grid.

    The prompt's example is 1-based, models answer 0-based about as often, and
    they disagree on whether the stub column counts as column 0 — three ways to
    be off by one, each of which silently shifts EVERY path by a line. Rather
    than guess the convention, try a small window and keep the shift whose
    deepest-level labels actually match the header text sitting on those grid
    lines. ``leaf_labels`` comes from the grid, never from gold, so this stays a
    reconstruction and not a peek at the answer.
    """
    best, best_score = 0, -1.0
    for base in search:
        paths = _paths_at(ts, base, n_data_lines)
        hit = sum(1 for p, leaf in zip(paths, leaf_labels)
                  if p and _norm(p[-1]) == _norm(leaf))
        score = hit / max(1, n_data_lines)
        if score > best_score:
            best, best_score = base, score
    return best, best_score


def paths_from_tuples(
    tuples: Sequence[Tuple[str, int, int, int, str]],
    axis: str,
    n_header_lines: int,
    n_data_lines: int,
    leaf_labels: Optional[Sequence[str]] = None,
) -> List[List[str]]:
    """Per-data-line header path, levels ascending.

    Without ``leaf_labels`` the spans are read as grid lines under the prompt's
    own 1-based convention; with them the origin is calibrated against the grid.
    """
    ts = [t for t in tuples if t[0] == axis.upper()]
    if not ts:
        return [[] for _ in range(n_data_lines)]
    if leaf_labels is not None:
        base, _ = calibrate_offset(ts, leaf_labels, n_data_lines)
        return _paths_at(ts, base, n_data_lines)
    return _paths_at(ts, n_header_lines + 1, n_data_lines)


class TreeThinker:
    """Turn-1 TreeThinker reconstruction with an on-disk response cache."""

    def __init__(self, llm, cache_path: Optional[str] = None,
                 max_tokens: int = 3000, max_grid_rows: int = 0):
        self.llm = llm
        self.max_tokens = max_tokens
        self.max_grid_rows = max_grid_rows
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: Dict[str, str] = {}
        self.n_calls = 0
        self.n_hits = 0
        if self.cache_path and self.cache_path.exists():
            with open(self.cache_path) as fh:
                for line in fh:
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec["response"]

    def _ask(self, latex: str) -> str:
        key = hashlib.sha1(
            (self.llm.name + "\n" + latex).encode("utf-8")
        ).hexdigest()
        if key in self.cache:
            self.n_hits += 1
            return self.cache[key]
        reply = self.llm.complete(
            "You are a precise table-structure parser.",
            GENERATE_TREE + "\n" + latex + "\n" + _TUPLE_INSTRUCTION,
            max_tokens=self.max_tokens,
        )
        self.n_calls += 1
        self.cache[key] = reply
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a") as fh:
                fh.write(json.dumps({"key": key, "response": reply}) + "\n")
        return reply

    def reconstruct(self, grid: Grid, n_header_rows: int, n_header_cols: int,
                    merged_regions: Optional[Sequence[dict]] = None
                    ) -> Tuple[List[List[str]], List[List[str]], dict]:
        """Return ``(col_paths, row_paths, trace)`` indexed by data line."""
        n_rows = len(grid)
        n_cols = max((len(r) for r in grid), default=0)
        n_data_rows = max(0, n_rows - n_header_rows)
        n_data_cols = max(0, n_cols - n_header_cols)

        def at(r: int, c: int) -> str:
            return grid[r][c] if 0 <= r < n_rows and 0 <= c < len(grid[r]) else ""

        # The last header line is where the leaf labels live — the anchor the
        # offset is calibrated against.
        col_leaves = [at(n_header_rows - 1, n_header_cols + i)
                      for i in range(n_data_cols)]
        row_leaves = [at(n_header_rows + i, n_header_cols - 1)
                      for i in range(n_data_rows)]

        latex = render_latex(grid, self.max_grid_rows, merged_regions)
        reply = self._ask(latex)
        ts = parse_tuples(reply)
        col_paths = paths_from_tuples(ts, "C", n_header_cols, n_data_cols, col_leaves)
        row_paths = paths_from_tuples(ts, "R", n_header_rows, n_data_rows, row_leaves)
        cbase, cscore = calibrate_offset([t for t in ts if t[0] == "C"],
                                         col_leaves, n_data_cols)
        rbase, rscore = calibrate_offset([t for t in ts if t[0] == "R"],
                                         row_leaves, n_data_rows)
        trace = {
            "n_tuples": len(ts),
            "n_tuples_C": sum(1 for t in ts if t[0] == "C"),
            "n_tuples_R": sum(1 for t in ts if t[0] == "R"),
            # A model that emits only (i, i) spans has produced a leaf list, not
            # a hierarchy — the single most useful signal that the tree failed.
            "degenerate_span_share": round(
                sum(1 for t in ts if t[2] == t[3]) / len(ts), 4) if ts else None,
            "max_level_C": max([t[1] for t in ts if t[0] == "C"], default=-1),
            "max_level_R": max([t[1] for t in ts if t[0] == "R"], default=-1),
            "col_offset": cbase, "col_leaf_anchor_rate": round(cscore, 4),
            "row_offset": rbase, "row_leaf_anchor_rate": round(rscore, 4),
            "empty_col_paths": sum(1 for p in col_paths if not p),
            "empty_row_paths": sum(1 for p in row_paths if not p),
            "reply_chars": len(reply or ""),
        }
        return col_paths, row_paths, trace


def demo() -> None:
    """Self-check on the prompt's own worked shape — no LLM involved."""
    reply = """
    (C0, 1, 2, 2023)
    (C1, 1, 1, Q1)
    (C1, 2, 2, Q2)
    (R0, 1, 2, Division)
    (R1, 1, 1, Domestic)
    (R1, 2, 2, Overseas)
    """
    ts = parse_tuples(reply)
    assert len(ts) == 6, ts
    assert ("C", 0, 1, 2, "2023") in ts

    # Uncalibrated: spans read as 1-based grid lines past the header band.
    cols = paths_from_tuples(ts, "C", n_header_lines=0, n_data_lines=2)
    assert cols == [["2023", "Q1"], ["2023", "Q2"]], cols
    rows = paths_from_tuples(ts, "R", n_header_lines=0, n_data_lines=2)
    assert rows == [["Division", "Domestic"], ["Division", "Overseas"]], rows

    # Calibration is what stops an off-by-one from shifting every path. The same
    # reply read against a grid whose leaves sit one line later must still land.
    shifted = parse_tuples("(C0, 2, 3, 2023)\n(C1, 2, 2, Q1)\n(C1, 3, 3, Q2)")
    assert paths_from_tuples(shifted, "C", 0, 2, ["Q1", "Q2"]) == \
        [["2023", "Q1"], ["2023", "Q2"]]
    base, score = calibrate_offset(shifted, ["Q1", "Q2"], 2)
    assert (base, score) == (2, 1.0), (base, score)

    # 0-based replies must not be shifted by the 1-based fallback.
    zero = parse_tuples("(C0, 0, 1, All)\n(C1, 0, 0, A)\n(C1, 1, 1, B)")
    assert paths_from_tuples(zero, "C", 0, 2, ["A", "B"]) == [["All", "A"], ["All", "B"]]

    # Labels containing a comma survive: the regex takes everything to the ")".
    comma = parse_tuples("(R0, 1, 1, Tokyo, Japan)")
    assert comma == [("R", 0, 1, 1, "Tokyo, Japan")], comma

    # A table the model said nothing about degrades to empty paths, not a crash.
    assert paths_from_tuples([], "C", 0, 3) == [[], [], []]

    g = [["", "2023", ""], ["City", "Q1", "Q2"], ["Seoul", "1", "2"]]
    assert r"\begin{tabular}{lll}" in render_latex(g)
    # Merge markup is the cue the prompt asks the model to read; it has to survive.
    lx = render_latex(g, merged_regions=[
        {"first_row": 0, "last_row": 0, "first_column": 1, "last_column": 2}])
    assert r"\multicolumn{2}{c}{2023}" in lx, lx
    assert lx.splitlines()[1].count("&") == 1, lx  # covered cell dropped, not blanked
    print("treethinker demo ok")


if __name__ == "__main__":
    demo()
