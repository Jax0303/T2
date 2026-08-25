# SPDX-License-Identifier: MIT
"""LLM header-tree reconstruction — TreeThinker's round 1, used at index time.

`header_grid.reconstruct_row_paths` reads the 2D grid and carries stub labels
down. That is blind to the one failure class that dominates HiTab: a parent row
that *also* carries its own values is indistinguishable from its children once
indentation is gone (STATUS-2026-08-10 §2: 388 tables, 0 of them fully
reconstructed). The distinction is semantic, not geometric — "Total" vs a
sibling line — so a rule over the grid cannot recover it in principle.

TreeThinker (Zhang et al., RealHiTBench, Findings of ACL 2025;
https://github.com/cspzyy/RealHiTBench) prompts an LLM to encode every header
cell as a tuple ``(R0, 1, 2, City)`` — axis+level, start line, end line, text —
and links them into a header tree. ``GENERATE_TREE`` below is their
``Generate_Tree`` prompt from ``inference/answer_prompt_tree.py``, verbatim.

The borrowing stops at the tree. They spend it per question, inside the prompt
that answers it; here it is spent once per table at index time, and what it
feeds is sentence generation + embedding.

A path is just the chain of tuples whose span covers a line, so this skips
building the tree object — ``reconstruct_row_paths_llm`` returns the same
``List[List[str]]`` as its rule-based counterpart and drops into the same call
sites.
"""
from __future__ import annotations

import re
from typing import List, Sequence

# Verbatim from RealHiTBench inference/answer_prompt_tree.py (Generate_Tree),
# minus its trailing "[TABLE]" placeholder — the table goes in the user turn.
GENERATE_TREE = """You are tasked with performing detailed table analysis. Your task is to generate a hierarchical tree structure for the top-row and left-column headers based on a LaTeX syntax complex table.
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

Let's get started!"""

# Their prompt never pins what "position" counts from, because their reader only
# has to be self-consistent. We score against gold grid lines, so it does.
COORD_NOTE = (
    "Rows are numbered 0..{n_rows} and columns 0..{n_cols}, top-to-bottom and "
    "left-to-right over the LaTeX table below, counting header rows. Use those "
    "numbers as the start/end positions.\n"
    "End your reply with the tuple list on one line, in the form:\n"
    "L=[(R0, 1, 3, Total), (R1, 1, 1, Male), ...]\n\n"
)

_TUPLE = re.compile(r"\(\s*([RC])\s*(-?\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([^)]*)\)")


def to_latex(texts: Sequence[Sequence[object]]) -> str:
    """Render a raw grid as a LaTeX tabular.

    HiTab's raw ``texts`` already blanks every merged cell but its origin, which
    is the same visual cue the prompt asks the model to read, so the blanks are
    carried through rather than expanded.
    """
    n_cols = max((len(r) for r in texts), default=0)
    lines = ["\\begin{tabular}{" + "l" * n_cols + "}"]
    for row in texts:
        cells = [str(row[c]).strip() if c < len(row) else "" for c in range(n_cols)]
        cells = [c.replace("\\", " ").replace("&", "\\&") for c in cells]
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def parse_tuples(reply: str, axis: str = "R"):
    """Pull ``(R0, 1, 2, City)`` nodes out of a reply -> [(level, start, end, text)].

    Takes the LAST occurrence of each (axis, level, start, end) key: the models
    restate the list after revising it, and the final statement is the revised
    one.
    """
    seen = {}
    for ax, lvl, start, end, text in _TUPLE.findall(reply):
        if ax != axis:
            continue
        s, e = int(start), int(end)
        if e < s:
            s, e = e, s
        seen[(int(lvl), s, e)] = text.strip().strip("'\"").strip()
    return [(lvl, s, e, t) for (lvl, s, e), t in seen.items() if t]


def paths_from_tuples(nodes, n_header_rows: int, n_rows: int) -> List[List[str]]:
    """One path per data line: the covering nodes, shallowest level first.

    A header tree's path to a leaf *is* the set of nodes whose span contains
    that line, so the tree never has to be materialised.
    """
    out: List[List[str]] = []
    for r in range(n_header_rows, n_rows):
        covering = sorted((n for n in nodes if n[1] <= r <= n[2]),
                          key=lambda n: (n[0], n[2] - n[1]))
        path: List[str] = []
        for _lvl, _s, _e, text in covering:
            if not path or path[-1].lower() != text.lower():
                path.append(text)
        out.append(path)
    return out


def reconstruct_row_paths_llm(grid, n_header_rows: int, n_header_cols: int,
                              llm, max_tokens: int = 4096) -> List[List[str]]:
    """Drop-in for ``reconstruct_row_paths``, backed by one LLM call per table.

    ``n_header_cols`` is unused — the model reads the stub block off the grid
    itself — but is kept so the two reconstructors share a signature.
    """
    latex = to_latex(grid)
    note = COORD_NOTE.format(n_rows=len(grid) - 1,
                             n_cols=max((len(r) for r in grid), default=1) - 1)
    reply = llm.complete(GENERATE_TREE, note + latex, max_tokens=max_tokens)
    return paths_from_tuples(parse_tuples(reply, "R"), n_header_rows, len(grid))


def _demo() -> None:
    reply = """
    Working through it: L=[(R0, 1, 4, Total), (R1, 1, 2, Male), (R2, 1, 1, Grade 1)]
    On review the second level was wrong, revised list:
    L=[(R0, 1, 4, Total), (R1, 1, 2, Male), (R2, 1, 1, Grade 1), (R2, 2, 2, Grade 2),
       (R1, 3, 4, Female), (C0, 1, 2, Count)]
    """
    nodes = parse_tuples(reply, "R")
    assert len(nodes) == 5, nodes
    assert (1, 1, 2, "Male") in nodes
    # revision wins over the first statement of the same key
    assert all(n[0] != 2 or n[3].startswith("Grade") for n in nodes)

    paths = paths_from_tuples(nodes, n_header_rows=1, n_rows=5)
    assert paths == [
        ["Total", "Male", "Grade 1"],
        ["Total", "Male", "Grade 2"],
        ["Total", "Female"],
        ["Total", "Female"],
    ], paths

    # a parent row that also carries values: it is its own leaf, not duplicated
    solo = paths_from_tuples([(0, 1, 3, "Total"), (1, 1, 1, "Total")], 1, 2)
    assert solo == [["Total"]], solo

    assert to_latex([["a", "b"], ["1", ""]]).splitlines()[1] == "a & b \\\\"
    print("ok")


if __name__ == "__main__":
    _demo()
