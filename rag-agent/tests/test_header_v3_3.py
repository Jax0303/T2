# SPDX-License-Identifier: MIT
"""Header rule v3.3 — v3.2 with S5n2 replaced by S5n3 (PREREG-2026-09-14-header-v3.md, 정정 4).

When refusing a number would make S5 reach past it for a higher heading, the tied group keeps its
v2 path. Synthetic tables only; v3.1 and v3.2 behaviour is pinned alongside.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from rag_agent.reconstruct.header_grid import reconstruct_row_paths  # noqa: E402

# the number sits between the second "Plans:" and "Section B", so skipping it reaches "Section B"
PAST_THE_NUMBER = [["", "2015"],
                   ["Section A", ""], ["Net", "1"],
                   ["Plans:", ""], ["Expense", "2"],
                   ["Section B", ""], ["Net", "3"],
                   ["", "544,823"], ["Other", "4"],
                   ["Plans:", ""], ["Expense", "5"]]
# the same headings, but "Section B" is nearer than the number
BEFORE_THE_NUMBER = [["", "2015"],
                     ["Section A", ""], ["Net", "1"],
                     ["Plans:", ""], ["Expense", "2"],
                     ["", "544,823"], ["Other", "4"],
                     ["Section B", ""], ["Net", "3"],
                     ["Plans:", ""], ["Expense", "5"]]


def test_a_group_that_would_take_a_heading_from_above_a_skipped_number_keeps_the_v2_path():
    g = PAST_THE_NUMBER
    v2 = reconstruct_row_paths(g, 1, 1)
    assert v2[3] == v2[9] == ["Plans:", "Expense"]
    assert reconstruct_row_paths(g, 1, 1, disambiguate="guarded")[9] == ["544,823", "Plans:", "Expense"]   # v3.1
    v32 = reconstruct_row_paths(g, 1, 1, disambiguate="no_numbers")
    assert v32[3] == ["Section A", "Plans:", "Expense"] and v32[9] == ["Section B", "Plans:", "Expense"]
    v33 = reconstruct_row_paths(g, 1, 1, disambiguate="no_numbers_guard")
    assert v33 == v2                                      # every member of the group, not only row 9


def test_a_split_decided_before_reaching_the_number_is_kept():
    g = BEFORE_THE_NUMBER
    v33 = reconstruct_row_paths(g, 1, 1, disambiguate="no_numbers_guard")
    assert v33 == reconstruct_row_paths(g, 1, 1, disambiguate="no_numbers")
    assert v33[9] == ["Section B", "Plans:", "Expense"]


def test_build_tables_v3_3_is_v3_2_with_s5n3():
    from mh_arms import V32_RULES, V33_RULES, build_tables
    assert [r for r in V33_RULES if r not in V32_RULES] == ["S5n3"] and len(V33_RULES) == len(V32_RULES)
    html = "<table>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
                               for row in PAST_THE_NUMBER) + "</table>"
    docs = {"u": ([html], "{}", [])}
    t32 = build_tables(docs, "v3.2")[0]["u::0"].table
    t33 = build_tables(docs, "v3.3")[0]["u::0"].table
    assert (t32.nhr, t32.nhc) == (t33.nhr, t33.nhc) == (1, 1)
    assert t32.row_path(9) == ["Section B", "Plans:", "Expense"]
    assert t33.row_path(9) == ["Plans:", "Expense"]
