# SPDX-License-Identifier: MIT
"""hierarchy_edges: paths -> parent-child edge set (scripts/tree_reconstruct_hitab_raw.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from tree_reconstruct_hitab_raw import hierarchy_edges


def test_shared_ancestor_counted_once():
    # Two leaves under one parent = 3 edges, not 4: the (root, a) edge is shared.
    assert hierarchy_edges([["a", "b"], ["a", "c"]]) == {
        ("", "a"), ("a", "b"), ("a", "c")}


def test_normalisation_and_dropped_level():
    # Case/whitespace folded (same as path exact match), and a missing middle
    # level shows up as a wrong edge, not a missing one — which is why the
    # metric needs precision as well as recall.
    assert hierarchy_edges([[" A ", "B"]]) == {("", "a"), ("a", "b")}
    gold, rec = hierarchy_edges([["a", "b", "c"]]), hierarchy_edges([["a", "c"]])
    assert len(gold & rec) == 1 and len(rec) == 2 and len(gold) == 3
