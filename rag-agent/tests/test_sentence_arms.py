# SPDX-License-Identifier: MIT
"""The cell->sentence arms that isolate WHAT makes the sentence work.

Everything downstream (reconstruction, retriever, reader) is held fixed; only
the text a cell becomes changes. These check the three arms added for that
ladder: the floor, and the two placebos that a reviewer will ask for.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from operand_collision_multihiertt import (_stamp_alien_paths, _s3_sentence,
                                           cell_text)

CELL = {"row_path": ["seoul", "gangnam", "subtotal"],
        "col_path": ["2023", "q1"],
        "value": "57"}


def _cell(**over):
    return {**CELL, "alien_rp": ["osaka", "kita"], "alien_cp": ["2024"], **over}


def test_value_arm_drops_every_label():
    assert cell_text(_cell(), "value") == "57"


def test_flat_keeps_only_leaves_and_S2_keeps_the_whole_path():
    assert cell_text(_cell(), "flat") == "subtotal q1: 57"
    assert cell_text(_cell(), "S2") == "seoul > gangnam > subtotal > 2023 > q1: 57"


def test_alien_arm_renders_the_donor_path_not_its_own():
    got = cell_text(_cell(), "S3_alien")
    assert got == "For osaka > kita, 2024 is 57."
    # the value is the cell's own -- only the headers are alien
    assert "57" in got and "gangnam" not in got


def test_pad_arm_matches_S3_length_using_only_leaves():
    c = _cell()
    pad = cell_text(c, "S3_pad")
    # never shorter than the sentence it controls for
    assert len(pad.split()) >= len(cell_text(c, "S3").split())
    # and carries no ancestor the flat arm did not already have
    for ancestor in ("seoul", "gangnam", "2023"):
        assert ancestor not in pad
    assert "subtotal" in pad and "q1" in pad


def test_pad_arm_terminates_on_a_cell_with_no_headers():
    bare = {"row_path": [], "col_path": [], "value": "9", "alien_rp": [], "alien_cp": []}
    assert cell_text(bare, "S3_pad") == _s3_sentence([], [], "9")


def test_alien_paths_never_come_from_the_cells_own_table():
    cells = [{"table": ("doc", t), "row_path": [f"r{t}"], "col_path": [f"c{t}"],
              "value": str(t)}
             for t in range(5)]
    _stamp_alien_paths(cells)
    for cell in cells:
        assert cell["alien_rp"] != cell["row_path"]
    # deterministic: a second corpus built the same way stamps the same donors
    again = [dict(c) for c in cells]
    for c in again:
        del c["alien_rp"], c["alien_cp"]
    _stamp_alien_paths(again)
    assert [c["alien_rp"] for c in again] == [c["alien_rp"] for c in cells]


def test_single_table_corpus_does_not_crash():
    cells = [{"table": ("doc", 0), "row_path": ["a"], "col_path": ["b"], "value": "1"}]
    _stamp_alien_paths(cells)
    assert cells[0]["alien_rp"] == ["a"]
