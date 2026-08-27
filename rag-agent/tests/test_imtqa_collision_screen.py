"""The screen must read IM-TQA's grid the way deployment would -- not its labels.

If the reconstruction front-end is mis-wired the collision numbers still look
plausible (they are just rates), so the check has to be on the paths themselves.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from imtqa_collision_screen import corpus_of


def _table(vals, matrix, ttype="hierarchical"):
    return {"table_id": "T1", "table_type": ttype, "cell_ID_matrix": matrix,
            "chinese_cell_value_list": vals, "english_cell_value_list": vals}


def test_two_level_column_header_becomes_a_two_segment_path():
    # a merged "2019" over M/F: the flattened grid repeats it, as IM-TQA ships it
    vals = ["", "2019", "2019",
            "", "M", "F",
            "Seoul", "10", "20",
            "Busan", "30", "40"]
    C = corpus_of([_table(vals, [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])])
    assert [(rp, cp, v) for rp, cp, v in C.cell_paths] == [
        (["Seoul"], ["2019", "M"], "10"), (["Seoul"], ["2019", "F"], "20"),
        (["Busan"], ["2019", "M"], "30"), (["Busan"], ["2019", "F"], "40")]
    assert len(C.cell_owner) == 4          # header cells are not indexed as data


def test_a_flat_table_yields_one_segment_paths():
    # the by-table-type contrast is only meaningful if this stays flat
    vals = ["city", "pop", "area",
            "Seoul", "9", "605",
            "Busan", "3", "770",
            "Daegu", "2", "883"]
    C = corpus_of([_table(vals, [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
                          ttype="vertical")])
    assert all(len(cp) == 1 for _rp, cp, _v in C.cell_paths), C.cell_paths


def test_blank_cells_are_not_indexed():
    vals = ["", "2019", "2019",
            "", "M", "F",
            "Seoul", "10", "",
            "Busan", "", "40"]
    C = corpus_of([_table(vals, [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])])
    assert [v for _rp, _cp, v in C.cell_paths] == ["10", "40"]


if __name__ == "__main__":
    test_two_level_column_header_becomes_a_two_segment_path()
    test_a_flat_table_yields_one_segment_paths()
    test_blank_cells_are_not_indexed()
    print("ok")
