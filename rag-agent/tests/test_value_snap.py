"""One check per rule in rag_agent/serialization/value_snap.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_agent.serialization.value_snap import snap

CELLS = ["0.518", "53.0", "5756745", "categoria primera a", "6435825"]


def test_rescale_and_complement():
    assert snap("51.8", CELLS)[0] == "0.518"          # x100
    assert snap("47.0", CELLS)[0] == "53.0"           # 100-x
    assert snap("5756.745", CELLS)[0] == "5756745"    # /1000


def test_a_value_already_in_context_is_never_moved():
    # 53.0 is a cell; 0.53 would also "explain" it, but identity wins.
    assert snap("53.0", CELLS) == ("53.0", {"snapped": False, "reason": "already_a_cell_value"})


def test_ambiguity_and_misses_are_left_alone():
    assert snap("37.9", CELLS)[0] == "37.9"           # nothing explains it
    # 5.0 is explained by 0.05? no cell. But 500 <- 5.0 via x100 AND 0.005 via /1000
    two = ["0.05", "5000"]
    assert snap("5.0", two)[0] == "5.0"               # two explanations -> no move


def test_substring_rule_only_in_S3():
    assert snap("primera a", CELLS, "S3")[0] == "categoria primera a"
    assert snap("primera a", CELLS, "S2")[0] == "primera a"


def test_narrow_rule_sets_do_not_use_wide_transforms():
    assert snap("47.0", CELLS, "S1")[0] == "47.0"     # 100-x not in S1
    assert snap("51.8", CELLS, "S1")[0] == "0.518"    # x100 is


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
