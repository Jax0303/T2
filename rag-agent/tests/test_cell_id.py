"""One check per non-trivial rule in rag_agent/serialization/cell_id.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_agent.serialization.cell_id import (value_from_sentence, render_cellid,
                                             parse_ids, predict)

TITLED = ("In the table 'crime rate in 2017, by area', among thunder bay, "
          "the value of violent crime > rate is 1,505.")
# A title and a header that both carry digits, and a value that carries a period.
TRAPS = ("In the table '2019 census, table 4.2', among 1990 > 15 to 24, "
         "the value of percent change 2016 to 2017 is 0.518.")
UNTITLED = "eastern ontario > french-language workers: 52.1"


def test_value_is_read_after_the_last_is():
    assert value_from_sentence(TITLED) == "1,505"
    assert value_from_sentence(TRAPS) == "0.518"     # not 2019, 4.2, 1990, 2016...
    assert value_from_sentence(UNTITLED) == "52.1"


def test_context_lines_are_preserved_verbatim():
    ctx = [TITLED, UNTITLED]
    assert render_cellid(ctx).split("\n") == [f"[#1] {TITLED}", f"[#2] {UNTITLED}"]


def test_parse_rules():
    assert parse_ids("[#7]", "hash") == [7]
    assert parse_ids("[7]", "hash") == []            # the v1 defect, kept reproducible
    assert parse_ids("[7]", "bracket") == [7]
    assert parse_ids("[#3], [5]", "bracket") == [3, 5]
    assert parse_ids("[#3], [3]", "bracket") == [3]  # deduped, order kept
    assert parse_ids("8,15", "bracket") == []
    assert parse_ids("8,15", "bracket_bare") == [8, 15]
    assert parse_ids("116150.0", "bracket_bare") == []   # a value, not a rank list


def test_predict_returns_stored_values_and_flags_breakage():
    ctx = [TITLED, UNTITLED]
    vm = {TITLED: "1,505", UNTITLED: "52.1"}
    assert predict("[#1]", ctx, vm, "bracket")[0] == "1,505"   # units/commas kept
    assert predict("[#2], [#1]", ctx, vm, "bracket")[0] == "52.1, 1,505"
    pred, audit = predict("[#9]", ctx, vm, "bracket")
    assert pred == "" and audit["failure"] == "out_of_range"
    pred, audit = predict("no cell answers this", ctx, vm, "bracket")
    assert pred == "" and audit["failure"] == "no_id"
    # A value the model typed itself is never used as the prediction.
    assert predict("1,505", ctx, vm, "bracket")[0] == ""


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
