"""The row-baseline table decides a claim, so the sign and the pairing must hold.

delta is cell - row: a positive number has to mean the cell arm won, and the
flip counts have to be the discordant pairs, not the totals.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from row_baseline_table import corpus_of, paired


def _rec(m, row_em, cell_em):
    return {"m": m, "row": {"answer_em": row_em}, "cell": {"answer_em": cell_em}}


def test_delta_is_cell_minus_row_and_flips_are_discordant_only():
    recs = ([_rec(1, 0, 1)] * 3 +        # cell only
            [_rec(1, 1, 0)] * 1 +        # row only
            [_rec(1, 1, 1)] * 4 +        # both right, not a flip
            [_rec(1, 0, 0)] * 2)         # both wrong, not a flip
    d = paired(recs, "row", "cell")
    assert d["n"] == 10
    assert d["em_a"] == 0.5 and d["em_b"] == 0.7      # row .5, cell .7
    assert d["delta"] == 0.2                          # cell - row, positive = cell won
    assert (d["b_only"], d["a_only"]) == (3, 1)       # ties excluded
    assert 0.6 < d["p"] < 0.7                         # binomtest(3, 4) = .625


def test_a_tie_is_p_one_not_a_crash():
    d = paired([_rec(1, 1, 1), _rec(1, 0, 0)], "row", "cell")
    assert d["delta"] == 0.0 and d["p"] == 1.0


def test_operand_split_does_not_pool():
    recs = [_rec(1, 0, 1), _rec(1, 0, 1), _rec(3, 1, 0), _rec(3, 1, 0)]
    m1 = paired([r for r in recs if r["m"] == 1], "row", "cell")
    m2 = paired([r for r in recs if r["m"] >= 2], "row", "cell")
    assert m1["delta"] == 1.0 and m2["delta"] == -1.0   # opposite signs, not +-0
    assert paired(recs, "row", "cell")["delta"] == 0.0  # pooling would hide both


def test_corpus_name_covers_both_realhitbench_spellings():
    assert corpus_of("rhb_all") == "RealHiTBench"
    assert corpus_of("realhitbench , gold cells recovered") == "RealHiTBench"
    assert corpus_of("hitab_dev_lookup_all") == "HiTab"


if __name__ == "__main__":
    test_delta_is_cell_minus_row_and_flips_are_discordant_only()
    test_a_tie_is_p_one_not_a_crash()
    test_operand_split_does_not_pool()
    test_corpus_name_covers_both_realhitbench_spellings()
    print("ok")
