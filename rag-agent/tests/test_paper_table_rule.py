"""The one rule that decides what "ours" is: state the title if there is one."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from paper_table_headtohead import TITLE_COVERAGE, TITLED, ours_series

base = {"q1": {"cell": {"answer_em": 0}}, "q2": {"cell": {"answer_em": 0}}}
s3 = {"q1": {"cell": {"answer_em": 1}}, "q2": {"cell": {"answer_em": 1}}}


def test_titled_corpus_uses_the_sentence_that_states_the_title():
    assert ours_series("HiTab", base, s3)[:2] == ([1, 1], "S3")


def test_untitled_corpus_never_pays_for_an_empty_title_clause():
    # even with the S3 run on disk and scoring higher, no titles means no S3
    for ds in ("MultiHiertt", "AIT-QA"):
        assert ours_series(ds, base, s3)[:2] == ([0, 0], "S2")


def test_the_rule_is_not_a_threshold_anyone_tuned():
    # 99.3% vs 0.0% -- any cut in the open interval gives the same table
    assert TITLE_COVERAGE["HiTab"] > TITLED > TITLE_COVERAGE["MultiHiertt"]
    assert all(c > 0.9 or c == 0.0 for c in TITLE_COVERAGE.values())


def test_missing_s3_run_falls_back_rather_than_crashing():
    assert ours_series("HiTab", base, None)[1] == "S2"


if __name__ == "__main__":
    test_titled_corpus_uses_the_sentence_that_states_the_title()
    test_untitled_corpus_never_pays_for_an_empty_title_clause()
    test_the_rule_is_not_a_threshold_anyone_tuned()
    test_missing_s3_run_falls_back_rather_than_crashing()
    print("ok")
