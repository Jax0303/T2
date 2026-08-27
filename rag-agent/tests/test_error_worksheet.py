"""The two classifiers the error worksheet adds, and nothing else.

Both decide what a wrong answer MEANS, so a silent change in either would
re-label an error taxonomy that gets quoted. The replay itself is checked at
run time by the drift assertion in ``error_worksheet.main``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from error_worksheet import error_class, pred_source


def test_sign_and_scale_are_notation_not_reading():
    assert error_class("-0.8", ["0.8"]) == "sign_only"
    assert error_class("0.8", ["-0.8"]) == "sign_only"
    assert error_class("81", ["0.81"]) == "scale_only"
    assert error_class("7500000", ["7.5"]) == "scale_only"


def test_a_different_number_is_a_different_number():
    assert error_class("136", ["122"]) == "different_value"
    assert error_class("81.5", ["81.0"]) == "within_2pct"


def test_missing_numbers_are_named_not_guessed():
    assert error_class("", ["1.0"]) == "pred_not_numeric"
    assert error_class("1.0", ["Ontario"]) == "gold_not_numeric"


def test_pred_source_separates_selection_from_invention():
    ctx = ["Ontario | 2016 : 44.2", "Quebec | 2016 : 31.7"]
    assert pred_source("31.7", ctx) == "value_present_in_context"
    assert pred_source("99.9", ctx) == "value_not_in_context"
    assert pred_source("no idea", ctx) == "no_number"


def test_classifier_assumes_the_row_already_failed():
    # abs-equality is what sign_only tests, so an EQUAL pair lands there too.
    # Harmless only because the worksheet calls this on rows the scorer already
    # rejected -- calling it on a passing row would inflate the notation bucket.
    assert error_class("0.8", ["0.8"]) == "sign_only"
