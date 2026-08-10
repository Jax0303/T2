# SPDX-License-Identifier: MIT
"""RealHiTBench's strict scorer compares at the gold's own precision."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from realhitbench_answer_accuracy import em_norm, gold_is_numeric, token_f1


def test_float_printing_noise_passes():
    # one float written two ways
    assert em_norm(184.66899999999998, "184.67")
    assert em_norm(15.133333333333331, "15.13")


def test_a_dropped_decimal_is_wrong():
    # what the old 1e-5 relative tolerance let through: the bigger the number,
    # the more it forgave
    assert not em_norm(21091.0, "21091.09")
    assert not em_norm(505919.0, "505919.50")
    assert not em_norm(59969486.0, "59969486.20")


def test_percent_and_comma_normalisation_survive():
    assert em_norm("14.60%", "14.60%")
    assert em_norm("1,234.5", "1234.50")


def test_a_near_miss_cell_still_fails():
    assert not em_norm(1119760, "1119800")


def test_non_numeric_gold_falls_back_to_string_equality():
    assert em_norm("Ontario", "ontario")
    assert not em_norm("Quebec", "Ontario")


def test_ties_round_away_from_zero_like_the_golds_were():
    # round() is banker's: it takes 4.125 to 4.12 and marks a correct answer
    # wrong against a gold written "4.13".
    assert em_norm(4.125, "4.13")
    assert em_norm(2.675, "2.68")
    # and the tie rule must not start forgiving genuinely different numbers
    assert not em_norm(4.124, "4.13")


def test_gold_is_numeric_separates_the_unanswerable_stratum():
    assert gold_is_numeric("1,234.50")
    assert gold_is_numeric("14.60%")
    assert not gold_is_numeric("Japan")
    assert not gold_is_numeric("Decrease by 0.03.")
    assert not gold_is_numeric("2015-11-03, 2403")


def test_token_f1_gives_text_golds_partial_credit():
    assert token_f1("14.60", "14.60") == 1.0
    assert token_f1("Japan", "Italy") == 0.0
    # the case EM cannot score: right number, gold written as a sentence
    assert 0.0 < token_f1("-0.03", "Decrease by 0.03.") < 1.0
    # F1 must not reward a wrong number just for sharing a word
    assert token_f1("Increase by 5", "Decrease by 0.03.") < \
        token_f1("Decrease by 0.03", "Decrease by 0.03.")
