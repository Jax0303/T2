# SPDX-License-Identifier: MIT
"""RealHiTBench's strict scorer compares at the gold's own precision."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from realhitbench_answer_accuracy import em_norm


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
