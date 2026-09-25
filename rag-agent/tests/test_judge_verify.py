import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from judge_verify import has, norm, pieces, verify, verify_words, words  # noqa: E402


def test_pieces_and_boundaries():
    assert pieces("Bank of America, N.A., 2016 ,") == ["bank of america", "n.a.", "2016"]
    assert has(norm("H&amp;R Block, Inc. in 2016"), "h&r block") and not has("20161", "2016")


def test_verify_unique_gold_only():
    texts = {"a": "citigroup inc | 2016 | 2017", "b": "citigroup inc | 2015", "c": "2016 | 2017"}
    assert verify(pieces("Citigroup Inc, 2016"), texts, "a") == (1, True, "예")
    assert verify(pieces("2016"), texts, "a") == (2, True, "아니오")
    assert verify(pieces("Citigroup Inc, 2015"), texts, "a") == (1, False, "아니오")


def test_semicolon_and_words():
    assert pieces("Citigroup Inc; 2016; 2017") == ["citigroup inc", "2016", "2017"]
    ws = words(pieces("the dealer performance guarantees of Caterpillar; 2016"))
    assert ws == ["dealer", "performance", "guarantees", "caterpillar", "2016"]
    tokens = {"a": {"caterpillar", "dealer", "performance", "guarantees", "2016"}, "b": {"caterpillar", "2016"}}
    assert verify_words(ws, tokens, "a") == (1, True, "예")
