# SPDX-License-Identifier: MIT
"""Asymmetric-retrieval prefixes (rag_agent/retrieve/encoders.py).

The failure this guards is silent: a wrong or missing prefix costs accuracy on
every arm at once and never raises, so the only thing that catches it is a test
pinning which family gets what.
"""
from rag_agent.retrieve.encoders import default_prefixes


def test_bge_v15_instructs_the_query_and_leaves_the_passage_bare():
    q, p = default_prefixes("BAAI/bge-small-en-v1.5")
    assert q == "Represent this sentence for searching relevant passages: "
    assert p == ""
    assert default_prefixes("BAAI/bge-large-en-v1.5") == (q, p)


def test_e5_marks_both_sides():
    assert default_prefixes("intfloat/e5-base-v2") == ("query: ", "passage: ")


def test_bge_m3_gets_nothing_despite_matching_bge():
    # trained without an instruction — the substring match on "bge-" must not
    # capture it, which is the whole reason for the special case
    assert default_prefixes("BAAI/bge-m3") == ("", "")


def test_unknown_and_empty_models_get_no_prefix():
    assert default_prefixes("sentence-transformers/all-MiniLM-L6-v2") == ("", "")
    assert default_prefixes("hashing") == ("", "")
    assert default_prefixes("") == ("", "")
    assert default_prefixes(None) == ("", "")
