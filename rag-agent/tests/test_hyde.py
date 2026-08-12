# SPDX-License-Identifier: MIT
"""HyDE query-side arm (rag_agent/query/hyde.py) — caching and the empty-reply
fallback, which are the two places a silent wrong answer can enter."""
import json

from rag_agent.query.hyde import Hyde


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, user, max_tokens=256):
        self.calls.append(user)
        return self.replies.pop(0)


def test_generates_n_passages_per_question():
    llm = FakeLLM(["p1", "p2", "p3", "p4"])
    out = Hyde(llm, n=2).passages(["q one", "q two"])
    assert out == [["p1", "p2"], ["p3", "p4"]]
    assert len(llm.calls) == 4


def test_empty_reply_falls_back_to_the_question():
    # a blank probe would retrieve noise for that query while looking like a
    # normal run; the question is the safe degradation
    llm = FakeLLM(["   "])
    assert Hyde(llm).passages(["what is the total?"]) == [["what is the total?"]]


def test_cache_is_reused_across_instances(tmp_path):
    cache = tmp_path / "hyde.jsonl"
    first = FakeLLM(["generated"])
    assert Hyde(first, cache_path=str(cache)).passages(["q"]) == [["generated"]]
    assert len(first.calls) == 1

    second = FakeLLM([])           # no replies left: a cache miss would IndexError
    assert Hyde(second, cache_path=str(cache)).passages(["q"]) == [["generated"]]
    assert second.calls == []

    rec = json.loads(cache.read_text().splitlines()[0])
    assert rec["question"] == "q" and rec["passage"] == "generated"


def test_editing_the_prompt_invalidates_the_cache(tmp_path):
    import rag_agent.query.hyde as mod
    cache = tmp_path / "hyde.jsonl"
    Hyde(FakeLLM(["old"]), cache_path=str(cache)).passages(["q"])

    original = mod.INSTRUCTION
    mod.INSTRUCTION = original + " and one more rule"
    try:
        # must MISS and regenerate, not serve the passage from the old prompt
        assert Hyde(FakeLLM(["new"]), cache_path=str(cache)).passages(["q"]) == [["new"]]
    finally:
        mod.INSTRUCTION = original


def test_draws_are_cached_separately(tmp_path):
    cache = tmp_path / "hyde.jsonl"
    llm = FakeLLM(["draw0", "draw1"])
    assert Hyde(llm, cache_path=str(cache), n=2).passages(["q"]) == [["draw0", "draw1"]]
    # a rerun serves both draws from cache rather than collapsing them to one
    assert Hyde(FakeLLM([]), cache_path=str(cache), n=2).passages(["q"]) == [["draw0", "draw1"]]
