# SPDX-License-Identifier: MIT
"""Self-check for scripts/query_reformulation_eval.py's non-LLM, non-encoder logic."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from query_reformulation_eval import mcnemar, pilot_subset, reformulate_all  # noqa: E402


def test_pilot_subset_takes_first_n_sorted_by_query_id():
    pop = {"b": 2, "a": 1, "c": 3}
    assert pilot_subset(pop, 2) == [("a", 1), ("b", 2)]
    assert pilot_subset(pop, 0) == [("a", 1), ("b", 2), ("c", 3)]


def test_mcnemar_counts_discordant_pairs_and_ignores_concordant_ones():
    a = {"q1": True, "q2": False, "q3": True, "q4": False}
    b = {"q1": True, "q2": True, "q3": False, "q4": False}
    out = mcnemar(a, b, "test")
    assert out["a_only"] == 1     # q3: a right, b wrong
    assert out["b_only"] == 1     # q2: a wrong, b right
    assert out["both"] == 1       # q1
    assert out["discordant"] == 2
    assert out["p_value"] == 1.0  # 1 vs 1 discordant: no evidence either way


class _FakeLLM:
    def __init__(self, replies):
        self.replies = replies
        self.calls = 0

    def complete(self, system, user, max_tokens=80):
        self.calls += 1
        return self.replies.get(user, "")


def test_reformulate_all_falls_back_to_question_on_empty_completion(tmp_path):
    llm = _FakeLLM({"q?": ""})
    ordered = [("qid1", {"question": "q?"})]
    cache = tmp_path / "cache.json"
    out = reformulate_all(llm, ordered, cache)
    assert out["qid1"] == "q?"          # empty LLM output falls back to the raw question
    assert llm.calls == 1
    assert json.loads(cache.read_text())["qid1"] == "q?"


def test_reformulate_all_reuses_cache_without_recalling_the_llm(tmp_path):
    llm = _FakeLLM({"q?": "For X, Y."})
    ordered = [("qid1", {"question": "q?"})]
    cache = tmp_path / "cache.json"
    reformulate_all(llm, ordered, cache)
    assert llm.calls == 1
    reformulate_all(llm, ordered, cache)  # second run: same cache file, same query_id
    assert llm.calls == 1
