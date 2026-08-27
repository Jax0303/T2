"""A 429 on the free tier is routine, not a fault -- it must not end a run."""
from rag_agent.llm.groq_llm import GroqLLM, retry_after

MSG = ("Error code: 429 - {'error': {'message': 'Rate limit reached for model "
       "`openai/gpt-oss-120b` ... on tokens per minute (TPM): Limit 8000, Used "
       "6833, Requested 1658. Please try again in 3.6825s.'}}")


def test_honours_the_hint_and_pads_the_window():
    assert retry_after(MSG, attempt=0) == 4.6825


def test_milliseconds():
    assert retry_after("please try again in 500ms", attempt=0) == 1.5


def test_falls_back_to_exponential_without_a_hint():
    assert retry_after("429 too many requests", attempt=3) == 8.0


def test_waits_out_the_daily_window_not_just_the_minute_one():
    # TPD 429s ask for minutes; a 60s cap burned every retry inside one wait
    assert retry_after("Please try again in 10m29.424s", attempt=0) == 630.424
    assert round(retry_after("Please try again in 4m0.191999999s", attempt=0), 3) == 241.192


def test_gpt_oss_does_not_pay_for_medium_reasoning():
    # medium is gpt-oss's default and it spends the completion budget on hidden
    # reasoning, which on 200k tokens/day bought 30 of 175 queries
    oss = GroqLLM("openai/gpt-oss-120b", api_key="x")
    assert oss.reasoning_effort == "low"
    # and it is part of the reader identity, so guard_resume will not join
    # records written at a different effort
    assert oss.name == "groq:openai/gpt-oss-120b?reasoning_effort=low"

    plain = GroqLLM("llama-3.3-70b-versatile", api_key="x")
    assert plain.reasoning_effort is None
    assert plain.name == "groq:llama-3.3-70b-versatile"


def test_capped():
    assert retry_after("try again in 9000s", attempt=0) == 900.0
    assert retry_after("429", attempt=20) == 900.0


# --- the sleep loop itself: patience is wall-clock, not attempts ------------
# The free tier's binding limit is tokens-per-day (200k, ~139 tok/min refill),
# so the minutes-long hints are honest and must be ridden out. Patience is the
# same total as the old attempt count bought; it is just spent in <=60s hops so
# a run takes capacity as soon as it exists instead of at the end of a hint.
import logging
from types import SimpleNamespace

from rag_agent.llm import groq_llm

TPD = ("Error code: 429 - rate limit reached on tokens per day (TPD). "
       "Please try again in 10m29.424s.")


class _Stub:
    """Fails with a 429 the first ``fails`` times, then answers."""

    def __init__(self, fails, msg=TPD):
        self.fails, self.msg, self.calls = fails, msg, 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_kw):
        self.calls += 1
        if self.calls <= self.fails:
            raise RuntimeError(self.msg)
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content=" 42 "))])


logging.disable(logging.WARNING)   # the loop logs one line per hop


def _llm(fails, slept):
    """A GroqLLM whose sleeps are recorded and advance a fake clock.

    The clock has to move: the loop stops on wall-clock, so a no-op sleep spins
    forever -- which is exactly what the first draft of this test did.
    """
    clock = [0.0]

    def _sleep(secs):
        slept.append(secs)
        clock[0] += secs

    llm = groq_llm.GroqLLM.__new__(groq_llm.GroqLLM)   # no key, no network
    llm.model_name, llm.temperature, llm.retry_on_429 = "m", 0.0, 8
    # complete() reads this; __new__ skips __init__, so the fixture supplies it
    llm.reasoning_effort = None
    llm.client, llm.last_finish_reason = _Stub(fails), None
    groq_llm.time = SimpleNamespace(sleep=_sleep, monotonic=lambda: clock[0])
    return llm


def test_rides_out_a_ten_minute_window_in_short_hops():
    slept = []
    llm = _llm(11, slept)
    assert llm.complete("s", "u") == "42"
    assert max(slept) <= 60.0, slept          # never sleeps through capacity
    assert sum(slept) <= 900.0 * 8            # inside the patience budget
    assert len(slept) == 11                   # 8 attempts would have died here


def test_gives_up_when_the_patience_budget_is_spent():
    slept = []
    llm = _llm(10_000, slept)   # never recovers
    try:
        llm.complete("s", "u")
    except RuntimeError as e:
        assert "exhausted" in str(e)
    else:
        raise AssertionError("should have given up")
    assert 900.0 * 8 - 60 <= sum(slept) <= 900.0 * 8 + 1, sum(slept)


def test_a_non_429_error_is_not_retried():
    llm = _llm(1, [])
    llm.client.msg = "500 internal error"
    try:
        llm.complete("s", "u")
    except RuntimeError as e:
        assert "500" in str(e)
    else:
        raise AssertionError("should have raised")
    assert llm.client.calls == 1


if __name__ == "__main__":
    test_honours_the_hint_and_pads_the_window()
    test_milliseconds()
    test_falls_back_to_exponential_without_a_hint()
    test_waits_out_the_daily_window_not_just_the_minute_one()
    test_capped()
    test_gpt_oss_does_not_pay_for_medium_reasoning()
    test_rides_out_a_ten_minute_window_in_short_hops()
    test_gives_up_when_the_patience_budget_is_spent()
    test_a_non_429_error_is_not_retried()
    print("ok")
