"""A 429 on the free tier is routine, not a fault -- it must not end a run."""
from rag_agent.llm.groq_llm import retry_after

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


def test_capped():
    assert retry_after("try again in 9000s", attempt=0) == 900.0
    assert retry_after("429", attempt=20) == 900.0


if __name__ == "__main__":
    test_honours_the_hint_and_pads_the_window()
    test_milliseconds()
    test_falls_back_to_exponential_without_a_hint()
    test_waits_out_the_daily_window_not_just_the_minute_one()
    test_capped()
    print("ok")
