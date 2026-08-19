"""Resuming a killed hours-long run: what counts as already done."""
import json


def replay(lines, arms):
    """The reader half of corpus_dump_vs_cell's resume, kept in one place so
    the rule -- stop at the first line that is truncated or missing an arm --
    is testable without a GPU and a 3-hour API run."""
    recs = []
    for line in lines:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            break
        if not all(a in r for a in arms):
            break
        recs.append(r)
    return recs, {r["query_id"] for r in recs}


def line(qid, *arms):
    return json.dumps({"query_id": qid, **{a: {"answer_em": 1} for a in arms}})


def test_takes_the_complete_prefix():
    recs, done = replay([line("a", "cell", "dump"), line("b", "cell", "dump")],
                        ("cell", "dump"))
    assert done == {"a", "b"} and len(recs) == 2


def test_drops_a_half_written_last_line():
    lines = [line("a", "cell", "dump"), '{"query_id": "b", "ce']
    assert replay(lines, ("cell", "dump"))[1] == {"a"}


def test_stops_at_a_record_missing_an_arm():
    # an earlier run with fewer arms must not be counted as done for more
    lines = [line("a", "cell", "dump"), line("b", "cell"), line("c", "cell", "dump")]
    assert replay(lines, ("cell", "dump"))[1] == {"a"}


if __name__ == "__main__":
    test_takes_the_complete_prefix()
    test_drops_a_half_written_last_line()
    test_stops_at_a_record_missing_an_arm()
    print("ok")
