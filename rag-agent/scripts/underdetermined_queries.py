"""How many questions do not name a unique cell?

The reader's remaining errors are not all fixable. Reading the OSC=1 misses by
hand (HiTab dev, budget 512) split them three ways: the scorer refusing a right
value, the reader picking a header-sharing neighbour, and questions that simply
do not say which neighbour they mean --

    "how many hours did volunteers aged 35 to 44 spend on volunteering?"
      age > 35 to 44 > average annual volunteer hours > 2013: 122   <- gold
      age > 35 to 44 > average annual volunteer hours > 2010: 136
      age > 35 to 44 > average annual volunteer hours > 2007: 158
      age > 35 to 44 > average annual volunteer hours > 2004: 152

Every year matches the question equally. No retrieval unit and no reader can
separate them, so those queries are a ceiling on answer EM rather than a loss
to be fixed. This counts them WITHOUT looking at the gold cell -- it scores
every cell of the gold table on how much of its header path the question names,
and asks whether the best score is a tie between cells holding different values.
Label-free, so it would be a corpus statistic in the same sense as the collision
rate -- it would say how much of the gap is reachable before any experiment runs.

RESULT: IT DOES NOT WORK. Do not build on it. On HiTab dev it flags 950 of 1064
queries (89.3%), and against the observed answers at budget 512 the flag does not
separate: EM .833 on the 12 queries it calls determined against .797 on the 64 it
calls underdetermined. A flag that fires on nine questions in ten has no
discriminating power. The volunteer case above is real -- reading it by hand is
what raised the question -- but whole-label lexical matching cannot isolate that
class, because HiTab questions are cut from surrounding report sentences and
routinely leave the axis implicit in the question text itself.

Kept as the record of a dead heuristic, not as a tool. A working version would
need the axis a question leaves implicit, which is the same problem one level up.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]

from corpus_dump_vs_cell import hitab_corpus  # noqa: E402

# words that match everything and so decide nothing
_STOP = frozenset("""a an the of in on at to for and or by with from as is are was were
be been total all other others percent percentage number value amount rate
""".split())
_TOK = re.compile(r"[a-z0-9]+")


def tokens(s: str) -> frozenset[str]:
    return frozenset(t for t in _TOK.findall(s.lower()) if t not in _STOP)


def named(label: str, q_tokens: frozenset[str]) -> bool:
    """Does the question name this header label? All content tokens must appear.

    Partial credit would let "2013" match on the "20" of "2013 to 2014"; the
    whole-label rule is what makes an unnamed year read as unnamed.
    """
    lt = tokens(label)
    return bool(lt) and lt <= q_tokens


def verdict(question: str, cells: list[tuple[list[str], str]]) -> tuple[int, int]:
    """(cells tied at the best score, distinct values among them)."""
    q = tokens(question)
    best, tied = -1, []
    for path, value in cells:
        score = sum(named(lab, q) for lab in path)
        if score > best:
            best, tied = score, [value]
        elif score == best:
            tied.append(value)
    return len(tied), len(set(tied))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--population", default="")
    ap.add_argument("--out", default="results/underdetermined_queries.json")
    args = ap.parse_args()

    C = hitab_corpus(args.data_dir, args.split, args.population)
    paths: dict[tuple, list[str]] = {}
    for own, txt in zip(C.cell_owner, C.cell_text):
        # the S2/S3 sentence renders as "a > b > c: v"; the labels are what the
        # question can name and the value is what it must return
        head, _, value = txt.rpartition(":")
        paths[own] = ([p.strip() for p in head.split(">")], value.strip())

    per_query, hist = [], Counter()
    for q in C.queries:
        cells = [v for k, v in paths.items() if k[0] == q["gold_table"]]
        if not cells:
            continue
        tied, distinct = verdict(q["question"], cells)
        per_query.append({"query_id": q["query_id"], "tied": tied, "distinct_values": distinct})
        hist[min(distinct, 5)] += 1

    n = len(per_query)
    under = sum(1 for r in per_query if r["distinct_values"] > 1)
    out = {
        "experiment": "questions that do not name a unique cell",
        "method": "label-free: score every cell of the gold table by how many of "
                  "its header labels the question names, then ask whether the top "
                  "score is a tie across cells holding different values",
        "population": args.population or f"{args.split}, all queries with a gold table",
        "n": n,
        "underdetermined": under,
        "underdetermined_rate": round(under / n, 4) if n else None,
        "distinct_values_at_best_score": {str(k): hist[k] for k in sorted(hist)},
        "per_query": per_query,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"n={n}  underdetermined={under} ({under / n:.1%})" if n else "no queries")
    print("distinct values tied at the best score:",
          {k: hist[k] for k in sorted(hist)})
    print(f"-> {args.out}")
    return 0


def demo() -> None:
    """The volunteer table above: without a year, four cells tie."""
    cells = [(["age", "35 to 44", "average annual volunteer hours", y], v)
             for y, v in (("2013", "122"), ("2010", "136"),
                          ("2007", "158"), ("2004", "152"))]
    cells += [(["age", "45 to 54", "average annual volunteer hours", "2013"], "150")]
    q = "how many hours did volunteers aged 35 to 44 spend on volunteering?"
    assert verdict(q, cells) == (4, 4), verdict(q, cells)

    # naming the year settles it
    q2 = q.replace("volunteering?", "volunteering in 2013?")
    assert verdict(q2, cells) == (1, 1), verdict(q2, cells)
    print("ok")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        raise SystemExit(main())
