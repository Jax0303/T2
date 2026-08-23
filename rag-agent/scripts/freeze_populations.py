#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Derive every experiment population once and commit it to ``populations/``.

Run this ONCE, then never again except on purpose (--force). After it runs the
population is a committed file rather than whatever the current reconstruction
code happens to produce, so a header-parser change moves the metric and not the
membership. See ``rag_agent/bench/population.py`` for why.

  PYTHONPATH=. python3 scripts/freeze_populations.py            # write what is missing
  PYTHONPATH=. python3 scripts/freeze_populations.py --force    # re-freeze (changes n!)
  PYTHONPATH=. python3 scripts/freeze_populations.py --check    # drift report, writes nothing
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_agent.bench import population as pop_mod
from rag_agent.bench.hitab import load_queries

# byte-identical to inject_osc_matched.py / pool_scale_matched.py
ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}


def hitab_dev_lookup_single(data_dir: str) -> tuple[list[str], dict]:
    """L-13 / L-14 population: single gold operand, seed-0 shuffle, first 100."""
    from manual_sentence_ceiling import build_population
    pop, _, _ = build_population(data_dir, "dev", 100)
    return [q.query_id for q in pop], {
        "dataset": "hitab", "split": "dev", "filter": "len(gold_operands)==1 and "
        "build_table_paths is not None and operand in grid",
        "order": "random.Random(0).shuffle", "n": 100,
        "used_by": ["manual_sentence_ceiling", "cell_retrieval_matrix",
                    "pipeline_lookup_llm"]}


def hitab_dev_lookup_all(data_dir: str) -> tuple[list[str], dict]:
    """Every dev lookup query, not the first 100 of them.

    ``hitab_dev_lookup_single`` takes the first 100 after a seed-0 shuffle, which
    is enough to separate arms that differ by 20-40 points and not enough to
    separate ones that differ by 7 (ours .610 vs the MT2Net template .540 came out
    17:10, p=.25). Taking the WHOLE derived pool -- 830 -- is the version of "more
    data" that cannot be accused of stopping where it suited: there is nothing
    left to add, so no stopping point was chosen. test stays untouched.
    """
    from manual_sentence_ceiling import build_population
    import rag_agent.bench.population as _pm
    from manual_sentence_ceiling import POPULATION as _pinned
    # build_population pins dev to the 100-query freeze; derive the full pool by
    # asking for the split it does not pin, then re-deriving dev without the pin
    orig = _pm.pin
    _pm.pin = lambda name, pop: pop          # noqa: E731 -- restored below
    try:
        pop, _, _ = build_population(data_dir, "dev", 10**9)
    finally:
        _pm.pin = orig
    return [q.query_id for q in pop], {
        "dataset": "hitab", "split": "dev", "filter": "len(gold_operands)==1 and "
        "build_table_paths is not None and operand in grid",
        "order": "random.Random(0).shuffle", "n": len(pop),
        "superset_of": _pinned,
        "used_by": ["corpus_dump_vs_cell (MT2Net head-to-head)"]}


def hitab_arith(data_dir: str, split: str, m_min: int) -> tuple[list[str], dict]:
    """The matched-control populations.

    ``m_min=1`` is what the scripts actually RUN (they record ``m`` per query);
    ``m_min=2`` is the subset the results are REPORTED on (n=161 dev / 750 train).
    Both are frozen because a summary that re-derives its own subset can drift
    away from the run just as easily as a run can.
    """
    queries, _ = load_queries(data_dir, split)
    ids = [q.query_id for q in queries
           if (q.aggregation or "none") in ARITH
           and len({(o.row, o.col) for o in q.gold_operands}) >= m_min]
    return ids, {"dataset": "hitab", "split": split,
                 "filter": f"aggregation in ARITH and m>={m_min}",
                 "order": "dataset order", "n": len(ids),
                 "used_by": ["inject_osc_matched", "pool_scale_matched"]}


def hitab_size_strata(data_dir: str, split: str, per_bucket: int,
                      edges=(512, 1024)) -> tuple[list[str], dict]:
    """Table-size strata: the crossover population.

    ``baseline_comparison_llm``'s own draw is stratified by question kind, and on
    HiTab that lands 85% of the sample on tables that fit a 1024-token budget
    whole -- the regime where dumping the table is simply better and retrieval
    has nothing to do. The aggregate number is then dominated by that regime and
    says nothing about the one the method is for.

    This draws equally from three strata by the size of the WHOLE table in the
    same tokenizer the budget uses, so "at what table size does retrieval start
    to win" is answerable instead of being averaged away.

    The edges are (512, 1024) and not something larger because HiTab has no large
    tables to split on: measured over every dev/train query whose header tree
    builds, the biggest whole table is 3,095 tokens and p99 is ~2,200. Edges of
    (1024, 4096) leave the top stratum EMPTY. Whatever this population shows, it
    cannot show a crossover that lives past a few thousand tokens -- that has to
    come from a dataset that has such tables.
    """
    from baseline_comparison_llm import Budget, markdown_table
    from point3_reconstruction_cost import build_table_paths

    queries, tables = load_queries(data_dir, split)
    raw_dir = Path(data_dir) / "data/tables/raw"
    bud = Budget()
    size: dict[str, int] = {}
    for tid, bt in tables.items():
        f = raw_dir / f"{tid}.json"
        if not f.exists():
            continue
        try:
            raw = json.load(open(f))
        except Exception:
            continue
        pt = build_table_paths(raw, bt)
        if pt is None:
            continue
        n_head = max(1, len(raw["texts"]) - pt["n_r"])
        size[tid] = bud.count("\n".join(markdown_table(raw, n_head)))

    buckets: dict[str, list[str]] = {"small": [], "mid": [], "large": []}
    for q in queries:
        n = size.get(q.gold_table_id)
        if n is None or not q.gold_operands:
            continue
        name = "small" if n < edges[0] else ("mid" if n < edges[1] else "large")
        buckets[name].append(q.query_id)

    rng = random.Random(42)
    ids, counts = [], {}
    for name in ("small", "mid", "large"):
        rng.shuffle(buckets[name])
        take = buckets[name][:per_bucket]
        counts[name] = {"drawn": len(take), "available": len(buckets[name])}
        ids += take
    return ids, {"dataset": "hitab", "split": split,
                 "filter": "gold_operands and header tree builds",
                 "stratify": f"whole-table tokens (bge-small), edges={list(edges)}",
                 "order": "small, mid, large; Random(42) shuffle within stratum",
                 "buckets": counts, "n": len(ids),
                 "used_by": ["baseline_comparison_llm"]}


def hitab_corpus_arith(data_dir: str, split: str) -> tuple[list[str], dict]:
    """``corpus_dump_vs_cell``'s population: arithmetic queries the corpus can hold.

    Narrower than ``hitab_{split}_arith`` by one condition -- the gold table's
    header tree must build, because that is what the corpus in that experiment is
    made of. 39 of the 214 dev arithmetic queries fail it. Keeping them would add
    a constant zero to every arm (no arm can retrieve a table that is not indexed)
    rather than telling the arms apart, so they are excluded here and the
    exclusion is named instead of being hidden inside a script.
    """
    from point3_reconstruction_cost import build_table_paths

    queries, tables = load_queries(data_dir, split)
    raw_dir = Path(data_dir) / "data/tables/raw"
    ok = set()
    for tid, bt in tables.items():
        f = raw_dir / f"{tid}.json"
        if not f.exists():
            continue
        try:
            raw = json.load(open(f))
        except Exception:
            continue
        if build_table_paths(raw, bt) is not None:
            ok.add(tid)
    ids = [q.query_id for q in queries
           if (q.aggregation or "none") in ARITH and q.gold_operands
           and q.gold_table_id in ok]
    return ids, {"dataset": "hitab", "split": split,
                 "filter": "aggregation in ARITH and gold_operands and "
                           "gold table's header tree builds",
                 "order": "dataset order", "n": len(ids),
                 "used_by": ["corpus_dump_vs_cell"]}


SPECS = {
    "hitab_dev_lookup_single": lambda a: hitab_dev_lookup_single(a.data_dir),
    "hitab_dev_lookup_all": lambda a: hitab_dev_lookup_all(a.data_dir),
    "hitab_dev_arith": lambda a: hitab_arith(a.data_dir, "dev", 1),
    "hitab_dev_arith_m2": lambda a: hitab_arith(a.data_dir, "dev", 2),
    "hitab_train_arith": lambda a: hitab_arith(a.data_dir, "train", 1),
    "hitab_train_arith_m2": lambda a: hitab_arith(a.data_dir, "train", 2),
    "hitab_dev_corpus_arith": lambda a: hitab_corpus_arith(a.data_dir, "dev"),
    "hitab_train_corpus_arith": lambda a: hitab_corpus_arith(a.data_dir, "train"),
    "hitab_dev_size_strata": lambda a: hitab_size_strata(a.data_dir, "dev", a.per_bucket),
    "hitab_train_size_strata": lambda a: hitab_size_strata(a.data_dir, "train", a.per_bucket),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--per-bucket", type=int, default=100,
                    help="queries per table-size stratum (size-strata populations)")
    ap.add_argument("--only", default="", help="comma-separated population names")
    ap.add_argument("--force", action="store_true", help="overwrite an existing freeze")
    ap.add_argument("--check", action="store_true", help="report drift, write nothing")
    args = ap.parse_args()

    names = [n for n in (args.only.split(",") if args.only else SPECS) if n]
    rc = 0
    for name in names:
        existing = pop_mod.read(name)
        if existing and not (args.force or args.check):
            print(f"[skip]  {name}: already frozen, n={len(existing[0])}")
            continue
        ids, meta = SPECS[name](args)
        if existing:
            old = set(existing[0])
            new = set(ids)
            lost, gained = sorted(old - new), sorted(new - old)
            status = "same" if not (lost or gained) else "DRIFT"
            print(f"[{status:5}] {name}: frozen {len(old)} -> derived {len(ids)} "
                  f"(lost {len(lost)}, gained {len(gained)})")
            if lost:
                print(f"          lost: {lost[:5]}")
            if args.check:
                rc |= 1 if (lost or gained) else 0
                continue
        p = pop_mod.write(name, ids, meta)
        print(f"[write] {p} n={len(ids)}")
    if args.check and rc:
        print("\ndrift detected: a frozen population is no longer derivable as-is")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
