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


def hitab_test_lookup_all(data_dir: str) -> tuple[list[str], dict]:
    """The same derivation as ``hitab_dev_lookup_all``, on the untouched split.

    Every design choice in this repo was made on dev. This is the confirmation
    population: same filter, same seed-0 order, whole pool, no pin to unset
    (``build_population`` pins dev only). It is derived ONCE, here, and the
    numbers that come out of it are reported whichever way they fall.
    """
    from manual_sentence_ceiling import build_population
    pop, _, _ = build_population(data_dir, "test", 10**9)
    return [q.query_id for q in pop], {
        "dataset": "hitab", "split": "test", "filter": "len(gold_operands)==1 and "
        "build_table_paths is not None and operand in grid",
        "order": "random.Random(0).shuffle", "n": len(pop),
        "used_by": ["corpus_dump_vs_cell (test-split confirmation)"]}


def hitab_train_lookup_all(data_dir: str) -> tuple[list[str], dict]:
    """The same derivation on train -- the only split an encoder may be fit on.

    Fine-tuning the retriever needs supervision, and the only supervision that
    cannot leak is train's. dev is where every design choice was made and stays
    the evaluation set; test stays untouched. Freezing the train pool the same
    way as the other two means the fitted encoder's training set is a file on
    disk, so a later run can prove no dev or test query was ever in it.
    """
    from manual_sentence_ceiling import build_population
    pop, _, _ = build_population(data_dir, "train", 10**9)
    return [q.query_id for q in pop], {
        "dataset": "hitab", "split": "train", "filter": "len(gold_operands)==1 and "
        "build_table_paths is not None and operand in grid",
        "order": "random.Random(0).shuffle", "n": len(pop),
        "used_by": ["encoder fine-tuning (PREREG-2026-08-25-retrieval-max)"],
        "note": "TRAINING ONLY -- never evaluate on this population"}


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


def hitab_lookup_multi(data_dir: str, split: str) -> tuple[list[str], dict]:
    """Lookup queries whose ANSWER spans more than one cell -- the multi-cell task.

    Every lookup population above is single-answer (``build_population`` filters
    ``len(ops) != 1``), so the multi-cell reading case has never had a population.
    It is not the arithmetic pool either: ``aggregation == "none"`` means nothing
    is computed -- the answer is simply several cells read out. That is the axis
    this isolates: reading N cells, not combining them.

    ``gold_operands`` is NOT the answer. ``_coords_of`` pools every bucket of
    ``quantity_link``, which carries the numbers the source sentence cites as
    well as the answer, so 47 of the 81 queries with >=2 operands answer with a
    single cell. Membership therefore counts the ``[ANSWER]`` bucket of the raw
    annotation, and only the grid check uses the resolved operands.

    The pool is small and that is a property of HiTab, not a choice made here:
    44 dev / 37 test / 194 train out of 1,195 / 1,133 / 5,208 lookup queries,
    before the reconstructor drops any. n=44 cannot separate .85 from .90; the
    dev leg is a ceiling probe and a claim, if made, is made on dev+test.

    Membership runs through the same reconstructor as the single-cell pools -- a
    table whose header tree will not build is dropped -- so the 21.5% alignment
    exclusion applies identically and the two legs stay comparable.
    """
    from manual_sentence_ceiling import build_population
    # n=0 runs the reconstructor pass and returns no queries; paths is the point
    _, _, paths = build_population(data_dir, split, 0)
    raw = {}
    with open(Path(data_dir) / f"data/{split}_samples.jsonl") as fh:
        for line in fh:
            d = json.loads(line)
            raw[str(d["id"])] = d
    queries, _ = load_queries(data_dir, split)
    pop, n_ans = [], {}
    for q in queries:
        pt = paths.get(q.gold_table_id)
        if pt is None:
            continue
        if (q.aggregation or "none") != "none":
            continue                      # computed answer -> that is the arith pool
        d = raw.get(str(q.query_id))
        if d is None:
            continue
        ans = ((d["linked_cells"].get("quantity_link") or {}).get("[ANSWER]") or {})
        if len(ans) < 2:
            continue                      # single cell -> hitab_{split}_lookup_all
        ops = {(op.row, op.col) for op in q.gold_operands}
        if not ops or not all(0 <= r < pt["n_r"] and 0 <= c < pt["n_c"]
                              for r, c in ops):
            continue                      # operand outside the reconstructed grid
        pop.append(q)
        n_ans[q.query_id] = len(ans)
    random.Random(0).shuffle(pop)         # same seed-0 order as every lookup pool
    return [q.query_id for q in pop], {
        "dataset": "hitab", "split": split,
        "filter": "aggregation == 'none' and len(linked_cells.quantity_link"
                  "['[ANSWER]']) >= 2 and build_table_paths is not None and "
                  "every resolved operand in grid",
        "order": "random.Random(0).shuffle", "n": len(pop),
        "answer_cells_hist": {str(k): sum(1 for q in pop if n_ans[q.query_id] == k)
                              for k in sorted({n_ans[q.query_id] for q in pop})},
        "note": "answer cells != gold_operands; OSC is defined over operands",
        "used_by": ["multi-cell lookup leg -- retrieval + gold_cell ceiling"]}


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


def aitqa_lookup_all(_data_dir: str) -> tuple[list[str], dict]:
    """Every AIT-QA question whose answer strings resolve to a unique cell set.

    AIT-QA annotates no gold cells, so the population is whatever
    :func:`corpus_dump_vs_cell.aitqa_corpus` recovers by matching answer strings
    against cell values -- i.e. a function of the code, which is the thing
    ``rag_agent/bench/population.py`` exists to freeze. Cross-dataset claims are
    read off this corpus, so it has to stop moving before they are made
    (PREREG-2026-08-25-structural-discriminator.md §9).
    """
    from corpus_dump_vs_cell import aitqa_corpus
    C = aitqa_corpus(pin=False)
    return [q["query_id"] for q in C.queries], {
        "dataset": "aitqa", "split": "all",
        "filter": "answer strings resolve to a unique cell set; ambiguous dropped",
        "order": "aitqa_questions.jsonl file order", "n": len(C.queries),
        "used_by": ["corpus_dump_vs_cell"]}


def rhb_lookup_all(_data_dir: str) -> tuple[list[str], dict]:
    """The same for RealHiTBench, where membership also runs through the parser.

    Here the answer match happens on a grid the header reconstructor produced, so
    a reconstruction change moves membership as well as the metric -- one more
    reason than AIT-QA has to freeze it. All SubQTypes; a subtype-filtered run
    does not pin.
    """
    from corpus_dump_vs_cell import realhitbench_corpus
    C = realhitbench_corpus(pin=False)
    return [q["query_id"] for q in C.queries], {
        "dataset": "realhitbench", "split": "all", "subqtypes": "all",
        "filter": "table parses, >=3 rows; answer strings resolve to a unique "
                  "cell set; ambiguous dropped",
        "order": "QA_final.json order", "n": len(C.queries),
        "used_by": ["corpus_dump_vs_cell"]}


def rhb_nr_all(_data_dir: str) -> tuple[list[str], dict]:
    """Every RealHiTBench Numerical Reasoning question, gold-cell filter LIFTED.

    The NR answer is usually a COMPUTED value that is written in no cell, so the
    answer-string match that defines `rhb_lookup_all` keeps 54 of 771 (7.0%) --
    and the 54 are the easy instances, the ones whose answer happens to already
    be in the table. That filter exists to make OSC measurable;
    PREREG-2026-08-27-rhb-nr-large-tables.md §2 gives OSC up for this population
    and reads EM instead, so the filter comes off. The run then sits on the same
    queries the published table reports on (arXiv:2506.13405 Table 2, NR EM).

    Membership still runs through the HTML parser and the header reconstructor
    -- 7 of the 771 tables do not parse -- which is why it is frozen: a
    reconstruction change would move the population under the numbers.
    """
    from corpus_dump_vs_cell import RHB_POPS, realhitbench_corpus
    C = realhitbench_corpus(pin=False, **RHB_POPS["rhb_nr_all"])
    return [q["query_id"] for q in C.queries], {
        "dataset": "realhitbench", "split": "all",
        "question_types": ["Numerical Reasoning"],
        "filter": "table parses, >=3 rows. NO answer-match filter: the answer is "
                  "computed and need not appear in any cell, so gold_cells may be "
                  "empty and OSC is undefined -- EM only",
        "order": "QA_final.json order", "n": len(C.queries),
        "used_by": ["corpus_dump_vs_cell", "PREREG-2026-08-27-rhb-nr-large-tables"]}


SPECS = {
    "hitab_dev_lookup_single": lambda a: hitab_dev_lookup_single(a.data_dir),
    "hitab_dev_lookup_all": lambda a: hitab_dev_lookup_all(a.data_dir),
    "hitab_test_lookup_all": lambda a: hitab_test_lookup_all(a.data_dir),
    "hitab_train_lookup_all": lambda a: hitab_train_lookup_all(a.data_dir),
    "hitab_dev_arith": lambda a: hitab_arith(a.data_dir, "dev", 1),
    "hitab_dev_arith_m2": lambda a: hitab_arith(a.data_dir, "dev", 2),
    "hitab_train_arith": lambda a: hitab_arith(a.data_dir, "train", 1),
    "hitab_train_arith_m2": lambda a: hitab_arith(a.data_dir, "train", 2),
    "hitab_dev_lookup_multi": lambda a: hitab_lookup_multi(a.data_dir, "dev"),
    "hitab_test_lookup_multi": lambda a: hitab_lookup_multi(a.data_dir, "test"),
    "hitab_train_lookup_multi": lambda a: hitab_lookup_multi(a.data_dir, "train"),
    "hitab_dev_corpus_arith": lambda a: hitab_corpus_arith(a.data_dir, "dev"),
    "hitab_train_corpus_arith": lambda a: hitab_corpus_arith(a.data_dir, "train"),
    "hitab_dev_size_strata": lambda a: hitab_size_strata(a.data_dir, "dev", a.per_bucket),
    "hitab_train_size_strata": lambda a: hitab_size_strata(a.data_dir, "train", a.per_bucket),
    "aitqa_lookup_all": lambda a: aitqa_lookup_all(a.data_dir),
    "rhb_lookup_all": lambda a: rhb_lookup_all(a.data_dir),
    "rhb_nr_all": lambda a: rhb_nr_all(a.data_dir),
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
