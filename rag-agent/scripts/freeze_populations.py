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


def hitab_dev_multicell_lookup(data_dir: str) -> tuple[list[str], dict]:
    """Lookup queries that need SEVERAL gold cells -- the multi-cell counterpart
    of ``hitab_dev_lookup_all``.

    Same filters as that population except the cell count: the header tree must
    build and every operand must land inside the grid, so the corpus can hold
    the query. The aggregation must NOT be arithmetic -- those are already
    ``hitab_dev_corpus_arith`` -- which leaves reading questions whose answer
    happens to be pinned by more than one cell.

    Two shapes live here and the analysis has to keep them apart (60 / 34 of the
    94 at the freeze): the answer is a single value that other cells only
    identify ("how many games did barnsley play in the season it scored eight
    goals" links the 8 and the 212), or the answer is a list as long as the cell
    set. `results/lookup_gap/multicell_pop.md` counts both.
    """
    from point3_reconstruction_cost import build_table_paths

    queries, tables = load_queries(data_dir, "dev")
    raw_dir = Path(data_dir) / "data/tables/raw"
    paths = {}
    for tid, bt in tables.items():
        f = raw_dir / f"{tid}.json"
        if not f.exists():
            continue
        try:
            raw = json.load(open(f))
        except Exception:
            continue
        pt = build_table_paths(raw, bt)
        if pt is not None:
            paths[tid] = pt
    ids = []
    for q in queries:
        ops = [(op.row, op.col) for op in q.gold_operands]
        pt = paths.get(q.gold_table_id)
        if len(ops) < 2 or (q.aggregation or "none") in ARITH or pt is None:
            continue
        if all(0 <= r < pt["n_r"] and 0 <= c < pt["n_c"] for r, c in ops):
            ids.append(q.query_id)
    return ids, {"dataset": "hitab", "split": "dev",
                 "filter": "len(gold_operands)>=2 and aggregation not in ARITH "
                           "and build_table_paths is not None and every operand "
                           "in grid",
                 "order": "dataset order", "n": len(ids),
                 "disjoint_from": ["hitab_dev_lookup_all",
                                   "hitab_dev_corpus_arith"],
                 "counted_by": "analysis/multicell_lookup_pop.py"}


def hitab_lookup_multi(data_dir: str, split: str) -> tuple[list[str], dict]:
    """Lookup queries whose ANSWER is more than one cell -- the multi-cell
    counterpart of ``hitab_{split}_lookup_all``.

    Answer cells are counted from the ``[ANSWER]`` bucket of ``quantity_link``
    ALONE. ``gold_operands`` is not the answer: `hitab.py: _coords_of` pools
    every bucket, so a figure the *question* quotes resolves to an operand too.
    Selecting on ``len(gold_operands)>=2`` therefore admits queries whose answer
    is a single cell -- 47 of 81 at the 2026-09-01 measurement, which is what
    ``hitab_dev_multicell_lookup`` (n=94) did and why this population replaces it.

    The rest of the filter matches the single-cell lookup pool so the two are
    comparable: non-arithmetic, header tree builds, every resolved operand lands
    inside the grid, same seed-0 shuffle.

    CAVEAT for anyone reporting this leg: membership is chosen by ANSWER cells,
    but OSC is still defined over the OPERAND set. State the mismatch.
    """
    from point3_reconstruction_cost import build_table_paths
    from rag_agent.bench.hitab import load_samples

    samples = {s["id"]: s for s in load_samples(data_dir, split)}
    queries, tables = load_queries(data_dir, split)
    raw_dir = Path(data_dir) / "data/tables/raw"
    paths = {}
    for tid, bt in tables.items():
        f = raw_dir / f"{tid}.json"
        if not f.exists():
            continue
        try:
            raw = json.load(open(f))
        except Exception:
            continue
        pt = build_table_paths(raw, bt)
        if pt is not None:
            paths[tid] = pt
    ids = []
    for q in queries:
        lc = (samples.get(q.query_id) or {}).get("linked_cells") or {}
        answer_cells = (lc.get("quantity_link") or {}).get("[ANSWER]") or {}
        pt = paths.get(q.gold_table_id)
        if (q.aggregation or "none") != "none" or len(answer_cells) < 2 or pt is None:
            continue
        ops = [(op.row, op.col) for op in q.gold_operands]
        if ops and all(0 <= r < pt["n_r"] and 0 <= c < pt["n_c"] for r, c in ops):
            ids.append(q.query_id)
    random.Random(0).shuffle(ids)
    return ids, {"dataset": "hitab", "split": split,
                 "filter": "aggregation == 'none' and "
                           "len(quantity_link['[ANSWER]'])>=2 and "
                           "build_table_paths is not None and every resolved "
                           "operand in grid",
                 "order": "random.Random(0).shuffle", "n": len(ids),
                 "answer_cells_from": "quantity_link['[ANSWER]'] only -- NOT "
                                      "gold_operands, which pools all buckets",
                 "metric_caveat": "membership by answer cells, OSC by operands",
                 "disjoint_from": [f"hitab_{split}_lookup_all"],
                 "supersedes": "hitab_dev_multicell_lookup (len(gold_operands)>=2, n=94)"}


def hitab_train_lookup_expanded(data_dir: str) -> tuple[list[str], dict]:
    """`hitab_train_lookup_all` 위에 gold 셀이 2개 이상인 질문을 얹은 학습 풀.

    검색기를 맞추는 데는 "이 질문에 이 셀이 가깝다"만 있으면 되고, 그것은 답이 몇
    칸이든 유효하다. 단일 gold만 쓰면 코퍼스가 담는 train 질문의 23%, (질문, gold셀)
    쌍의 48%가 버려진다 (2026-09-02 실측: 4,774 질문 / 7,019 쌍).

    필터는 `hitab_train_lookup_all`에서 개수 조건만 뺀 것이다: 헤더 트리가 서고
    해결된 gold 피연산자가 격자 안에 있는 train 질문 전부. 순서도 같은 seed-0 shuffle.

    **학습 전용.** dev/test와 표를 하나도 공유하지 않는다 (교집합 0, 실측).
    `hitab_train_lookup_multi`(150)와는 겹친다 -- 그 모집단도 학습 전용이라 누출이
    아니지만, 다중 셀 레그를 보고할 때 train에 들어갔다는 사실을 밝힐 것.

    PREREG-2026-09-02-train-expand.md
    """
    import corpus_dump_vs_cell as cdv
    C = cdv.hitab_corpus(data_dir, "train", "")      # 동결 미적용 = 전체 풀
    have = set(C.cell_owner)
    ids = [q["query_id"] for q in C.queries
           if any(k in have for k in q["gold_cells"])]
    random.Random(0).shuffle(ids)
    return ids, {"dataset": "hitab", "split": "train",
                 "filter": "build_table_paths is not None and at least one "
                           "resolved gold operand is in the indexed grid "
                           "(no cap on the number of gold cells)",
                 "order": "random.Random(0).shuffle", "n": len(ids),
                 "note": "TRAINING ONLY -- never evaluate on this population",
                 "superset_of": "hitab_train_lookup_all",
                 "used_by": ["PREREG-2026-09-02-train-expand.md"]}


def hitab_train_table_split(data_dir: str, part: str,
                            frac: float = 0.2) -> tuple[list[str], dict]:
    """Split ``hitab_train_lookup_all`` BY TABLE into a fit pool and a selection pool.

    Every arm in this repo was chosen on dev, three layers deep (encoder arm ->
    alpha -> data expansion), and dev flipped against test all three times. dev is
    no longer a selection instrument and no longer a clean evaluation set either.
    This gives selection its own split so dev can go back to being an evaluation
    set that nothing was chosen on.

    Split BY TABLE, not by query. Fine-tuning draws its hard negatives from the
    gold cell's own table, so a table split across the two pools would put the
    selection queries' sibling cells into training. HiTab's own train/dev/test
    share zero tables for the same reason.

    The corpus is unaffected: ``hitab_corpus`` indexes every table of the split it
    is given, so a selection query is searched against the WHOLE train corpus --
    fit tables included. Cells of fit tables were seen in training (as siblings of
    other queries' gold), cells of selection tables never were. State that when
    reporting: selection numbers are not comparable in absolute value to dev, whose
    corpus is entirely unseen.

    Order is inherited from the parent freeze (seed-0 shuffle); the table sample is
    ``random.Random(0)`` over the sorted table ids.
    """
    parent = pop_mod.read("hitab_train_lookup_all")
    if parent is None:
        raise SystemExit("freeze hitab_train_lookup_all first")
    ids = parent[0]
    queries, _ = load_queries(data_dir, "train")
    gt = {q.query_id: q.gold_table_id for q in queries}
    tables = sorted({gt[i] for i in ids})
    sel_t = set(random.Random(0).sample(tables, round(len(tables) * frac)))
    keep = [i for i in ids if (gt[i] in sel_t) == (part == "sel")]
    return keep, {
        "dataset": "hitab", "split": "train",
        "filter": f"hitab_train_lookup_all, {part} side of a by-TABLE split",
        "order": "inherited from hitab_train_lookup_all (random.Random(0).shuffle)",
        "n": len(keep), "n_tables": len(sel_t) if part == "sel"
        else len(tables) - len(sel_t), "table_split": f"random.Random(0).sample "
        f"of {len(tables)} tables, frac={frac} to sel",
        "subset_of": "hitab_train_lookup_all",
        "disjoint_from": [f"hitab_train_{'fit' if part == 'sel' else 'sel'}_lookup_all"],
        "note": ("SELECTION ONLY -- never train on this population, never report it "
                 "as an evaluation result" if part == "sel" else
                 "TRAINING ONLY -- never evaluate on this population"),
        "used_by": ["PREREG-2026-09-02-devbias.md"]}


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
    """Every RealHiTBench Numerical Reasoning query -- EM only, no gold cells.

    ``rhb_lookup_all`` requires the answer string to resolve to a unique cell
    set, which is what OSC needs. An NR answer is usually a COMPUTED value that
    no cell holds, so that requirement keeps 55 of 771 and, within each subtype,
    the instances whose answer was already printed in the table. This population
    drops the requirement so the run sits on the same queries the published
    table reports on (arXiv:2506.13405 Table 2, NR EM), and pays for it by not
    being able to report OSC.
    """
    from corpus_dump_vs_cell import realhitbench_corpus
    C = realhitbench_corpus(pin=False, question_types=("Numerical Reasoning",),
                            require_gold_cells=False)
    return [q["query_id"] for q in C.queries], {
        "dataset": "realhitbench", "split": "all",
        "question_types": ["Numerical Reasoning"],
        "filter": "table parses, >=3 rows; answer is non-empty. Gold cells NOT "
                  "required -- OSC is null where the answer resolves to no cell",
        "order": "QA_final.json order", "n": len(C.queries),
        "used_by": ["corpus_dump_vs_cell", "PREREG-2026-08-27-rhb-nr-large-tables"]}


def hitab_train_alltypes(data_dir: str, part: str) -> tuple[list[str], dict]:
    """모든 질의 종류(단일·다중 조회 + 산술)를 한 학습 풀로 — 표 기준 fit/sel 분할을 그대로 따른다.

    `hitab_train_fit_lookup_all` 은 단일 셀 조회만 담아, 그것으로 학습한 인코더(p0)는 산술
    질의(피연산자 여러 개)를 본 적이 없다. 이 풀은 `hitab_train_lookup_expanded`(gold 셀 수 제한
    없음) ∪ `hitab_train_corpus_arith` 를 합치고, sel 표(`hitab_train_sel_lookup_all` 의 표)에
    속한 질의를 `sel` 쪽으로, 나머지를 `fit` 쪽으로 보낸다. 표 단위 분할이므로 sel 질의의 형제
    셀이 학습에 들어가지 않는다.
    """
    exp = pop_mod.read("hitab_train_lookup_expanded")
    ar = pop_mod.read("hitab_train_corpus_arith")
    sel = pop_mod.read("hitab_train_sel_lookup_all")
    if not (exp and ar and sel):
        raise SystemExit("freeze hitab_train_lookup_expanded / corpus_arith / sel_lookup_all first")
    queries, _ = load_queries(data_dir, "train")
    gt = {q.query_id: q.gold_table_id for q in queries}
    sel_t = {gt[i] for i in sel[0]}
    ids = sorted(set(exp[0]) | set(ar[0]), key=lambda i: (exp[0] + ar[0]).index(i))
    keep = [i for i in ids if (gt[i] in sel_t) == (part == "sel")]
    return keep, {"dataset": "hitab", "split": "train",
                  "filter": f"(hitab_train_lookup_expanded ∪ hitab_train_corpus_arith), {part} side "
                            "of the by-TABLE split defined by hitab_train_sel_lookup_all's tables",
                  "order": "expanded order then corpus_arith order", "n": len(keep),
                  "n_tables": len({gt[i] for i in keep}),
                  "note": ("SELECTION ONLY -- never train on this population" if part == "sel"
                           else "TRAINING ONLY -- never evaluate on this population")}


SPECS = {
    "hitab_dev_lookup_single": lambda a: hitab_dev_lookup_single(a.data_dir),
    "hitab_dev_lookup_all": lambda a: hitab_dev_lookup_all(a.data_dir),
    "hitab_test_lookup_all": lambda a: hitab_test_lookup_all(a.data_dir),
    "hitab_train_lookup_all": lambda a: hitab_train_lookup_all(a.data_dir),
    "hitab_train_lookup_expanded": lambda a: hitab_train_lookup_expanded(a.data_dir),
    "hitab_train_fit_lookup_all": lambda a: hitab_train_table_split(a.data_dir, "fit"),
    "hitab_train_sel_lookup_all": lambda a: hitab_train_table_split(a.data_dir, "sel"),
    "hitab_train_fit_alltypes": lambda a: hitab_train_alltypes(a.data_dir, "fit"),
    "hitab_train_sel_alltypes": lambda a: hitab_train_alltypes(a.data_dir, "sel"),
    "hitab_dev_arith": lambda a: hitab_arith(a.data_dir, "dev", 1),
    "hitab_dev_arith_m2": lambda a: hitab_arith(a.data_dir, "dev", 2),
    "hitab_train_arith": lambda a: hitab_arith(a.data_dir, "train", 1),
    "hitab_train_arith_m2": lambda a: hitab_arith(a.data_dir, "train", 2),
    "hitab_dev_multicell_lookup": lambda a: hitab_dev_multicell_lookup(a.data_dir),
    "hitab_dev_lookup_multi": lambda a, s="dev": hitab_lookup_multi(a.data_dir, s),
    "hitab_test_lookup_multi": lambda a, s="test": hitab_lookup_multi(a.data_dir, s),
    "hitab_train_lookup_multi": lambda a, s="train": hitab_lookup_multi(a.data_dir, s),
    "hitab_dev_corpus_arith": lambda a: hitab_corpus_arith(a.data_dir, "dev"),
    "hitab_test_corpus_arith": lambda a: hitab_corpus_arith(a.data_dir, "test"),
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
