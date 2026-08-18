#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Hand-written cell sentence = the ceiling. How much of the answer gap is the template?

Retrieval reaches recall@1 ~= .98 while the end-to-end answer sits far below it,
so the loss is downstream of finding the cell. This asks the whiteboard question
directly: swap the ALGORITHM's sentence for the gold cell with one a human wrote,
change nothing else, and re-measure. If the number jumps, the template is the
bottleneck; if it does not, the template is exonerated and the loss lives in the
encoder or the reader.

Three arms over the SAME population, encoder, pool, top-k and LLM. Only the text
of the gold cell's own unit differs (every other cell in the table keeps its
algorithmic text in all three arms, so the distractor pool is held fixed):

  algo_path : "row-path > col-path: value"      -- what production indexes today
  algo_sent : "In {title}, {cols} for {rows} is {v}."  -- the template's sentence form
  manual    : the hand-written sentence from --manual

Two yes/no scores per arm, the two boxes on the whiteboard:
  * ``recall_at_1``  -- is the gold cell the single top hit?
  * ``answer_em``    -- does the reader's answer match gold (hitab_exact_match)?

Population is byte-identical to ``pipeline_lookup_llm.py``: HiTab dev, single
gold operand, ``random.Random(0)`` shuffle, first --n. Gate 1 (table finding) is
oracle there and here.

  # 1. write the worksheet, fill in "manual" for each line by hand
  PYTHONPATH=. .venv/bin/python scripts/manual_sentence_ceiling.py --export \
      diag/manual_sentence_worksheet.jsonl --n 100
  # 2. run
  PYTHONPATH=. .venv/bin/python scripts/manual_sentence_ceiling.py \
      --manual diag/manual_sentences.jsonl --resume
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from rag_agent.bench import population as pop_mod
from rag_agent.bench.hitab import load_queries
from rag_agent.eval.metrics import hitab_exact_match
from rag_agent.generate.answerer import _DIRECT_SYS
from rag_agent.llm.factory import build_llm
from rag_agent.retrieve.encoders import default_encoder
from rag_agent.runenv import guard_resume, run_env
from rag_agent.serialize.verbalize import verbalize_cell
from point3_reconstruction_cost import build_table_paths, cell_text

ARMS = ("algo_path", "algo_sent", "manual")


def mcnemar(a_vals, b_vals) -> dict:
    """Exact two-sided McNemar over paired 0/1 outcomes.

    Only the discordant pairs carry information: n01 queries the first arm gets
    and the second does not, n10 the reverse. Under H0 each discordant pair is a
    coin flip, so p is the two-sided binomial tail. Exact rather than the chi2
    approximation because these runs are n=100 with single-digit discordances.
    """
    from math import comb
    n01 = sum(1 for x, y in zip(a_vals, b_vals) if x and not y)
    n10 = sum(1 for x, y in zip(a_vals, b_vals) if y and not x)
    n = n01 + n10
    p = (1.0 if n == 0 else
         min(1.0, 2 * sum(comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n))
    return {"only_first": n01, "only_second": n10, "exact_p": round(p, 4)}


POPULATION = "hitab_dev_lookup_single"


def build_population(data_dir: str, split: str, n: int):
    """The pipeline_lookup_llm population: single-operand lookup queries.

    The derivation below depends on the reconstructor (a table whose header tree
    will not build is dropped), so once ``populations/hitab_dev_lookup_single.txt``
    exists it -- not this code -- decides the membership. See
    ``rag_agent/bench/population.py``.
    """
    queries, tables = load_queries(data_dir, split)
    raw_dir = Path(data_dir) / "data/tables/raw"
    paths = {}
    for tid, bt in tables.items():
        p = raw_dir / f"{tid}.json"
        if not p.exists():
            continue
        try:
            raw = json.load(open(p))
        except Exception:
            continue
        pt = build_table_paths(raw, bt)
        if pt is not None:
            paths[tid] = pt

    pop = []
    for q in queries:
        pt = paths.get(q.gold_table_id)
        if pt is None:
            continue
        ops = [(op.row, op.col) for op in q.gold_operands]
        if len(ops) != 1:
            continue
        r, c = ops[0]
        if not (0 <= r < pt["n_r"] and 0 <= c < pt["n_c"]):
            continue
        pop.append(q)
    random.Random(0).shuffle(pop)
    if split == "dev":
        pop = pop_mod.pin(POPULATION, pop)
    return pop[:n], tables, paths


def export(pop, tables, paths, out: Path) -> None:
    """One line per query with everything a human needs to write the sentence."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for q in pop:
            bt, pt = tables[q.gold_table_id], paths[q.gold_table_id]
            r, c = q.gold_operands[0].row, q.gold_operands[0].col
            fh.write(json.dumps({
                "query_id": q.query_id,
                "question": q.question,
                "answer": q.answer,
                "title": bt.title,
                "row_path": pt["gold_rp"][r],
                "col_path": pt["gold_cp"][c],
                "value": bt.data[r][c],
                "algo_path": cell_text(pt["gold_rp"][r], pt["gold_cp"][c], bt.data[r][c], "S2"),
                "algo_sent": verbalize_cell(bt, r, c, "long"),
                "manual": "",
            }, ensure_ascii=False) + "\n")
    print(f"wrote worksheet -> {out}  ({len(pop)} rows, fill in 'manual')")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--model", default="local:Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--export", default="", help="write the worksheet and stop")
    ap.add_argument("--manual", default="diag/manual_sentences.jsonl")
    ap.add_argument("--no-llm", action="store_true", help="retrieval only")
    ap.add_argument("--out", default="results/manual_sentence_ceiling.json")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force-resume", action="store_true",
                    help="append even though the records file was written under\n"
                         "a different reader/seed/population")
    args = ap.parse_args()

    env = run_env(0, "BAAI/bge-small-en-v1.5")
    pop, tables, paths = build_population(args.data_dir, args.split, args.n)
    print(f"[pop] {len(pop)} single-operand lookup queries (split={args.split})", flush=True)

    if args.export:
        export(pop, tables, paths, Path(args.export))
        return 0

    manual = {}
    for line in open(args.manual):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("manual", "").strip():
            manual[r["query_id"]] = r["manual"].strip()
    pop = [q for q in pop if q.query_id in manual]
    print(f"[manual] {len(manual)} hand-written sentences -> {len(pop)} scorable queries", flush=True)
    if not pop:
        print("no overlap between --manual and the population; nothing to score")
        return 1

    enc = default_encoder(model_name="BAAI/bge-small-en-v1.5")
    llm = None if args.no_llm else build_llm(args.model)

    rec_path = Path(str(Path(args.out).with_suffix("")) + "_records.jsonl")
    done = {}
    if args.resume and rec_path.exists():
        for line in open(rec_path):
            r = json.loads(line)
            done[(r["query_id"], r["arm"])] = r
        print(f"[resume] {len(done)} (query,arm) records already on disk", flush=True)

    guard_resume(rec_path, env, reader=(None if llm is None else llm.name),
                 population=POPULATION, force=args.force_resume)
    rec_fh = open(rec_path, "a")
    try:
        for n_done, q in enumerate(pop, 1):
            bt, pt = tables[q.gold_table_id], paths[q.gold_table_id]
            gr, gc = q.gold_operands[0].row, q.gold_operands[0].col

            # the fixed pool: every cell of the gold table under the production
            # serialization, gold cell included at index `gold_idx`
            units, gold_idx = [], -1
            for i in range(pt["n_r"]):
                for j in range(pt["n_c"]):
                    if i == gr and j == gc:
                        gold_idx = len(units)
                    units.append(cell_text(pt["gold_rp"][i], pt["gold_cp"][j],
                                           bt.data[i][j], "S2"))
            swap = {"algo_path": units[gold_idx],
                    "algo_sent": verbalize_cell(bt, gr, gc, "long"),
                    "manual": manual[q.query_id]}
            qv = np.asarray(enc.encode([q.question])[0])
            # the pool is identical across arms except for one row, so encode it
            # once and swap that single vector -- 3x fewer table encodes
            base_vecs = np.asarray(enc.encode(units))

            for arm in ARMS:
                if (q.query_id, arm) in done:
                    continue
                pool = list(units)
                pool[gold_idx] = swap[arm]          # only the gold cell's text moves
                vecs = base_vecs.copy()
                vecs[gold_idx] = np.asarray(enc.encode([swap[arm]])[0])
                order = np.argsort(-(vecs @ qv))
                rank = int(np.where(order == gold_idx)[0][0]) + 1
                rec = {"query_id": q.query_id, "arm": arm, "rank": rank,
                       "hit_at_1": rank == 1, "pool": len(pool), "text": swap[arm]}
                if llm is not None:
                    ctx = "\n".join(pool[o] for o in order[: args.topk])
                    user = f"ROWS:\n{ctx}\n\nQUESTION: {q.question}\n\nAnswer:"
                    raw = llm.complete(system=_DIRECT_SYS, user=user, max_tokens=512)
                    if not raw and llm.last_finish_reason == "length":
                        raw = llm.complete(system=_DIRECT_SYS, user=user, max_tokens=1024)
                    rec["correct"] = bool(hitab_exact_match(raw, q.answer))
                    rec["pred"] = raw[:120]
                done[(q.query_id, arm)] = rec
                rec_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                rec_fh.flush()
            if n_done % 10 == 0:
                shown = {a: sum(r.get("hit_at_1", False) for (_, x), r in done.items() if x == a)
                         for a in ARMS}
                print(f"  {n_done}/{len(pop)}  recall@1 hits {shown}", flush=True)
    except RuntimeError as e:
        print(f"[stopped] {str(e)[:160]} ... progress saved, rerun with --resume", flush=True)
    finally:
        rec_fh.close()

    def agg(arm, key):
        vals = [r[key] for (_, a), r in done.items() if a == arm and key in r]
        return (round(sum(vals) / len(vals), 4) if vals else None), len(vals)

    def col(arm, key):
        """Per-arm values keyed by query, over queries scored in EVERY arm."""
        return {q: r[key] for (q, a), r in done.items() if a == arm and key in r}

    complete = sorted(set.intersection(*[set(col(a, "rank")) for a in ARMS]))
    if any("correct" in r for r in done.values()):
        complete = sorted(set(complete) & set.intersection(
            *[set(col(a, "correct")) for a in ARMS]))

    paired = {}
    for key in ("rank", "correct"):
        vals = {a: col(a, key) for a in ARMS}
        if not all(vals[a] for a in ARMS):
            continue
        metric = "recall_at_1" if key == "rank" else "answer_em"
        for x in ARMS:
            for y in ARMS:
                if x >= y:
                    continue
                fx = [(vals[x][q] == 1) if key == "rank" else vals[x][q] for q in complete]
                fy = [(vals[y][q] == 1) if key == "rank" else vals[y][q] for q in complete]
                paired[f"{metric}:{x}_vs_{y}"] = mcnemar(fx, fy)

    out = {
        "experiment": "hand-written cell sentence vs template (ceiling diagnosis)",
        "population": f"HiTab {args.split}, single gold operand, seed-0 shuffle, n={len(pop)}",
        "table": "oracle (gate-1 solved: HiTab table recall@20=1.0)",
        "pool": "all cells of the gold table; only the GOLD cell's text differs by arm",
        "model": None if llm is None else llm.name,
        "topk": args.topk, "scorer": "hitab_exact_match", "env": env,
        "n_complete_all_arms": len(complete),
        "arms": {a: {"recall_at_1": agg(a, "hit_at_1")[0],
                     "n_retrieval": agg(a, "hit_at_1")[1],
                     # gold cell inside the window the reader actually sees
                     "recall_at_topk": (lambda v: round(sum(r <= args.topk for r in v) / len(v), 4)
                                        if v else None)(list(col(a, "rank").values())),
                     "median_rank": (lambda v: float(np.median(v)) if v else None)(
                         list(col(a, "rank").values())),
                     "answer_em": agg(a, "correct")[0],
                     # reader-only view: scored on the queries where the gold cell
                     # WAS in context, so retrieval loss is factored out
                     "answer_em_given_gold_in_context": (
                         lambda v: round(sum(v) / len(v), 4) if v else None)(
                         [c for q, c in col(a, "correct").items()
                          if col(a, "rank").get(q, 10 ** 9) <= args.topk]),
                     "n_answered": agg(a, "correct")[1]} for a in ARMS},
        "paired_tests": paired,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out["arms"], indent=2))
    print(f"wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
