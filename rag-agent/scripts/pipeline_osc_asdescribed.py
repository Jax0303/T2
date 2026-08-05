#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""OSC measured through the pipeline the disclosure actually DESCRIBES.

Every other OSC script in this repo measures a second, parallel stack:
``operand_retriever.HybridRetriever`` over ``serialize_table(BenchTable, S2)``
ROW chunks. The stack the invention disclosure describes — S3 caption sentences
as the index unit, index-time structural attribution, the structural complement,
and the all-or-nothing completeness gate — was only ever exercised by unit
tests. So "the system we describe" and "the code that produced the numbers"
were not the same system. This script closes that gap: it runs
:class:`OperandTargetedRetriever` (+ :func:`retrieve_with_gate`) over the SAME
population and reports the SAME metric, so the two can be quoted side by side.

It does NOT restate the other scripts' numbers and is not a replacement for
them — the serializations, the chunk granularity and the fusion all differ, so
the two stacks' OSC values are not interchangeable. Read this one as "what the
described pipeline scores", not as a correction of anything.

Population: HiTab arithmetic aggregations with resolved gold operands, same as
``e1_osc_baseline.py``; m>=2 is the primary population, m==1 the anchor.

Arms:
  gate_off  — OperandTargetedRetriever.retrieve(k)
  gate_on   — retrieve_with_gate(...): escalates budget -> structural -> enumerate
              while the decomposed operand set is judged incomplete

LLM-free (decomposition and enumeration both run their no-LLM path), so this
costs no API tokens.

Run:
    PYTHONPATH=. python3 scripts/pipeline_osc_asdescribed.py --split dev
    PYTHONPATH=. python3 scripts/pipeline_osc_asdescribed.py --max 50   # quick
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.bench.hitab import load_queries
from rag_agent.data.loader import load_table
from rag_agent.eval.operand_set import operand_set_completeness, per_cell_recall
from rag_agent.retrieve.completeness_gate import retrieve_with_gate
from rag_agent.retrieve.encoders import default_encoder
from rag_agent.retrieve.operand_retrieval import OperandTargetedRetriever
from rag_agent.runenv import run_env
from rag_agent.stores.original_store import build_original_table

ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}
K_LIST = (1, 3, 5, 10, 20)


def retrieved_cells(result) -> list:
    """(row, col) of every retrieved index unit, best-scored first.

    Safe because HiTab's ``BenchTable`` is built straight from the
    ``OriginalTable`` and shares its ``data`` (``bench/hitab.py``
    ``_bench_table_from_original``), so a chunk's ``row_index``/``col_index``
    live in the same data space as ``GoldOperand.row``/``.col``.
    """
    return [(rc.chunk.row_index, rc.chunk.col_index) for rc in result.retrieved
            if rc.chunk.row_index is not None and rc.chunk.col_index is not None]


def check_coordinate_bridge(retriever, table, gold_operands) -> None:
    """Fail loudly if the two coordinate spaces ever diverge.

    A silent mismatch here would not crash — it would just score OSC against
    the wrong cells and report a plausible, wrong number. So assert instead:
    every gold operand cell must have an index unit at those coordinates.
    """
    index = retriever.index_table(table)
    have = {(ch.row_index, ch.col_index) for ch in index.chunks}
    for op in gold_operands:
        assert (op.row, op.col) in have, (
            f"coordinate bridge broken on {table.table_id}: gold cell "
            f"({op.row},{op.col}) has no index unit "
            f"(table is {table.n_rows}x{table.n_cols})"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max", type=int, default=0, help="cap population (0 = all)")
    ap.add_argument("--k", type=int, default=5, help="base k for the gated arm")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--out", default="results/pipeline_osc_asdescribed.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    env = run_env(args.seed, args.embed_model)

    queries, _bench_tables = load_queries(args.data_dir, args.split)
    pop = [q for q in queries
           if (q.aggregation or "none") in ARITH and len(q.gold_operands) >= 1]
    if args.max:
        pop = pop[: args.max]
    n_m2 = sum(1 for q in pop if len(q.gold_operands) >= 2)
    print(f"[pop] arithmetic w/ operands: {len(pop)} (m>=2: {n_m2})", flush=True)

    enc = default_encoder(model_name=args.embed_model)
    retriever = OperandTargetedRetriever(encoder=enc, scheme="S3",
                                         caption_length="long")
    print(f"[pipeline] OperandTargetedRetriever scheme={retriever.scheme} "
          f"caption_length={retriever.caption_length} fusion={retriever.fusion}",
          flush=True)

    ots: dict = {}
    recs, gate_recs = [], []
    bridge_checked = False
    for n, q in enumerate(pop, 1):
        tid = q.gold_table_id
        if tid not in ots:
            try:
                ots[tid] = build_original_table(load_table(tid, args.data_dir))
            except Exception:
                ots[tid] = None
        ot = ots[tid]
        if ot is None:
            continue
        if not bridge_checked:                    # once is enough to catch a
            check_coordinate_bridge(retriever, ot, q.gold_operands)  # space shift
            bridge_checked = True

        m = len({(o.row, o.col) for o in q.gold_operands})

        # --- arm 1: no gate, OSC across budgets -------------------------
        res = retriever.retrieve(q.question, ot, k=max(K_LIST))
        cells = retrieved_cells(res)
        recs.append({"m": m, "osc": {k: operand_set_completeness(q.gold_operands, cells[:k])
                                     for k in K_LIST},
                     "pcr": {k: per_cell_recall(q.gold_operands, cells[:k])
                             for k in K_LIST}})

        # --- arm 2: the completeness gate at base k ---------------------
        # The ungated control MUST be the same retrieval call the gate starts
        # from — `cells[:args.k]` is the top-k of a k=20 union, a different
        # evidence set entirely, and differencing against it would credit the
        # gate for a budget change it never made.
        ungated = retriever.retrieve(q.question, ot, k=args.k)
        gres, trace = retrieve_with_gate(retriever, q.question, ot, k=args.k)
        gcells = retrieved_cells(gres)
        gate_recs.append({
            "m": m,
            "osc_gated": operand_set_completeness(q.gold_operands, gcells),
            "osc_initial": operand_set_completeness(q.gold_operands,
                                                    retrieved_cells(ungated)),
            "fired": trace.fired, "resolved_at": trace.resolved_at,
            "n_injected": gres.n_injected, "n_units": len(gcells),
            # The gate judges the DECOMPOSED operand set, not the gold one, so
            # n_required < m means it can pass while OSC still fails. Recorded
            # because otherwise "gate never fired, OSC low" reads as a bug.
            "n_required": trace.attempts[0].verdict.n_required,
        })
        if n % 50 == 0:
            print(f"  {n}/{len(pop)}", flush=True)

    def osc_at(k, only_m2=False):
        rs = [r for r in recs if not only_m2 or r["m"] >= 2]
        return round(sum(r["osc"][k] for r in rs) / len(rs), 4) if rs else None

    def pcr_at(k, only_m2=False):
        rs = [r for r in recs if not only_m2 or r["m"] >= 2]
        return round(sum(r["pcr"][k] for r in rs) / len(rs), 4) if rs else None

    g2 = [r for r in gate_recs if r["m"] >= 2]
    fired = [r for r in gate_recs if r["fired"]]
    out = {
        "env": env,
        "pipeline": "AS DESCRIBED — OperandTargetedRetriever(S3 caption cells) "
                    "+ index-time structural attribution + completeness gate",
        "note": "NOT comparable to the HybridRetriever/S2-row-chunk OSC scripts "
                "(different index unit, granularity and fusion). Reported so the "
                "described system has a measurement of its own.",
        "population": {"name": "hitab_arith_with_operands", "split": args.split,
                       "n": len(recs), "n_m_ge_2": sum(1 for r in recs if r["m"] >= 2)},
        "index_unit": {"scheme": retriever.scheme, "length": retriever.caption_length,
                       "granularity": "cell", "fusion": retriever.fusion},
        "osc_by_budget": {str(k): osc_at(k) for k in K_LIST},
        "osc_by_budget_m_ge_2": {str(k): osc_at(k, True) for k in K_LIST},
        "per_cell_recall_by_budget": {str(k): pcr_at(k) for k in K_LIST},
        "completeness_gate": {
            "base_k": args.k,
            "n": len(gate_recs),
            "osc_initial": round(sum(r["osc_initial"] for r in gate_recs) / len(gate_recs), 4)
                           if gate_recs else None,
            "osc_after_gate": round(sum(r["osc_gated"] for r in gate_recs) / len(gate_recs), 4)
                              if gate_recs else None,
            "osc_after_gate_m_ge_2": round(sum(r["osc_gated"] for r in g2) / len(g2), 4)
                                     if g2 else None,
            "n_fired": len(fired),
            "gate_delta": round(
                (sum(r["osc_gated"] for r in gate_recs)
                 - sum(r["osc_initial"] for r in gate_recs)) / len(gate_recs), 4)
                if gate_recs else 0.0,
            "resolved_at": {s: sum(1 for r in fired if r["resolved_at"] == s)
                            for s in ("initial", "budget", "structural", "enumerate", None)},
            "mean_injected_when_fired": round(
                sum(r["n_injected"] for r in fired) / len(fired), 2) if fired else 0.0,
            # --- what the gate could even see -------------------------------
            "mean_operands_required": round(
                sum(r["n_required"] for r in gate_recs) / len(gate_recs), 2)
                if gate_recs else 0.0,
            "mean_gold_operands": round(
                sum(r["m"] for r in gate_recs) / len(gate_recs), 2) if gate_recs else 0.0,
            "n_zero_operand_vacuous": sum(1 for r in gate_recs if r["n_required"] == 0),
            "ceiling_note": "The gate is gold-free: it judges whether the "
                            "operands the DECOMPOSER asked for were found, not "
                            "whether they were the right ones. So n_fired=0 "
                            "alongside OSC<1 is not a contradiction and not a "
                            "bug: retrieval did fetch every requested operand "
                            "(note mean_operands_required is not below "
                            "mean_gold_operands here — the decomposer asks for "
                            "MORE, just partly for the wrong header paths). The "
                            "gate bounds retrieval failures; on this population "
                            "the binding constraint is decomposition ACCURACY, "
                            "which no escalation rung can see or repair.",
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
