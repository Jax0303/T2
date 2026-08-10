#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Does the table title in the cell sentence earn anything?

The S3 STRUCTURAL template opens every index unit with the table's own title --
"In the table '{title}', among {row_path}, the value of {col_path} is {value}."
MT2Net's published cell sentence (Zhao et al. 2022, arXiv:2206.01347 §4) carries
the two header paths and the value but no table-level context, so the title is a
real difference in construction. Whether it is a difference in RETRIEVAL is a
separate question, and this is that question.

Isolated: the only thing that changes between arms is `include_title`. Same
tables, same cells, same encoder, same query, same k. The arm searched is the one
the pipeline actually uses after the gate reassignment -- one whole-question
similarity ranking (see pool_scale_matched_multihiertt.py) -- so the number
answers "does the title help the retriever we ship".

Two pool scopes, because a table-level phrase can only discriminate when there is
more than one table to discriminate between:
  * within-table  -- the query's own table only. The title is constant across
    every candidate, so it is pure noise here and can only hurt.
  * corpus        -- every table of the population in one index. Now the title
    varies across candidates and could act as a table gate.

Run:
    PYTHONPATH=. .venv/bin/python scripts/caption_title_ablation.py --split dev
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.bench.hitab import load_queries
from rag_agent.data.loader import load_table
from rag_agent.eval.operand_set import operand_set_completeness, per_cell_recall
from rag_agent.retrieve.encoders import default_encoder
from rag_agent.retrieve.hybrid_index import HybridIndex
from rag_agent.runenv import run_env
from rag_agent.serialization import caption as s3
from rag_agent.stores.original_store import build_original_table

ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}


def mcnemar_p(b: int, c: int) -> float:
    from scipy.stats import binomtest
    n = b + c
    return binomtest(b, n, 0.5).pvalue if n else 1.0


def cells_of(chunks, table_id):
    return [(c.row_index, c.col_index) for c in chunks
            if c.table_id == table_id and c.row_index is not None]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    env = run_env(args.seed, args.embed_model)
    out_path = args.out or f"results/caption_title_ablation_{args.split}.json"

    queries, _ = load_queries(args.data_dir, args.split)
    pop = [q for q in queries
           if (q.aggregation or "none") in ARITH and len(q.gold_operands) >= 1]
    if args.max:
        pop = pop[: args.max]

    ots = {}
    for q in pop:
        if q.gold_table_id not in ots:
            try:
                ots[q.gold_table_id] = build_original_table(
                    load_table(q.gold_table_id, args.data_dir))
            except Exception:
                ots[q.gold_table_id] = None
    pop = [q for q in pop if ots.get(q.gold_table_id) is not None]
    n_titled = sum(1 for o in ots.values() if o is not None and str(o.title).strip())
    print(f"[pop] hitab {args.split}: {len(pop)} queries over {len(ots)} tables "
          f"({n_titled} of them carry a non-empty title)", flush=True)

    enc = default_encoder(model_name=args.embed_model)

    def chunks_for(ot, with_title):
        return s3.serialize(ot, granularity="cell", template="structural",
                            include_title=with_title)

    # corpus indexes: one per arm, every table of the population
    corp = {}
    for with_title in (True, False):
        ch = [c for ot in ots.values() if ot is not None
              for c in chunks_for(ot, with_title)]
        corp[with_title] = HybridIndex(ch, encoder=enc)
        print(f"[corpus] title={with_title}: {len(ch)} cells", flush=True)

    recs = []
    for n, q in enumerate(pop, 1):
        ot = ots[q.gold_table_id]
        tid = ot.table_id
        gold = q.gold_operands
        rec = {"m": len({(o.row, o.col) for o in gold}),
               "titled": bool(str(ot.title).strip())}
        # within-table indexes are cheap and per-query disposable
        loc = {}
        for wt in (True, False):
            idx = HybridIndex(chunks_for(ot, wt), encoder=enc)
            loc[wt] = idx
        for k in args.ks:
            e = {}
            for scope, src in (("table", loc), ("corpus", corp)):
                for wt in (True, False):
                    hits = [h.chunk for h in src[wt].search(q.question, k=k)]
                    cells = cells_of(hits, tid)
                    tag = f"{scope}_{'title' if wt else 'notitle'}"
                    e[tag] = operand_set_completeness(gold, cells)
                    e["pcr_" + tag] = per_cell_recall(gold, cells)
                    e["own_" + tag] = len(cells)
            rec[k] = e
        for idx in loc.values():
            idx.close()
        recs.append(rec)
        if n % 25 == 0:
            print(f"  {n}/{len(pop)}", flush=True)

    def block(rs):
        out = {}
        for k in args.ks:
            m_ = lambda f: round(mean(r_[k][f] for r_ in rs), 4)
            row = {}
            for scope in ("table", "corpus"):
                t, nt = f"{scope}_title", f"{scope}_notitle"
                b = sum(1 for r_ in rs if r_[k][t] > r_[k][nt])
                c = sum(1 for r_ in rs if r_[k][nt] > r_[k][t])
                row[scope] = {
                    "osc_with_title": m_(t), "osc_without_title": m_(nt),
                    "delta": round(m_(t) - m_(nt), 4),
                    "pcr_with_title": m_("pcr_" + t),
                    "pcr_without_title": m_("pcr_" + nt),
                    "own_table_cells_with_title": m_("own_" + t),
                    "own_table_cells_without_title": m_("own_" + nt),
                    "b_title_only": b, "c_notitle_only": c,
                    "p_mcnemar": round(mcnemar_p(b, c), 6),
                }
            out[str(k)] = row
        return out

    m2 = [r for r in recs if r["m"] >= 2]
    titled = [r for r in recs if r["titled"]]
    out = {
        "env": env,
        "leg": "does the table title inside the S3 cell sentence change retrieval? "
               "include_title=True vs False, everything else identical",
        "arm": "one whole-question similarity ranking (the post-reassignment "
               "cell-retrieval arm), hybrid alpha=0.5",
        "population": {"name": "hitab_arith_with_operands", "split": args.split,
                       "n": len(recs), "n_m_ge_2": len(m2),
                       "n_titled_queries": len(titled),
                       "tables": len(ots)},
        "scopes": {
            "table": "candidates = the query's own table only; the title is "
                     "constant across candidates, so it can only add noise",
            "corpus": "candidates = every table of the population in one index; "
                      "the title varies and could act as a table gate",
        },
        "all": block(recs),
        "m_ge_2": block(m2) if m2 else {},
        "titled_only": block(titled) if titled else {},
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["m_ge_2"] or out["all"], indent=2))
    print(f"\nwrote -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
