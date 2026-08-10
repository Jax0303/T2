#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Does operand targeting's budget-matched loss hold on a REAL multi-table corpus?

`inject_osc_matched.py` (one HiTab table, ~207 cells) and `pool_scale_matched.py`
(the same queries against 82 and 882 HiTab tables) both found plain
whole-question similarity beating operand-targeted retrieval at an equal cell
budget, the gap shrinking monotonically as the pool grows (-.180 -> -.062 at
k=5). That sweep names pool size as the variable and answers it *inside HiTab*:
same tables, HiTab's own gold header trees, and the bigger pools assembled by
adding tables no query asks about.

This is the external-validity leg. Same two arms, same per-query budget-matched
control, same index unit (S3 structural cell sentence) and resolver (embed), but
on MultiHiertt: documents that really do hold several tables each, and header
trees this repo RECONSTRUCTS from raw HTML instead of reading from a gold
annotation. Population, corpus construction and gold come from
`operand_collision_multihiertt.py` unchanged, so n and the pool match every
other MultiHiertt claim in the repo.

CONFIRMATION, not decision. The pool-size sweep already established the shape,
so a null here reproduces a known result on new data. What it can newly rule out
is that the HiTab loss was an artifact of gold trees or of synthetic pool
inflation -- and what it can newly show is a sign flip, which HiTab hinted at
(k=20, 128k cells: +.050, p=.28) without significance.

Both arms are handed the same table gate (operands are resolved against the gold
table) and cells from other tables cannot cover a gold operand, so this measures
the cell-retrieval stage alone.

Run:
    PYTHONPATH=. .venv/bin/python scripts/pool_scale_matched_multihiertt.py \
        --max-queries 300
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_agent.eval.operand_set import operand_set_completeness, per_cell_recall
from rag_agent.query.header_embed_resolver import EmbedResolver
from rag_agent.retrieve.encoders import default_encoder
from rag_agent.retrieve.hybrid_index import HybridIndex
from rag_agent.retrieve.operand_retrieval import decompose_operands
from rag_agent.runenv import run_env
from rag_agent.serialization.base import Chunk
from rag_agent.serialization.caption import caption_sentence
from rag_agent.stores.original_store import OriginalTable

from operand_collision_multihiertt import build_corpus, load_population


def mcnemar_p(b: int, c: int) -> float:
    from scipy.stats import binomtest
    n = b + c
    return binomtest(b, n, 0.5).pvalue if n else 1.0


def as_original(key, t) -> OriginalTable:
    """The reconstructed table as an :class:`OriginalTable`, for the resolver.

    Only the header paths matter here -- this object is used for decomposition,
    never for scoring, so its data region is indexed from the data origin while
    gold and retrieval keep the grid coordinates they came with.
    """
    grid, nhr, nhc = t["grid"], t["nhr"], t["nhc"]
    width = len(grid[0])
    data = [[grid[r][c] for c in range(nhc, width)] for r in range(nhr, len(grid))]
    return OriginalTable(table_id=str(key), title="", data=data,
                         top_paths=t["cols"], left_paths=t["rows"])


def own_cells(chunks, table_id: str):
    """(row, col) of the chunks from ``table_id``. Cells of other tables spend
    budget but can never cover a gold operand, so they are dropped only AFTER
    the budget cut -- otherwise the control gets a free table gate."""
    return [(ch.row_index, ch.col_index) for ch in chunks
            if ch.table_id == table_id and ch.row_index is not None]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-queries", type=int, default=300)
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--deep", type=int, default=400,
                    help="plain ranked pool, cut per query to the targeted arm's size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/pool_scale_matched_multihiertt.json")
    ap.add_argument("--records", default="")
    args = ap.parse_args()
    env = run_env(args.seed, args.embed_model)
    t_start = time.time()

    queries, docs = load_population(args.max_queries)
    tables, cells, kept = build_corpus(queries, docs)
    print(f"[pop] multihiertt arith_multi: {len(kept)} queries "
          f"over {len(tables)} tables / {len(cells)} cells", flush=True)

    enc = default_encoder(model_name=args.embed_model)

    # Corpus index: one S3 structural sentence per data cell, rendered by the
    # same caption_sentence the HiTab leg indexes, so the only things that
    # changed from that leg are the dataset and the pool.
    chunks = [
        Chunk(table_id=str(cell["table"]),
              chunk_id=f"{cell['table']}::r{cell['r']}c{cell['c']}",
              text=caption_sentence("", cell["row_path"], cell["col_path"],
                                    value=cell["value"]),
              scheme="S3", kind="cell",
              row_index=cell["r"], col_index=cell["c"],
              header_paths=[list(cell["row_path"]) + list(cell["col_path"])])
        for cell in cells
    ]
    print(f"[index] embedding {len(chunks)} cells ...", flush=True)
    corpus = HybridIndex(chunks, encoder=enc)
    print(f"[index] built in {time.time() - t_start:.0f}s "
          f"(vector backend: {corpus.vector_backend})", flush=True)

    resolver = EmbedResolver(enc)
    kmax = max(args.ks)
    recs = []
    for n, q in enumerate(kept, 1):
        key = q["table"]
        t = tables.get(key)
        if t is None:
            continue
        tid = str(key)
        ops = decompose_operands(q["question"], as_original(key, t), encoder=resolver)

        # one search per operand at the widest k, sliced per budget below
        per_op = [corpus.search(op.query_text, k=kmax) for op in ops]
        deep = [h.chunk for h in corpus.search(q["question"], k=args.deep)]

        rec = {"uid": q["uid"], "m": len(set(q["cells"])), "n_ops": len(ops)}
        for k in args.ks:
            seen, tgt = set(), []
            for hits in per_op:
                for h in hits[:k]:
                    if h.chunk.chunk_id not in seen:
                        seen.add(h.chunk.chunk_id)
                        tgt.append(h.chunk)
            budget = len(tgt)
            tgt_cells = own_cells(tgt, tid)
            plain_cells = own_cells(deep[:budget], tid)
            # Second control, table-gate matched. The targeted arm's subqueries
            # ARE the gold table's header text, so they pull its own cells and
            # spend far less budget on other tables than one whole-question
            # ranking does. That is a localization advantage, not a cell-ranking
            # one, and the arm above cannot tell the two apart. Here plain
            # similarity is given the same table gate and the SAME number of
            # own-table cells, which leaves only the ranking of cells inside the
            # right table as the difference.
            plain_pool = own_cells(deep, tid)
            gate_cells = plain_pool[: len(tgt_cells)]
            rec[k] = {
                "tgt": operand_set_completeness(q["cells"], tgt_cells),
                "plain": operand_set_completeness(q["cells"], plain_cells),
                "plain_gate": operand_set_completeness(q["cells"], gate_cells),
                "pcr_tgt": per_cell_recall(q["cells"], tgt_cells),
                "pcr_plain": per_cell_recall(q["cells"], plain_cells),
                "pcr_plain_gate": per_cell_recall(q["cells"], gate_cells),
                "budget": budget,
                # the control cannot be given more cells than the deep pool holds
                "control_truncated": int(budget > len(deep)),
                "n_tgt_own_table": len(tgt_cells),
                "n_plain_own_table": len(plain_cells),
                "n_gate_own_table": len(gate_cells),
                "gate_truncated": int(len(plain_pool) < len(tgt_cells)),
            }
        recs.append(rec)
        if n % 25 == 0:
            print(f"  {n}/{len(kept)}  ({time.time() - t_start:.0f}s)", flush=True)

    def block(rs):
        out = {}
        for k in args.ks:
            m_ = lambda f: round(mean(r_[k][f] for r_ in rs), 4)
            pair = lambda ctl: (sum(1 for r_ in rs if r_[k]["tgt"] > r_[k][ctl]),
                                sum(1 for r_ in rs if r_[k][ctl] > r_[k]["tgt"]))
            b, c = pair("plain")
            bg, cg = pair("plain_gate")
            out[str(k)] = {
                "osc_targeted": m_("tgt"), "osc_plain_matched": m_("plain"),
                "osc_plain_gate_matched": m_("plain_gate"),
                "delta": round(m_("tgt") - m_("plain"), 4),
                "delta_vs_gate_matched": round(m_("tgt") - m_("plain_gate"), 4),
                "pcr_targeted": m_("pcr_tgt"), "pcr_plain": m_("pcr_plain"),
                "pcr_plain_gate": m_("pcr_plain_gate"),
                "budget_cells": m_("budget"),
                "own_table_cells_targeted": m_("n_tgt_own_table"),
                "own_table_cells_plain": m_("n_plain_own_table"),
                "own_table_cells_gate": m_("n_gate_own_table"),
                "n_control_truncated": sum(r_[k]["control_truncated"] for r_ in rs),
                "n_gate_truncated": sum(r_[k]["gate_truncated"] for r_ in rs),
                "b_targeted_only": b, "c_plain_only": c,
                "p_mcnemar": round(mcnemar_p(b, c), 6),
                "b_targeted_only_vs_gate": bg, "c_gate_only": cg,
                "p_mcnemar_vs_gate": round(mcnemar_p(bg, cg), 6),
            }
        return out

    out = {
        "env": env,
        "leg": "operand-targeted vs plain similarity at EQUAL cell budget, on the "
               "MultiHiertt multi-table corpus with RECONSTRUCTED header trees. "
               "External-validity companion to pool_scale_matched.py (HiTab).",
        "population": {"name": "multihiertt_arith_multi", "split": "train",
                       "n": len(recs), "requested": args.max_queries,
                       "pool_cells": len(cells), "pool_tables": len(tables),
                       "docs": len(docs)},
        "index_unit": {"scheme": "S3", "template": "structural",
                       "granularity": "cell", "resolver": "embed"},
        "control": "per-query budget-matched: one whole-question ranked list over "
                   "the same corpus index, cut to the targeted arm's chunk count "
                   "(deep pool k=%d)" % args.deep,
        "all": block(recs),
        "note": "Every gold operand here has m>=2 by construction (the population "
                "requires >=2 evidence cells), so there is no m_ge_2 subgroup to "
                "split out. Both arms receive the table gate; only the cell "
                "ranking differs. A truncated control is disfavoured, so a "
                "negative delta is a conservative lower bound.",
        "runtime_s": round(time.time() - t_start, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    if args.records:
        Path(args.records).write_text(
            "\n".join(json.dumps(r) for r in recs) + "\n")
    print(json.dumps(out["all"], indent=2))
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
