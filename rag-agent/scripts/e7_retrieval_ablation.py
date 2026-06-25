#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""E7 — controlled retrieval ablation at a fixed solver (H5).

Design: `docs/e7_retrieval_ablation_design.md`. Hold the LLM solver and the prompt
format constant; vary **only the retrieved cell set**, so any difference in answer
accuracy is attributable to retrieval, not the model. This neutralizes the "you
used a bigger LLM than HiTab's 2022 BERT parser" objection.

Arms (each is a set of cells, rendered in the SAME E4 `(header-path = value)`
format and sent to the SAME LLM):
  dense_k5 / dense_k10 / dense_k20 : similarity top-k row-chunks (table-RAG baseline)
  enum_base                        : header-tree enumeration (hybrid resolver)
  enum_treated                     : enumeration + total-row + sibling augmentation
  whole_table                      : every numeric cell (the "give it everything" point)
  oracle                           : gold operand cells (ceiling = solver's own limit)

Metrics per arm: answer accuracy (numeric match), retrieval OSC, mean cells.
Paired McNemar + bootstrap CI of accuracy vs dense_k10. Population: HiTab dev
arithmetic m>=2.

Cost controls: identical (question, context) answers are cached (temp=0 -> stable),
and per-query records stream to a JSONL checkpoint so a rate-limit interruption
resumes instead of restarting.

Run (needs GROQ_API_KEY; reads rag-agent/.env if present):
    PYTHONPATH=. python scripts/e7_retrieval_ablation.py --split dev \
        --llm groq:llama-3.1-8b-instant
Dry run (no LLM; validates cell sets / OSC / context only):
    PYTHONPATH=. python scripts/e7_retrieval_ablation.py --split dev --dry-run --max 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_dotenv():
    """Minimal .env loader (no dependency): rag-agent/.env then repo-root/.env."""
    here = Path(__file__).resolve().parent.parent
    for env in (here / ".env", here.parent / ".env"):
        if env.is_file():
            for line in env.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

from rag_agent.bench.hitab import load_queries
from rag_agent.bench.schema import Chunk
from rag_agent.data.loader import load_table
from rag_agent.eval.operand_set import operand_set_completeness
from rag_agent.generate import answer as gen_answer, evaluate_answer
from rag_agent.eval.metrics import numeric_match, exact_match
from rag_agent.query.header_embed_resolver import EmbedResolver
from rag_agent.query.operand_decomposer import Embedder
from rag_agent.retrieve.header_enum import enumerate_scope, is_ratio_query
from rag_agent.retrieve.operand_retriever import HybridRetriever, retrieve
from rag_agent.serialize import S2, serialize_table
from rag_agent.stores.original_store import build_original_table

SEED = 42
ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}
DENSE_KS = (5, 10, 20)
# whole_table / dense_k20 excluded by default: on the free tier their context
# (~100-160 cells) exceeds the 6000 TPM single-request cap — itself evidence that
# "dump the whole table" is operationally infeasible. Add them explicitly if on a
# higher tier.
ALL_ARMS = ["dense_k5", "dense_k10", "enum_base", "enum_treated", "oracle"]


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def struct_chunk_from_cells(ot, cells, table_id) -> Chunk:
    """Render a cell set as E4 `(row-path > col-path = value)` lines (one per cell).

    Same format for every arm — only *which* cells appear differs.
    """
    lines = []
    for (r, c) in sorted(cells):
        path = list(ot.row_path(r)) + list(ot.col_path(c))
        head = " > ".join(s for s in path if s) or f"r{r}c{c}"
        lines.append(f"{head} = {ot.cell(r, c)}")
    return Chunk(table_id=table_id, chunk_id="ctx", text="\n".join(lines))


def numeric_cells(ot, rows, cols):
    return {(r, c) for r in rows for c in cols if ot.cell_num(r, c) is not None}


def ohd_serialize(ot, cells) -> str:
    """OHD-style whole-table serialization (the 2602.01969 baseline, approximated).

    Each cell rendered as `Context → Key → Value` lineage, presented in BOTH
    row-major and column-major orderings (OHD feeds both to its LLM arbitrator).
    We omit OHD's learned tree induction + the arbitrator selection — those affect
    representation *quality*, not the whole-table-vs-retrieval axis we test.
    """
    cells = sorted(cells)
    by_row, by_col = {}, {}
    for (r, c) in cells:
        by_row.setdefault(r, []).append(c)
        by_col.setdefault(c, []).append(r)

    def lineage_row(r):
        return " > ".join(s for s in ot.row_path(r) if s) or f"row{r}"

    def lineage_col(c):
        return " > ".join(s for s in ot.col_path(c) if s) or f"col{c}"

    lines = ["[row-major]"]
    for r in sorted(by_row):
        for c in sorted(by_row[r]):
            lines.append(f"{lineage_row(r)} | {lineage_col(c)} = {ot.cell(r, c)}")
    lines.append("[column-major]")
    for c in sorted(by_col):
        for r in sorted(by_col[c]):
            lines.append(f"{lineage_col(c)} | {lineage_row(r)} = {ot.cell(r, c)}")
    return "\n".join(lines)


def dense_cells(res, ot, k):
    """Numeric cells covered by the top-k dense chunks (the budget the LLM sees)."""
    out = set()
    for ch in res.retrieved[:k]:
        out |= numeric_cells(ot, ch.rows, ch.cols)
    return out


def bucket(ans, gold):
    if evaluate_answer(ans, gold):
        return "correct"
    if _is_number(ans) or numeric_match(ans, ans):
        return "silent_wrong"
    return "non_number"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--llm", default="groq:llama-3.1-8b-instant")
    ap.add_argument("--mode", default="codegen", choices=["codegen", "direct"])
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--arms", default=",".join(ALL_ARMS),
                    help="comma list from: " + ",".join(ALL_ARMS))
    ap.add_argument("--dry-run", action="store_true",
                    help="build cell sets / OSC / context but skip the LLM call")
    ap.add_argument("--max-ctx-tokens", type=int, default=4500,
                    help="skip (mark oversize) any context whose rough token "
                         "estimate exceeds this; free-tier TPM is 6000")
    ap.add_argument("--out", default="results/e7_retrieval_ablation.json")
    ap.add_argument("--checkpoint", default="results/e7_records.jsonl")
    ap.add_argument("--baseline", default="dense_k10",
                    help="arm to pair against for ΔAcc / CI / McNemar (E8: ohd_lite)")
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    queries, tables = load_queries(args.data_dir, args.split, args.max)
    pop = [q for q in queries
           if (q.aggregation or "none") in ARITH
           and len({(o.row, o.col) for o in q.gold_operands}) >= 2]
    print(f"[pop] arithmetic m>=2: {len(pop)}  arms={arms}  dry_run={args.dry_run}")

    embedder = Embedder(args.embed_model, device=args.device)
    resolver = EmbedResolver(embedder, row_mode="embed", col_mode="lexical")
    needed = {q.gold_table_id for q in pop}
    ots, retr = {}, {}
    for tid in needed:
        ots[tid] = build_original_table(load_table(tid, args.data_dir))
        retr[tid] = HybridRetriever(serialize_table(tables[tid], S2), embedder)

    llm = None
    if not args.dry_run:
        from rag_agent.llm.factory import build_llm
        llm = build_llm(args.llm)
        print(f"[llm] {args.llm} mode={args.mode}")

    # resume: skip query_ids already in the checkpoint
    done = {}
    cp = Path(args.checkpoint)
    if cp.exists() and not args.dry_run:
        for line in cp.read_text().splitlines():
            try:
                r = json.loads(line)
                done[r["query_id"]] = r
            except Exception:
                pass
        print(f"[resume] {len(done)} queries already in {cp}")

    ans_cache = {}  # sha1(question||context) -> answer  (temp=0 => stable)
    # Free-tier TPM (6000) caps a single request; skip contexts that won't fit so a
    # too-large arm (e.g. whole_table) is recorded as "oversize", not a crash. A
    # rough char/3 token estimate with headroom for the system prompt + reasoning.
    max_ctx_tokens = args.max_ctx_tokens

    def solve(question, ctx_text, gold_answer):
        if args.dry_run:
            return None, None
        if len(ctx_text) // 3 > max_ctx_tokens:
            return None, "oversize"
        key = hashlib.sha1((question + "␟" + ctx_text).encode()).hexdigest()
        if key not in ans_cache:
            try:
                res = gen_answer(question, [Chunk(table_id="t", chunk_id="ctx", text=ctx_text)],
                                 llm, mode=args.mode)
                ans_cache[key] = res.answer
            except Exception as e:  # rate-limit exhausted / oversize / transient
                return None, "error"
        ans = ans_cache[key]
        return ans, bucket(ans, gold_answer)

    recs = list(done.values())
    cp_fh = None if args.dry_run else open(cp, "a")
    try:
        for i, q in enumerate(pop):
            if q.query_id in done:
                continue
            ot = ots[q.gold_table_id]
            gold = q.gold_operands
            gold_cells = {(o.row, o.col) for o in gold}
            intent = resolver.resolve(q.question, ot)
            ratio = is_ratio_query(q.question)
            res_dense = retrieve(q.question, tables[q.gold_table_id], gold, mode="plain",
                                 k=max(DENSE_KS), scheme=S2, embedder=embedder,
                                 retriever=retr[q.gold_table_id])
            all_cells = numeric_cells(ot, range(ot.n_rows), range(ot.n_cols))

            def cells_for(arm):
                if arm == "enum_base":
                    return enumerate_scope(ot, intent.row_paths, intent.col_paths).cells
                if arm == "enum_treated":
                    return enumerate_scope(ot, intent.row_paths, intent.col_paths,
                                           add_total_rows=ratio, expand_siblings=True).cells
                if arm in ("whole_table", "ohd_lite"):  # both = the entire table
                    return set(all_cells)
                if arm == "oracle":
                    return set(gold_cells)
                if arm.startswith("dense_k"):
                    return dense_cells(res_dense, ot, int(arm[len("dense_k"):]))
                raise ValueError(arm)

            rec = {"query_id": q.query_id, "m": len(gold_cells),
                   "aggregation": q.aggregation, "ratio": int(ratio)}
            for arm in arms:
                cset = cells_for(arm)
                # ohd_lite uses OHD's whole-table dual serialization; all other
                # arms use the identical (header-path = value) format so only the
                # *cell content* differs.
                ctx = (ohd_serialize(ot, cset) if arm == "ohd_lite"
                       else struct_chunk_from_cells(ot, cset, q.gold_table_id).text)
                ans, bk = solve(q.question, ctx, q.answer)
                rec[f"{arm}_osc"] = operand_set_completeness(gold, cset)
                rec[f"{arm}_cells"] = len(cset)
                if not args.dry_run:
                    rec[f"{arm}_correct"] = int(bk == "correct")          # numeric-match
                    rec[f"{arm}_em"] = int(exact_match(ans, q.answer))    # exact-match
                    rec[f"{arm}_bucket"] = bk
            recs.append(rec)
            if cp_fh:
                cp_fh.write(json.dumps(rec) + "\n"); cp_fh.flush()
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(pop)}  (cache={len(ans_cache)})")
    finally:
        if cp_fh:
            cp_fh.close()

    # ---------- aggregate ----------
    n = len(recs)
    rng = random.Random(SEED)
    base = args.baseline

    def rate(arm, field):
        return sum(r.get(f"{arm}_{field}", 0) for r in recs) / n if n else 0.0

    def boot_ci(arm, field):
        diffs = []
        for _ in range(2000):
            s = 0
            for _ in range(n):
                r = recs[rng.randrange(n)]
                s += r.get(f"{arm}_{field}", 0) - r.get(f"{base}_{field}", 0)
            diffs.append(s / n)
        diffs.sort()
        return [round(diffs[50], 4), round(diffs[1949], 4)]

    out = {"experiment": "E7_retrieval_ablation",
           "hypothesis": "H5/H6: at a fixed solver, retrieval (small+complete) vs whole-table/dense; gain tracks precision",
           "split": args.split, "seed": SEED, "llm": args.llm, "mode": args.mode,
           "baseline_arm": base,
           "dry_run": args.dry_run, "population": {"name": "arithmetic_m>=2", "n": n},
           "arms": {}}
    for arm in arms:
        a = {"osc": round(sum(r.get(f"{arm}_osc", 0) for r in recs) / n, 4) if n else 0,
             "mean_cells": round(sum(r.get(f"{arm}_cells", 0) for r in recs) / n, 1) if n else 0}
        if not args.dry_run:
            a["accuracy_nm"] = round(rate(arm, "correct"), 4)   # numeric-match
            a["accuracy_em"] = round(rate(arm, "em"), 4)        # exact-match
            a["n_oversize"] = sum(1 for r in recs if r.get(f"{arm}_bucket") == "oversize")
            a["n_error"] = sum(1 for r in recs if r.get(f"{arm}_bucket") == "error")
            if arm != base:
                a[f"delta_nm_vs_{base}"] = round(rate(arm, "correct") - rate(base, "correct"), 4)
                a[f"delta_nm_ci95_vs_{base}"] = boot_ci(arm, "correct")
                a[f"delta_em_vs_{base}"] = round(rate(arm, "em") - rate(base, "em"), 4)
                a[f"mcnemar_nm_vs_{base}"] = {
                    "arm_only": sum(1 for r in recs if r.get(f"{arm}_correct") and not r.get(f"{base}_correct")),
                    "base_only": sum(1 for r in recs if r.get(f"{base}_correct") and not r.get(f"{arm}_correct"))}
        out["arms"][arm] = a

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    hdr = f"{'arm':<14}{'OSC':>7}{'cells':>7}" + (
        "" if args.dry_run else f"{'NM':>7}{'EM':>7}{'ΔNM':>9}{'ovsz':>6}")
    print("\n" + hdr + f"   (baseline={base})")
    for arm in arms:
        a = out["arms"][arm]
        row = f"{arm:<14}{a['osc']:>7.3f}{a['mean_cells']:>7.1f}"
        if not args.dry_run:
            dv = a.get(f"delta_nm_vs_{base}")
            row += f"{a['accuracy_nm']:>7.3f}{a['accuracy_em']:>7.3f}"
            row += (f"{dv:>+9.3f}" if dv is not None else f"{'--':>9}") + f"{a['n_oversize']:>6}"
        print(row)
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
