#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""External-validity leg of the 2-dataset strategy: answer accuracy on RealHiTBench.

`scripts/answer_accuracy_injection.py` measured the total-row-injection payoff on
HiTab, where header trees are GOLD (dataset-provided). This is the same paired
experiment on RealHiTBench (Zhang et al., ACL 2025; HF `spzy/RealHiTBench`), where
tables are genuinely RAW PhpSpreadsheet HTML and the header trees the pipeline
retrieves over are RECONSTRUCTED by our markup front-end
(`parse_html_table_with_merges` + `reconstruct_paths_with_merges` with guessed
header boundaries) — i.e. the full "arbitrary raw table" pipeline end to end:

  raw HTML -> reconstruct tree -> S2 cell-sentence chunks -> dense top-k
    baseline  = top-k row-chunks                      -> solver -> answer
    treatment = top-k UNION total-like row-chunks     -> solver -> answer

Differences from the HiTab run, and why:
  * NO OSC. RealHiTBench has no operand-cell annotations (gold answers only), so
    operand-set completeness cannot be computed. We record the injected-cell
    count and score ANSWER accuracy only (the metric this dataset is for).
  * Ordering: no OSC-flip ordering is possible; `--injected-first` runs queries
    where the treatment actually adds chunks first, so a daily-quota cutoff
    still yields discordance-informative pairs. Same caveat as HiTab's
    --flips-first: the partial-n headline Δ over-represents injection-active
    queries; complete the population before quoting an effect size.
  * Scoring: gold = `ProcessedAnswer` (numeric string throughout the
    aggregation subset, e.g. "543", "14.44%", "63.90"). `strict` = `em_norm`:
    strip `%`/commas from the gold, numeric equality at rel_tol 1e-3 — NO
    scale (x100), sign, or +-2% leniency. NOT RealHiTBench's own LLM-judge
    protocol; do not compare against the paper's leaderboard numbers. The
    first pilot (n=14, 2026-07-23) showed hitab_exact_match is unusable here:
    golds are percent-SCALE strings while the HiTab prompt's ratio rule makes
    the solver emit fractions — so this script also OVERRIDES the ratio rule
    to ask for percent-scale numbers, matching the gold convention. `lenient`
    = numeric_match (±2%, %/fraction scale) — internal diagnostic only.
  * Context budget: RealHiTBench rows are wide (~400 tokens each); the 4096
    default truncated 7/14 treat contexts in the pilot, silently dropping the
    injected chunks. Budget is 8192 here and injected chunks are capped at
    `--max-inject` (dense-rank order, best first) so truncation stays rare;
    per-query truncation is still recorded.
  * Population: SubQType in {Calculation, Multi-hop Numerical Reasoning} = 334
    queries / 258 tables (HiTab arith m>=2 analogue). `--sample N --seed S`
    fixes a CompStrucCata-stratified random subpopulation up front when the
    full 334 is quota-infeasible; the sample is deterministic, so --resume
    continues the SAME subpopulation across days.

Run (needs GROQ_API_KEY):
  PYTHONPATH=. python3 scripts/realhitbench_answer_accuracy.py \
    --solver-model llama-3.3-70b-versatile --codegen-max-tokens 160 \
    --sample 100 --injected-first --resume
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import rag_agent.generate.answerer as answerer_mod
from rag_agent.bench.schema import BenchTable
from rag_agent.generate.answerer import answer, evaluate_answer
from rag_agent.llm.groq_llm import GroqLLM
from rag_agent.query.operand_decomposer import Embedder
from rag_agent.reconstruct import (guess_n_header_cols, guess_n_header_rows,
                                   parse_html_table_with_merges,
                                   reconstruct_paths_with_merges)
from rag_agent.retrieve.header_enum import total_like_rows_hybrid
from rag_agent.retrieve.operand_retriever import HybridRetriever
from rag_agent.serialize import S2, serialize_table
from rag_agent.stores.original_store import _to_float

HF_REPO = "spzy/RealHiTBench"
AGG_SUBQTYPES = {"Calculation", "Multi-hop Numerical Reasoning"}
BenchTable.cell_num = lambda self, r, c: _to_float(self.cell(r, c))  # type: ignore[attr-defined]

# RealHiTBench golds are percent-SCALE ("14.44%", "63.90"), unlike the fraction
# convention the shared answerer prompt enforces for HiTab — align the solver's
# output convention with this dataset's gold convention.
_RHB_RATIO_RULE = (
    "If the question asks for a percentage or share, give the number on the "
    "0-100 percent scale (e.g. 14.44 for \"14.44%\") — do NOT give a 0-1 fraction."
)


def _patch_ratio_rule() -> None:
    for name in ("_DIRECT_SYS", "_CODEGEN_SYS"):
        setattr(answerer_mod, name,
                getattr(answerer_mod, name).replace(answerer_mod._RATIO_RULE,
                                                    _RHB_RATIO_RULE))


def em_norm(pred, gold_str, rel_tol: float = 1e-5) -> bool:
    """Deterministic strict scorer for RealHiTBench golds: `%`/comma-normalised
    numeric equality at ``rel_tol`` (HiTab-grade float tolerance — absorbs FP
    noise only, e.g. 21091.0 vs "21091.09" passes at 4e-6 but a wrong-cell
    near-miss like 1119760 vs 1119800 at 3.6e-5 fails). No x100/sign/±2%
    leniency (see docstring)."""
    g_s = str(gold_str).strip().rstrip("%").replace(",", "")
    try:
        g = float(g_s)
    except ValueError:
        return str(pred).strip().lower() == str(gold_str).strip().lower()
    try:
        p = float(str(pred).strip().rstrip("%").replace(",", ""))
    except (TypeError, ValueError):
        return False
    return abs(p - g) <= rel_tol * max(abs(g), 1e-9)


def build_table(fname: str, hf_repo: str) -> BenchTable | None:
    """Raw HTML -> reconstructed BenchTable (markup front-end, guessed boundaries).

    Returns None when the table is unusable (parse failure / too small / no data
    region) — the same skip rule as `scripts/realhitbench_ingest.py`.
    """
    from huggingface_hub import hf_hub_download
    try:
        html = open(hf_hub_download(hf_repo, f"html/{fname}.html",
                                    repo_type="dataset")).read()
        grid, merges = parse_html_table_with_merges(html)
    except Exception:  # noqa: BLE001
        return None
    if not grid or len(grid) < 3 or len(grid[0]) < 2:
        return None
    nhc = guess_n_header_cols(grid)
    nhr = guess_n_header_rows(grid, n_header_cols=nhc)
    nhr = max(1, min(nhr, len(grid) - 1))
    col_paths, row_paths = reconstruct_paths_with_merges(grid, merges, nhr,
                                                         n_header_cols=nhc)
    data = [row[nhc:] for row in grid[nhr:]]
    if not data or not data[0]:
        return None
    return BenchTable(
        table_id=fname,
        title=fname.replace("-", " "),
        data=data,
        top_paths=col_paths,
        left_paths=row_paths,
        source="realhitbench",
    )


def mcnemar_p(b: int, c: int) -> float:
    from scipy.stats import binomtest
    n = b + c
    return binomtest(b, n, 0.5).pvalue if n else 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-repo", default=HF_REPO)
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--solver-model", default="llama-3.3-70b-versatile")
    ap.add_argument("--mode", default="codegen", choices=["codegen", "direct"])
    ap.add_argument("--codegen-max-tokens", type=int, default=160,
                    help="reasoning models (gpt-oss) need ~1024")
    ap.add_argument("--max-context-tokens", type=int, default=8192,
                    help="context budget per solver call (wide RealHiTBench rows "
                         "overflow the 4096 default and drop injected chunks)")
    ap.add_argument("--max-inject", type=int, default=12,
                    help="cap on injected total-like chunks (dense-rank order)")
    ap.add_argument("--sample", type=int, default=0,
                    help="fix a CompStrucCata-stratified random subpopulation of "
                         "this size (0 = full aggregation subset)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--injected-first", action="store_true",
                    help="run queries whose treatment adds chunks first (quota "
                         "cutoff still yields informative pairs; partial-n Δ is "
                         "then NOT population-representative)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default="results/realhitbench_answer_accuracy.json")
    ap.add_argument("--records",
                    default="results/realhitbench_answer_accuracy_records.jsonl")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    qa = json.load(open(hf_hub_download(args.hf_repo, "QA_final.json",
                                        repo_type="dataset")))["queries"]
    pop = sorted((q for q in qa if q.get("SubQType") in AGG_SUBQTYPES),
                 key=lambda q: q["id"])
    n_full = len(pop)
    if args.sample and args.sample < n_full:
        by_cata = defaultdict(list)
        for q in pop:
            by_cata[q["CompStrucCata"]].append(q)
        rng = random.Random(args.seed)
        picked = []
        # proportional allocation, largest-remainder rounding
        quota = {k: len(v) * args.sample / n_full for k, v in by_cata.items()}
        base = {k: int(v) for k, v in quota.items()}
        rest = sorted(quota, key=lambda k: -(quota[k] - base[k]))
        short = args.sample - sum(base.values())
        for k in rest[:short]:
            base[k] += 1
        for k, v in sorted(by_cata.items()):
            picked += rng.sample(v, min(base[k], len(v)))
        pop = sorted(picked, key=lambda q: q["id"])
    n = len(pop)
    print(f"[pop] RealHiTBench agg (Calculation+Multi-hop NR): {n}"
          f"{f' (stratified sample of {n_full}, seed={args.seed})' if args.sample else ''}"
          f"  k={args.k} solver={args.solver_model}", flush=True)

    _patch_ratio_rule()
    emb = Embedder(args.embed_model, device="cpu")
    llm = GroqLLM(model_name=args.solver_model, retry_on_429=8)

    # ---- reconstruct + index each table lazily (LLM-free) -----------------
    tables: dict[str, BenchTable | None] = {}
    retr: dict[str, HybridRetriever] = {}
    total_chunk_idx: dict[str, list[int]] = {}

    def get_table(fname: str):
        if fname not in tables:
            t = build_table(fname, args.hf_repo)
            tables[fname] = t
            if t is not None:
                R = HybridRetriever(serialize_table(t, S2), emb)
                retr[fname] = R
                trows = total_like_rows_hybrid(t)
                total_chunk_idx[fname] = [i for i, ch in enumerate(R.chunks)
                                          if set(ch.rows) & trows]
        return tables[fname]

    # ---- retrieval contexts for every query (LLM-free) --------------------
    prep, n_table_skipped = [], 0
    for q in pop:
        t = get_table(q["FileName"])
        if t is None:
            n_table_skipped += 1
            continue
        R = retr[q["FileName"]]
        qv = np.asarray(emb.encode([q["Question"]])[0])
        order = R._rank(np.asarray(R._emb) @ qv)
        base_idx = list(order[:args.k])
        base_chunks = [R.chunks[i] for i in base_idx]
        rank_of = {i: r for r, i in enumerate(order)}
        extra_idx = sorted((i for i in total_chunk_idx[q["FileName"]]
                            if i not in set(base_idx)),
                           key=lambda i: rank_of.get(i, len(order)))[:args.max_inject]
        treat_chunks = base_chunks + [R.chunks[i] for i in extra_idx]
        prep.append({"q": q, "base_chunks": base_chunks,
                     "treat_chunks": treat_chunks, "n_injected": len(extra_idx)})
    n_inj_active = sum(1 for p in prep if p["n_injected"])
    print(f"[prep] {len(prep)} queries prepared ({n_table_skipped} skipped: table "
          f"unusable); treatment adds chunks on {n_inj_active}", flush=True)

    if args.injected_first:
        prep.sort(key=lambda p: (-p["n_injected"], p["q"]["id"]))
        print(f"[order] injected-first: {n_inj_active} injection-active queries "
              "run before the rest", flush=True)

    done: dict[str, dict] = {}
    if args.resume and Path(args.records).exists():
        with open(args.records) as fh:
            for line in fh:
                r = json.loads(line)
                done[r["qid"]] = r
        print(f"[resume] {len(done)} qids already recorded -> skipped", flush=True)

    # ---- solver pass: incremental append; quota cutoff keeps progress -----
    Path(args.records).parent.mkdir(parents=True, exist_ok=True)
    rec_fh = open(args.records, "a" if args.resume else "w")
    t0, n_run, cutoff = time.time(), 0, None
    for qi, p in enumerate(prep):
        q = p["q"]
        qid = f"rhb-{q['id']}"
        if qid in done:
            continue
        gold = [q["ProcessedAnswer"]]
        try:
            rb = answer(q["Question"], p["base_chunks"], llm, mode=args.mode,
                        max_context_tokens=args.max_context_tokens,
                        codegen_max_tokens=args.codegen_max_tokens)
            rt = answer(q["Question"], p["treat_chunks"], llm, mode=args.mode,
                        max_context_tokens=args.max_context_tokens,
                        codegen_max_tokens=args.codegen_max_tokens)
        except Exception as e:  # daily-quota 429 -> keep what we have
            cutoff = f"{type(e).__name__}: {e}"
            print(f"\n[cutoff] solver failed at {qi+1}/{len(prep)}: {cutoff}",
                  flush=True)
            break
        if rt.context_truncated:
            print(f"  [warn] qid={qid}: treat context truncated — injected "
                  "chunks may have been dropped", flush=True)
        rec = {"qid": qid, "cata": q["CompStrucCata"],
               "n_injected": p["n_injected"],
               "correct_base": evaluate_answer(rb.answer, gold),
               "correct_treat": evaluate_answer(rt.answer, gold),
               "correct_base_strict": em_norm(rb.answer, gold[0]),
               "correct_treat_strict": em_norm(rt.answer, gold[0]),
               "pred_base": rb.answer, "pred_treat": rt.answer, "gold": gold,
               "base_context_truncated": rb.context_truncated,
               "treat_context_truncated": rt.context_truncated}
        done[qid] = rec
        rec_fh.write(json.dumps(rec) + "\n")
        rec_fh.flush()
        n_run += 1
        if n_run % 4 == 0:
            d = list(done.values())
            print(f"  {len(done)}/{len(prep)}  "
                  f"acc_b={sum(r['correct_base'] for r in d)/len(d):.3f} "
                  f"acc_t={sum(r['correct_treat'] for r in d)/len(d):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    rec_fh.close()

    recs = [done[f"rhb-{p['q']['id']}"] for p in prep
            if f"rhb-{p['q']['id']}" in done]
    # recompute strict from stored pred/gold so resumed records written under an
    # older scorer are re-scored uniformly by the current em_norm
    for r in recs:
        r["correct_base_strict"] = em_norm(r["pred_base"], r["gold"][0])
        r["correct_treat_strict"] = em_norm(r["pred_treat"], r["gold"][0])
    ne = len(recs)
    if not ne:
        print("no evaluations recorded (quota exhausted immediately?)")
        return 1

    def block(rs, strict: bool):
        kb = "correct_base_strict" if strict else "correct_base"
        kt = "correct_treat_strict" if strict else "correct_treat"
        b = sum(r[kt] and not r[kb] for r in rs)   # treat-only
        c = sum(r[kb] and not r[kt] for r in rs)   # base-only
        m = len(rs)
        return {"n": m,
                "base": round(sum(r[kb] for r in rs) / m, 4),
                "treat": round(sum(r[kt] for r in rs) / m, 4),
                "delta": round((sum(r[kt] for r in rs) - sum(r[kb] for r in rs)) / m, 4),
                "treat_only": b, "base_only": c,
                "mcnemar_p": round(float(mcnemar_p(b, c)), 5)}

    inj_recs = [r for r in recs if r["n_injected"]]
    by_cata = {}
    for cata in sorted({r["cata"] for r in recs}):
        by_cata[cata] = block([r for r in recs if r["cata"] == cata], strict=True)

    out = {
        "population": {
            "name": "realhitbench_agg(Calculation+MultihopNR)",
            "n_full_subset": n_full, "n_target": n,
            "sample": args.sample or None, "seed": args.seed,
            "n_prepared": len(prep), "n_table_skipped": n_table_skipped,
            "n_evaluated": ne,
            "n_injection_active_target": n_inj_active,
            "n_injection_active_evaluated": len(inj_recs),
            "injected_first": args.injected_first, "cutoff": cutoff,
            "n_base_context_truncated": sum(r["base_context_truncated"] for r in recs),
            "n_treat_context_truncated": sum(r["treat_context_truncated"] for r in recs),
        },
        "pipeline": {"trees": "RECONSTRUCTED from raw HTML (markup front-end, "
                              "guessed header boundaries) — no gold structure",
                     "retriever": "dense", "k": args.k,
                     "solver": f"groq:{args.solver_model}", "mode": args.mode},
        "note": ("no OSC — RealHiTBench has no operand-cell annotations. strict = "
                 "em_norm on ProcessedAnswer (%/comma-normalised numeric equality, "
                 "rel_tol 1e-3, no scale/sign leniency; solver prompt asks percent-"
                 "scale to match the gold convention). Deterministic and symmetric, "
                 "NOT the paper's LLM-judge protocol — do not compare to the "
                 "RealHiTBench leaderboard. lenient = numeric_match (diagnostic)."),
        "answer_accuracy_strict": block(recs, strict=True),
        "answer_accuracy_lenient": block(recs, strict=False),
        "injection_active_subset_strict": block(inj_recs, strict=True) if inj_recs else None,
        "by_compstruccata_strict": by_cata,
        "mean_injected_chunks": round(sum(r["n_injected"] for r in recs) / ne, 2),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    s, l = out["answer_accuracy_strict"], out["answer_accuracy_lenient"]
    print("\n=== RESULT (RealHiTBench, reconstructed trees) ===")
    print(f"evaluated {ne}/{len(prep)} (cutoff={cutoff})")
    print(f"Accuracy strict-EM   base {s['base']}  ->  treat {s['treat']}  "
          f"(Δ {s['delta']:+.4f})   treat_only={s['treat_only']} "
          f"base_only={s['base_only']}  McNemar p={s['mcnemar_p']}")
    print(f"Accuracy lenient     base {l['base']}  ->  treat {l['treat']}  "
          f"(Δ {l['delta']:+.4f})   [internal diagnostic]")
    if inj_recs:
        i = out["injection_active_subset_strict"]
        print(f"injection-active subset (n={i['n']}): {i['base']} -> {i['treat']}  "
              f"treat_only={i['treat_only']} base_only={i['base_only']}")
    print("by CompStrucCata (strict):")
    for k, v in by_cata.items():
        print(f"  {k:<24} n={v['n']:>3}  {v['base']} -> {v['treat']}  (Δ {v['delta']:+.3f})")
    print(f"mean injected chunks/query: {out['mean_injected_chunks']}")
    print(f"wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
