#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Raw-dataset leg of the 2-dataset strategy: MY METHOD vs BASELINE serialization.

The thesis claim on genuinely RAW hierarchical tables (no gold structure, only
parse structure + gold answers) is that *my* preprocessing — reconstruct the
header tree, then verbalize every cell 1:1 as a sentence carrying its full
row/col header path (S2) — beats the structure-naive baseline serialization
(S1, leaf headers only) on BOTH retrieval and answer accuracy, and does so
independently of the solver LLM (same retriever, same LLM, only the
serialization differs).

RealHiTBench (Zhang et al., ACL 2025; HF `spzy/RealHiTBench`) is the raw dataset:
tables are PhpSpreadsheet HTML and the header trees the pipeline retrieves over
are RECONSTRUCTED by our markup front-end (`parse_html_table_with_merges` +
`reconstruct_paths_with_merges` with guessed header boundaries). This script is
the ANSWER-accuracy leg of the my-vs-baseline comparison (retrieval accuracy
needs gold operand cells, which RealHiTBench does not ship — that leg runs on
MultiHiertt/HiTab via the within-doc bench):

  raw HTML -> reconstruct tree -> serialize cells -> dense top-k -> solver -> answer
    base  (baseline) = S1 flat serialization  (leaf headers only)
    treat (mine)     = S2 header-path serialization  (full row/col path per cell)

Both arms use the SAME dense retriever, SAME solver, SAME k; the ONLY difference
is the serialization the retriever indexes and the solver reads. A positive Δ
(treat - base) is my method beating the baseline; McNemar tests the paired flip
counts (S2-only-correct vs S1-only-correct).

Notes / caveats:
  * NO OSC / no retrieval-accuracy here: RealHiTBench has no operand-cell
    annotations (gold answers only), so operand-set completeness / set-EM is
    not computable. We score ANSWER accuracy only.
  * Scoring: gold = `ProcessedAnswer` (numeric string throughout the
    aggregation subset, e.g. "543", "14.44%", "63.90"). `strict` = `em_norm`:
    strip `%`/commas from the gold, then compare rounded to the gold's own
    decimal places (a relative tolerance forgave dropped decimals on the large
    values these tables carry — see `em_norm`) — NO
    scale (x100), sign, or +-2% leniency. NOT RealHiTBench's own LLM-judge
    protocol; do not compare against the paper's leaderboard numbers. Golds
    are percent-SCALE strings, so the solver prompt's ratio rule is OVERRIDDEN
    to ask for percent-scale numbers, matching the gold convention. `lenient`
    = numeric_match (±2%, %/fraction scale) — internal diagnostic only.
  * Context budget: RealHiTBench rows are wide (~400 tokens each); budget is
    8192 and per-query truncation is recorded.
  * Population: SubQType in {Calculation, Multi-hop Numerical Reasoning} = 334
    queries / 258 tables. `--sample N --seed S` fixes a CompStrucCata-stratified
    random subpopulation up front; deterministic, so --resume continues the SAME
    subpopulation across days. Queries run in id order.

Run (needs the chosen backend's key: GROQ_API_KEY / OPENAI_API_KEY):
  PYTHONPATH=. python3 scripts/realhitbench_answer_accuracy.py \
    --solver-backend groq --solver-model llama-3.3-70b-versatile \
    --codegen-max-tokens 160 --sample 100 --seed 0 --resume

Omit --solver-model to take the backend's own default. Give each backend its
OWN --out/--records paths: the records file carries no solver field, so a
--resume onto another backend's file silently mixes solvers and the mix is
not recoverable after the fact.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import rag_agent.generate.answerer as answerer_mod
from rag_agent.bench.schema import BenchTable, Chunk
from rag_agent.generate.answerer import answer, evaluate_answer
from rag_agent.llm.factory import build_llm
from rag_agent.query.operand_decomposer import Embedder
from rag_agent.reconstruct import (guess_n_header_cols, guess_n_header_rows,
                                   parse_html_table_with_merges,
                                   reconstruct_paths_with_merges)
from rag_agent.retrieve.operand_retriever import HybridRetriever
from rag_agent.serialize import S1, S2, serialize_table
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


def _half_up(x: float, dp: int) -> float:
    """Round half AWAY FROM ZERO, the convention the golds were written under.

    ``round()`` is banker's rounding, so it takes ``4.125`` to ``4.12`` while the
    gold for that cell reads "4.13". The gold is the rounded form of the true
    value, so the scorer has to round the way whoever produced it did, or a
    correct prediction is marked wrong on the tie alone.
    """
    q = Decimal(1).scaleb(-dp)
    return float(Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP))


def gold_is_numeric(gold_str) -> bool:
    """True if the gold is a single number once `%` and commas are stripped.

    16 of the 94 questions in the aggregation subset have golds like "Japan",
    "Crew C", "2015-11-03, 2403" or "Decrease by 0.03." — names, dates, IDs and
    sentences. The solver runs in codegen mode under a prompt that requires a
    number on the 0-100 percent scale, so those questions cannot be answered
    correctly by construction, and they score 0 in EVERY arm. They are reported
    as their own stratum rather than silently diluting both arms.
    """
    s = str(gold_str).strip().rstrip("%").replace(",", "")
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def _f1_tokens(s) -> List[str]:
    """Lowercase, split on non-word runs, then strip edge punctuation.

    ``.`` and ``-`` have to survive INSIDE a token — ``0.03`` and ``2015-11-03``
    are single tokens — but not at the edges, or the sentence-final period in
    "Decrease by 0.03." makes that number a different token from "0.03".
    """
    return [t for t in (tok.strip(".-")
                        for tok in re.split(r"[^0-9a-z.\-]+", str(s).strip().lower()))
            if t]


def token_f1(pred, gold_str) -> float:
    """Token-level F1 — the second metric RealHiTBench reports alongside EM.

    Zhao et al. score Fact Checking / Numerical Reasoning / Structure
    Comprehending with "F1 and EM" (arXiv:2506.13405 §5.1). EM alone is the
    stricter half of that pair, so reporting only EM understates this pipeline
    against every number the benchmark publishes. F1 also gives the text golds
    a way to earn partial credit — "-0.03" against "Decrease by 0.03." is 0 under
    EM and non-zero here.
    """
    # A numeric gold is scored numerically, not as a bag of characters: the
    # solver returns a raw float, so "0.2978723404255319" and the gold "0.30"
    # share no token and would score 0 on an answer EM calls correct. On this
    # stratum F1 collapses to EM by construction, which is the honest reading —
    # there is no partial credit to be had inside a single number.
    if gold_is_numeric(gold_str):
        return float(em_norm(pred, gold_str))
    p, g = _f1_tokens(pred), _f1_tokens(gold_str)
    if not p or not g:
        return float(bool(p) == bool(g))
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def em_norm(pred, gold_str) -> bool:
    """Deterministic strict scorer for RealHiTBench golds: `%`/comma-normalised,
    compared at the precision the gold was written at.

    This used to be a RELATIVE tolerance of 1e-5, described as absorbing float
    noise only. It did not: relative tolerance loosens as the number grows, so
    `21091.0` against "21091.09" passed at 4.3e-6, and `59969486.0` against
    "59969486.20" at 3.3e-9. A dropped decimal place is a different number, not
    noise, and forgiving it inflated the treated arm by two records per solver.

    Rounding both sides to the gold's own decimal places draws the line where
    the data does: `184.66899999999998` against "184.67" is one float printed
    two ways and passes, while `21091.0` against "21091.09" fails. Still no
    x100, sign or +/-2% leniency.
    """
    g_s = str(gold_str).strip().rstrip("%").replace(",", "")
    try:
        g = float(g_s)
    except ValueError:
        return str(pred).strip().lower() == str(gold_str).strip().lower()
    try:
        p = float(str(pred).strip().rstrip("%").replace(",", ""))
    except (TypeError, ValueError):
        return False
    dp = len(g_s.split(".")[1]) if "." in g_s else 0
    return _half_up(p, dp) == _half_up(g, dp)


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


def table_md_chunk(t: BenchTable) -> Chunk:
    """The WHOLE reconstructed table as one markdown block — the input format
    the RealHiTBench paper and the table-LLM literature actually feed a model.

    Scored by the same em_norm as the retrieval arms, so this measures what
    retrieval costs us, holding the scorer fixed. It is still NOT the paper's
    LLM-judge protocol, so it does not license a leaderboard comparison.
    """
    head = [""] + [" > ".join(p) for p in t.top_paths]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for i, row in enumerate(t.data):
        lab = " > ".join(t.left_paths[i]) if i < len(t.left_paths) else ""
        lines.append("| " + " | ".join([lab] + [str(x) for x in row]) + " |")
    return Chunk(table_id=t.table_id, chunk_id=f"{t.table_id}::table_md",
                 text=f"{t.title}\n" + "\n".join(lines),
                 rows=list(range(len(t.data))),
                 cols=list(range(len(t.data[0]) if t.data else 0)))


def mcnemar_p(b: int, c: int) -> float:
    from scipy.stats import binomtest
    n = b + c
    return binomtest(b, n, 0.5).pvalue if n else 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-repo", default=HF_REPO)
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default=None,
                    help="embedder device (default: cuda if available else cpu)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--solver-backend", default="groq",
                    choices=["groq", "openai", "local"],
                    help="solver family. A second backend is how the S1-vs-S2 "
                         "result is shown to be solver-independent — keep each "
                         "backend in its OWN --records file, never mixed")
    ap.add_argument("--solver-model", default="",
                    help="empty = that backend's own default (see llm/factory.py); "
                         "a Groq model name here would be a 400 on --solver-backend "
                         "openai, and the failure only surfaces after the full "
                         "reconstruct+index pass")
    ap.add_argument("--mode", default="codegen", choices=["codegen", "direct"])
    ap.add_argument("--codegen-max-tokens", type=int, default=160,
                    help="reasoning models (gpt-oss) need ~1024")
    ap.add_argument("--max-context-tokens", type=int, default=8192,
                    help="context budget per solver call (wide RealHiTBench rows "
                         "overflow the 4096 default)")
    ap.add_argument("--sample", type=int, default=0,
                    help="fix a CompStrucCata-stratified random subpopulation of "
                         "this size (0 = full aggregation subset)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--base-arm", default="s1", choices=["s1", "table_md"],
                    help="what the BASE arm reads: s1 = flat serialization "
                         "(default), table_md = the whole reconstructed table as "
                         "markdown, no retrieval — the literature's input format. "
                         "Reusing the base slot keeps the run at two solver calls "
                         "per query, which the free Groq daily budget needs.")
    ap.add_argument("--full-table", action="store_true",
                    help="add a third arm: the whole reconstructed table as "
                         "one markdown block, no retrieval")
    ap.add_argument("--out", default="results/realhitbench_s1_vs_s2.json")
    ap.add_argument("--records",
                    default="results/realhitbench_s1_vs_s2_records.jsonl")
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
          f"  k={args.k} solver={args.solver_model}  arms: base=S1(flat) vs treat=S2(mine)",
          flush=True)

    _patch_ratio_rule()
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] embedder on {args.device}", flush=True)
    emb = Embedder(args.embed_model, device=args.device)
    llm = build_llm(f"{args.solver_backend}:{args.solver_model}", retry_on_429=8)
    print(f"[solver] {llm.name}", flush=True)

    # ---- reconstruct each table once, index it under BOTH serializations ----
    tables: dict[str, BenchTable | None] = {}
    retr_s1: dict[str, HybridRetriever] = {}
    retr_s2: dict[str, HybridRetriever] = {}

    def get_table(fname: str):
        if fname not in tables:
            t = build_table(fname, args.hf_repo)
            tables[fname] = t
            if t is not None:
                retr_s1[fname] = HybridRetriever(serialize_table(t, S1), emb)
                retr_s2[fname] = HybridRetriever(serialize_table(t, S2), emb)
        return tables[fname]

    def topk_chunks(R: HybridRetriever, qv: np.ndarray):
        order = R._rank(np.asarray(R._emb) @ qv)
        return [R.chunks[i] for i in order[:args.k]]

    # ---- retrieval contexts for every query (LLM-free) --------------------
    prep, n_table_skipped = [], 0
    for q in pop:
        t = get_table(q["FileName"])
        if t is None:
            n_table_skipped += 1
            continue
        qv = np.asarray(emb.encode([q["Question"]])[0])
        prep.append({"q": q,
                     "base_chunks": ([table_md_chunk(t)]
                                     if args.base_arm == "table_md"
                                     else topk_chunks(retr_s1[q["FileName"]], qv)),
                     "treat_chunks": topk_chunks(retr_s2[q["FileName"]], qv),
                     "md_chunks": [table_md_chunk(t)] if args.full_table else None})
    print(f"[prep] {len(prep)} queries prepared ({n_table_skipped} skipped: table "
          f"unusable)", flush=True)

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
            rm = (answer(q["Question"], p["md_chunks"], llm, mode=args.mode,
                         max_context_tokens=args.max_context_tokens,
                         codegen_max_tokens=args.codegen_max_tokens)
                  if args.full_table else None)
        except Exception as e:  # daily-quota 429 -> keep what we have
            cutoff = f"{type(e).__name__}: {e}"
            print(f"\n[cutoff] solver failed at {qi+1}/{len(prep)}: {cutoff}",
                  flush=True)
            break
        rec = {"qid": qid, "cata": q["CompStrucCata"],
               "correct_base": evaluate_answer(rb.answer, gold),
               "correct_treat": evaluate_answer(rt.answer, gold),
               "correct_base_strict": em_norm(rb.answer, gold[0]),
               "correct_treat_strict": em_norm(rt.answer, gold[0]),
               "pred_base": rb.answer, "pred_treat": rt.answer, "gold": gold,
               "base_context_truncated": rb.context_truncated,
               "treat_context_truncated": rt.context_truncated}
        if rm is not None:
            rec["pred_md"] = rm.answer
            rec["correct_md"] = evaluate_answer(rm.answer, gold)
            rec["correct_md_strict"] = em_norm(rm.answer, gold[0])
            rec["md_context_truncated"] = rm.context_truncated
        done[qid] = rec
        rec_fh.write(json.dumps(rec) + "\n")
        rec_fh.flush()
        n_run += 1
        if n_run % 4 == 0:
            d = list(done.values())
            print(f"  {len(done)}/{len(prep)}  "
                  f"acc_S1={sum(r['correct_base'] for r in d)/len(d):.3f} "
                  f"acc_S2={sum(r['correct_treat'] for r in d)/len(d):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    rec_fh.close()

    recs = [done[f"rhb-{p['q']['id']}"] for p in prep
            if f"rhb-{p['q']['id']}" in done]
    # recompute strict from stored pred/gold so resumed records written under an
    # older scorer are re-scored uniformly by the current em_norm
    for r in recs:
        r["correct_base_strict"] = em_norm(r["pred_base"], r["gold"][0])
        r["correct_treat_strict"] = em_norm(r["pred_treat"], r["gold"][0])
        if "pred_md" in r:
            r["correct_md_strict"] = em_norm(r["pred_md"], r["gold"][0])
    ne = len(recs)
    if not ne:
        print("no evaluations recorded (quota exhausted immediately?)")
        return 1

    def block(rs, strict: bool):
        kb = "correct_base_strict" if strict else "correct_base"
        kt = "correct_treat_strict" if strict else "correct_treat"
        b = sum(r[kt] and not r[kb] for r in rs)   # treat(S2/mine)-only
        c = sum(r[kb] and not r[kt] for r in rs)   # base(S1/baseline)-only
        m = len(rs)
        return {"n": m,
                "base": round(sum(r[kb] for r in rs) / m, 4),
                "treat": round(sum(r[kt] for r in rs) / m, 4),
                "delta": round((sum(r[kt] for r in rs) - sum(r[kb] for r in rs)) / m, 4),
                "treat_only": b, "base_only": c,
                "mcnemar_p": round(float(mcnemar_p(b, c)), 5)}

    by_cata = {}
    for cata in sorted({r["cata"] for r in recs}):
        by_cata[cata] = block([r for r in recs if r["cata"] == cata], strict=True)

    out = {
        "population": {
            "name": "realhitbench_agg(Calculation+MultihopNR)",
            "n_full_subset": n_full, "n_target": n,
            "sample": args.sample or None, "seed": args.seed,
            "n_prepared": len(prep), "n_table_skipped": n_table_skipped,
            "n_evaluated": ne, "cutoff": cutoff,
            "n_base_context_truncated": sum(r["base_context_truncated"] for r in recs),
            "n_treat_context_truncated": sum(r["treat_context_truncated"] for r in recs),
        },
        "arms": {"base": ("whole reconstructed table as one markdown block, NO "
                          "retrieval — the literature's input format"
                          if args.base_arm == "table_md" else
                          "S1 (s1_flat) — baseline serialization, leaf headers only"),
                 "treat": "S2 (s2_headerpath) — mine, full row/col header path per cell"},
        "pipeline": {"trees": "RECONSTRUCTED from raw HTML (markup front-end, "
                              "guessed header boundaries) — no gold structure",
                     "retriever": "dense", "k": args.k,
                     # llm.name, not the raw args: --solver-model may be empty
                     # and resolved to the backend's default by the factory.
                     "solver": llm.name, "mode": args.mode,
                     "controls": "same retriever/solver/k for both arms; only the "
                                 "serialization differs (isolates preprocessing)"},
        "note": ("no OSC / no retrieval-accuracy — RealHiTBench has no operand-cell "
                 "annotations (answer accuracy only). strict = em_norm on "
                 "ProcessedAnswer (%/comma-normalised numeric equality, compared at "
                 "the gold's own decimal places, "
                 "no scale/sign leniency; solver prompt asks percent-scale to match "
                 "the gold convention). Deterministic and symmetric, NOT the paper's "
                 "LLM-judge protocol — do not compare to the RealHiTBench leaderboard. "
                 "lenient = numeric_match (diagnostic)."),
        "answer_accuracy_strict": block(recs, strict=True),
        "full_table_arm": ({
            "n": len(recs),
            "table_md": round(sum(r["correct_md_strict"] for r in recs) / len(recs), 4),
            "s2_retrieval": round(sum(r["correct_treat_strict"] for r in recs) / len(recs), 4),
            "md_only": sum(r["correct_md_strict"] and not r["correct_treat_strict"] for r in recs),
            "s2_only": sum(r["correct_treat_strict"] and not r["correct_md_strict"] for r in recs),
            "mcnemar_p": round(float(mcnemar_p(
                sum(r["correct_treat_strict"] and not r["correct_md_strict"] for r in recs),
                sum(r["correct_md_strict"] and not r["correct_treat_strict"] for r in recs))), 5),
            "n_md_truncated": sum(r.get("md_context_truncated", False) for r in recs),
        } if all("correct_md_strict" in r for r in recs) and recs else None),
        "answer_accuracy_lenient": block(recs, strict=False),
        "by_compstruccata_strict": by_cata,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    s, l = out["answer_accuracy_strict"], out["answer_accuracy_lenient"]
    print("\n=== RESULT (RealHiTBench, reconstructed trees; S1 baseline vs S2 mine) ===")
    print(f"evaluated {ne}/{len(prep)} (cutoff={cutoff})")
    print(f"Accuracy strict-EM   S1(base) {s['base']}  ->  S2(mine) {s['treat']}  "
          f"(Δ {s['delta']:+.4f})   S2_only={s['treat_only']} "
          f"S1_only={s['base_only']}  McNemar p={s['mcnemar_p']}")
    print(f"Accuracy lenient     S1 {l['base']}  ->  S2 {l['treat']}  "
          f"(Δ {l['delta']:+.4f})   [internal diagnostic]")
    print("by CompStrucCata (strict):")
    for k, v in by_cata.items():
        print(f"  {k:<24} n={v['n']:>3}  {v['base']} -> {v['treat']}  (Δ {v['delta']:+.3f})")
    print(f"wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
