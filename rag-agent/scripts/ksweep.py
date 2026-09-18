#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Context size k vs reader EM, two legs, over a fixed 300-query sample.

  LEG A  natural truncation of the cached top-20 P4_path_cell (BGE-Large-En)
         ranking to the top k. No forced insertion. All 300 queries.
  LEG B  GOLD-FORCED (not deployable) diagnostic: gold cell pinned at context
         position 0, remaining k-1 slots filled by the top-ranked non-gold
         cells. Only queries whose gold cell the cached top-20 actually found.

Retrieval is never re-run here: this script only reads the *_records.jsonl a
single scripts/retrieval_accuracy.py invocation already wrote (context_units,
already ranked by hybrid score, already cell-unit / template=s3c). See
RESULTS_KSWEEP.md for the exact retrieval command.

GATE (user-mandated, not a script flag that auto-advances):
  --stage 1   LEG B, k in {1, 20} only. Prints both EM values, then stops.
  --stage 2   LEG B, remaining k in {2, 3, 5, 10}. Run only after approval.
  --stage 3   LEG A, all k in {1, 2, 3, 5, 10, 20}. Run only after approval.

  PYTHONPATH=. .venv/bin/python scripts/ksweep.py --stage 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_agent.eval.artifacts import digest, file_digest, provenance      # noqa: E402
from rag_agent.eval.metrics import hitab_exact_match_text                 # noqa: E402
from rag_agent.llm.factory import build_llm                               # noqa: E402
from scripts.answer_accuracy import (PROMPTS, check_context_limit,        # noqa: E402
                                     load_evidence, load_saved_rows)
from scripts.bottleneck_diagnosis import stratified_sample                # noqa: E402

RECORDS = "results/retrieval_accuracy/s3c_bgelarge_v2_records.jsonl"
READER = "local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit"
PROMPT_NAME = "neutral"
MAX_TOKENS = 64
SEED = 42
B_BOOTSTRAP = 10000
K_ALL = (1, 2, 3, 5, 10, 20)
K_STAGE = {1: (1, 20), 2: (2, 3, 5, 10), 3: K_ALL}
LEG_BY_STAGE = {1: "B", 2: "B", 3: "A"}

RESULTS = ROOT / "results"
RAW_DIR = RESULTS / "ksweep_raw"
POP_FILE = RESULTS / "ksweep_population_300.json"


def population_300() -> list:
    """The 300-query sample the user fixed LEG A/B to. Reuses this repo's own
    stratified-sample utility (rank-bucket allocation, seed-fixed) rather than
    inventing a new sampling method; cached to disk once so every stage reads
    the identical list."""
    if POP_FILE.exists():
        return json.loads(POP_FILE.read_text(encoding="utf-8"))["query_ids"]
    qids = stratified_sample(300, SEED)
    POP_FILE.write_text(json.dumps({
        "n": len(qids), "seed": SEED,
        "method": "scripts.bottleneck_diagnosis.stratified_sample(n=300, seed=42) "
                 "over load_primary_population() rank buckets "
                 "(results/evaluation_v2/s3c_v2_records.jsonl)",
        "query_ids": qids}, indent=2, ensure_ascii=False), encoding="utf-8")
    return qids


def leg_a_context(units: list, k: int) -> list:
    return [u["text"] for u in units[:k]]


def leg_b_context(units: list, gold: set, k: int) -> list:
    gi = next(i for i, u in enumerate(units) if set(map(tuple, u["cells"])) == gold)
    rest = units[:gi] + units[gi + 1:]
    return [units[gi]["text"]] + [u["text"] for u in rest[:k - 1]]


def build_context(leg: str, r: dict, k: int) -> tuple:
    units = r["context_units"]
    if leg == "A":
        ctx = leg_a_context(units, k)
        got = {tuple(c) for u in units[:k] for c in u["cells"]}
        gold_in_context = int(set(map(tuple, r["gold_cells"])) <= got)
        return ctx, gold_in_context
    ctx = leg_b_context(units, set(map(tuple, r["gold_cells"])), k)
    return ctx, 1


def run_leg_k(leg: str, k: int, records: dict, order: list, llm, limit: int) -> dict:
    raw_path = RAW_DIR / f"leg{leg}_k{k}.jsonl"
    fail_path = RAW_DIR / f"leg{leg}_k{k}_failures.jsonl"
    done = load_saved_rows(raw_path, order) if raw_path.exists() else {}
    rows, t0 = list(done.values()), time.time()
    failed = ([json.loads(line) for line in fail_path.read_text(encoding="utf-8").splitlines()]
             if fail_path.exists() else [])
    failed_ids = {f["query_id"] for f in failed}
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with raw_path.open("a", encoding="utf-8", newline="\n") as stream, \
         fail_path.open("a", encoding="utf-8", newline="\n") as fstream:
        for i, qid in enumerate(order, 1):
            if qid in done or qid in failed_ids:
                continue
            r = records[qid]
            try:
                ctx, gold_in_context = build_context(leg, r, k)
                user = "Context:\n" + "\n".join(ctx) + f"\n\nQuestion: {r['question']}\nAnswer:"
                n_tok = (llm.n_prompt_tokens(PROMPTS[PROMPT_NAME], user)
                         if hasattr(llm, "n_prompt_tokens") else None)
                check_context_limit(n_tok, MAX_TOKENS, limit)
                pred = llm.complete(PROMPTS[PROMPT_NAME], user, max_tokens=MAX_TOKENS,
                                    temperature=0.0)
                em = int(hitab_exact_match_text(pred, r["answer"]))
            except Exception as exc:                                     # noqa: BLE001
                fstream.write(json.dumps({"query_id": qid, "leg": leg, "k": k,
                                          "error": repr(exc)}, ensure_ascii=False) + "\n")
                fstream.flush()
                os.fsync(fstream.fileno())
                print(f"  [FAIL] {leg} k={k} {qid}: {exc!r}", flush=True)
                continue
            row = {"query_id": qid, "leg": leg, "k": k, "pred": pred,
                  "answer": r["answer"], "em": em, "gold_in_context": gold_in_context,
                  "n_tok": n_tok}
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            rows.append(row)
            if i % 50 == 0:
                print(f"  leg{leg} k={k} {i}/{len(order)} {time.time() - t0:.0f}s", flush=True)
    return {"rows": rows, "n_failed": len(failed) + sum(
        1 for qid in order if qid not in done and qid in failed_ids)}


def paired_bootstrap_ci(hits: list, idx: np.ndarray) -> dict:
    arr = np.array(hits, dtype=np.float64)
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": round(float(arr.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def mcnemar(a: dict, b: dict) -> dict:
    qids = sorted(set(a) & set(b))
    a_only = sum(a[q] and not b[q] for q in qids)
    b_only = sum(b[q] and not a[q] for q in qids)
    n_disc = a_only + b_only
    p = binomtest(min(a_only, b_only), n_disc, 0.5).pvalue if n_disc else 1.0
    return {"n": len(qids), "a_only": a_only, "b_only": b_only,
           "discordant": n_disc, "p_value": round(p, 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--records", default=RECORDS)
    ap.add_argument("--reader", default=READER)
    a = ap.parse_args()

    qids300 = population_300()
    records, retrieval_meta = load_evidence(a.records)
    missing = [q for q in qids300 if q not in records or "correct" not in records[q]]
    if missing:
        raise SystemExit(f"{len(missing)}개 쿼리가 검색 캐시에 없거나 미채점 상태다: {missing[:10]}...")

    leg = LEG_BY_STAGE[a.stage]
    ks = K_STAGE[a.stage]
    if leg == "A":
        order = sorted(qids300)
    else:
        order = sorted(q for q in qids300 if records[q]["correct"] == 1)
        excluded = sorted(q for q in qids300 if records[q]["correct"] == 0)
        print(f"[LEG B population] n={len(order)} 제외(top-20 검색 실패)={len(excluded)}", flush=True)

    llm = build_llm(a.reader)
    import torch                                                          # noqa: E402
    torch.manual_seed(SEED)
    limit = getattr(llm, "context_limit", 0)

    stage_result = {}
    for k in ks:
        print(f"== stage {a.stage} leg {leg} k={k} n={len(order)} ==", flush=True)
        out = run_leg_k(leg, k, records, order, llm, limit)
        hits = [r["em"] for r in out["rows"]]
        em = round(sum(hits) / len(hits), 4) if hits else None
        stage_result[k] = {"n": len(hits), "n_failed": out["n_failed"], "em": em}
        print(f"-- leg {leg} k={k}: n={len(hits)} n_failed={out['n_failed']} EM={em}", flush=True)

    if a.stage == 1:
        print(json.dumps({"LEG_B_k1_EM": stage_result[1]["em"],
                          "LEG_B_k20_EM": stage_result[20]["em"]}, indent=2))
        print("STOP — 1단계 완료. 사용자 확인 없이 2단계로 진행하지 않는다.", flush=True)

    summary_path = RESULTS / f"ksweep_leg{leg}.json"
    by_k = {}
    if summary_path.exists():
        by_k = json.loads(summary_path.read_text(encoding="utf-8")).get("by_k", {})
    by_k.update({str(k): res for k, res in stage_result.items()})
    records_out = []
    for raw_path in sorted(RAW_DIR.glob(f"leg{leg}_k*.jsonl")):
        if raw_path.name.endswith("_failures.jsonl"):
            continue
        with raw_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                records_out.append({"query_id": row["query_id"], "leg": row["leg"],
                                    "k": row["k"], "pred": row["pred"],
                                    "answer": row["answer"], "em": row["em"]})
    summary = {
        "leg": leg, "label": "GOLD-FORCED (not deployable)" if leg == "B" else "natural truncation",
        "reader": a.reader, "prompt": PROMPT_NAME, "seed": SEED, "max_new_tokens": MAX_TOKENS,
        "retrieval_records": a.records, "retrieval_records_sha256": file_digest(a.records),
        "population_n": len(order), "by_k": by_k,
        "excluded_query_ids": excluded if leg == "B" else None,
        "n_excluded": len(excluded) if leg == "B" else None,
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "provenance": provenance(ROOT),
        "records": records_out,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[written] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
