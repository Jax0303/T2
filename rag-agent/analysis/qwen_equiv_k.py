#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Phase 4 Task A1 -- equivalent k under the READER's tokenizer.

Phase 3b sized k with ``Budget(BAAI/bge-small-en-v1.5)`` because that is the
counter every retrieval budget in this repo uses. The reader is Qwen2.5, whose
BPE counts the same text differently -- measured at GATE A, a nominal 5,120
bge-token context arrived as 8,043 Qwen tokens. So k is recomputed here against
the tokenizer that actually holds the context:

    k = floor((B_reader - prompt_overhead) / mean_chunk_tokens_qwen)

Everything is measured: chunk length over the whole corpus chunk set, overhead
over the whole candidate query pool. Nothing estimated.

  PYTHONPATH=. .venv/bin/python analysis/qwen_equiv_k.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis"))

import numpy as np                                                   # noqa: E402
from baseline_comparison_llm import Budget                           # noqa: E402
from header_path_coverage import load_corpus                         # noqa: E402
from stratified_recall import build_chunks                           # noqa: E402
from transformers import AutoTokenizer                               # noqa: E402

MODEL = "Qwen/Qwen2.5-7B-Instruct"
REV = "a09a35458c702b33eeacc393d103063234e8bc28"
B_READER = 4096
POLICIES = ("P1_fixed_512", "P4_path_cell")
SYS = "Answer with the value only. No explanation."

# (name, dataset, population, qtype). qtype filters the loaded population
# AFTER pin() -- passing it to realhitbench_corpus instead would drop queries the
# freeze still expects and trip the population check.
POOLS = [
    ("hitab_lookup", "hitab", "hitab_dev_lookup_all", ""),
    ("hitab_arith", "hitab", "hitab_dev_corpus_arith", ""),
    ("aitqa", "aitqa", "aitqa", ""),
    ("rhb_fact", "realhitbench", "rhb_lookup_all", "Fact Checking"),
    ("rhb_num", "realhitbench", "rhb_lookup_all", "Numerical Reasoning"),
]


def rhb_qtype():
    """query_id -> question type, straight from the shipped QA file."""
    qa = json.load(open("data/realhitbench/QA_final.json"))["queries"]
    return {str(q["id"]): q["QuestionType"] for q in qa}


class A:
    data_dir = "data/hitab"
    split = "dev"
    seed = 42
    mh_queries = 400
    rhb_em_only = False


def args_for(dataset, population):
    a = A()
    a.dataset, a.population, a.rhb_question_types = dataset, population, []
    a.data_dir = {"hitab": "data/hitab", "aitqa": "data/aitqa",
                  "realhitbench": "data/realhitbench"}[dataset]
    return a


def prompt_text(tok, ctx, question):
    """Exactly what LocalQwenLLM.complete tokenizes, minus the model call."""
    user = f"CONTEXT:\n{ctx}\n\nQUESTION: {question}\n\nAnswer:"
    return tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def count_all(tok, texts, batch=2000):
    n = []
    for i in range(0, len(texts), batch):
        n += [len(x) for x in tok(texts[i:i + batch],
                                  add_special_tokens=False)["input_ids"]]
    return np.array(n)


def main() -> int:
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    bud = Budget("BAAI/bge-small-en-v1.5")
    out = {"model": MODEL, "revision": REV, "B_reader": B_READER,
           "system": SYS, "pools": {}, "chunks": {}}

    # --- prompt overhead: the whole template with an EMPTY context ---
    corpora, qt_of = {}, rhb_qtype()
    for name, ds, pop, qt in POOLS:
        key = (ds, pop)
        if key not in corpora:
            corpora[key] = load_corpus(args_for(ds, pop))
        C = corpora[key]
        qs = [q for q in C.queries
              if not qt or qt_of.get(str(q["query_id"])) == qt]
        ov = count_all(tok, [prompt_text(tok, "", q["question"]) for q in qs])
        out["pools"][name] = {
            "dataset": ds, "population": pop, "qtype": qt or "(all)",
            "n_queries": len(qs),
            "overhead_mean": float(ov.mean()), "overhead_median": float(np.median(ov)),
            "overhead_min": int(ov.min()), "overhead_max": int(ov.max())}
        print(f"[overhead] {name:<13} n={len(qs):<5} "
              f"mean={ov.mean():.1f} median={np.median(ov):.0f} "
              f"min={ov.min()} max={ov.max()}", flush=True)

    # --- chunk length, per dataset (hitab's chunk set is population-independent:
    #     hitab_corpus indexes every dev table before pin() touches the queries) ---
    for ds, key in (("hitab", ("hitab", "hitab_dev_lookup_all")),
                    ("aitqa", ("aitqa", "aitqa")),
                    ("realhitbench", ("realhitbench", "rhb_lookup_all"))):
        C = corpora[key]
        by = defaultdict(dict)
        for n, (t, i, j) in enumerate(C.cell_owner):
            by[t][(i, j)] = n
        for pol in POLICIES:
            ch, _ = build_chunks(C, pol, bud, by, "S3c")
            # "\n" separator counted in, since the reader pays for it
            t_q = count_all(tok, [c.text + "\n" for c in ch])
            t_b = np.array([bud.count(c.text) for c in ch])
            out["chunks"][f"{ds}|{pol}"] = {
                "dataset": ds, "policy": pol, "n_chunks": len(ch),
                "qwen_mean": float(t_q.mean()), "qwen_median": float(np.median(t_q)),
                "qwen_max": int(t_q.max()), "qwen_total": int(t_q.sum()),
                "bge_mean": float(t_b.mean()),
                "qwen_over_bge": float(t_q.mean() / t_b.mean())}
            print(f"[chunk] {ds:<13} {pol:<13} n={len(ch):<7} "
                  f"qwen_mean={t_q.mean():.2f} bge_mean={t_b.mean():.2f} "
                  f"ratio={t_q.mean()/t_b.mean():.3f}", flush=True)

    # --- k = floor((B - overhead) / mean chunk) ---
    ds_of = {"hitab_lookup": "hitab", "hitab_arith": "hitab", "aitqa": "aitqa",
             "rhb_fact": "realhitbench", "rhb_num": "realhitbench"}
    out["k"] = {}
    for name, ds in ds_of.items():
        for pol in POLICIES:
            c = out["chunks"][f"{ds}|{pol}"]
            p = out["pools"][name]
            for tag, ov in (("mean", p["overhead_mean"]), ("max", p["overhead_max"])):
                out["k"][f"{name}|{pol}|ov_{tag}"] = {
                    "budget": B_READER, "overhead": ov,
                    "usable": B_READER - ov, "mean_chunk": c["qwen_mean"],
                    "k": max(1, int((B_READER - ov) // c["qwen_mean"]))}

    p = Path("results/phase4/qwen_equiv_k.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
