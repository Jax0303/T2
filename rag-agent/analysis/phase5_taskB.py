#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Phase 5 Task B -- operator inventory + a 10-query extract trial."""
from __future__ import annotations

import collections
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis"))

import torch                                                         # noqa: E402
from phase5_extract import OPS, SYS_EXTRACT, parse                   # noqa: E402
from phase5_taskA import smoke_jobs                                  # noqa: E402
from phase4_summary import em                                        # noqa: E402
from rag_agent.bench.hitab import load_queries                       # noqa: E402
from rag_agent.llm.local_qwen import LocalQwenLLM                    # noqa: E402

MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
REV = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
SEED, MAXNEW, N = 42, 128, 10
OUT = Path("results/phase5/taskB.json")


def inventory():
    qs, _ = load_queries("data/hitab", "dev")
    d = {str(q.query_id): q for q in qs}
    pop = [l.strip() for l in open("populations/hitab_dev_corpus_arith.txt")
           if l.strip() and not l.startswith("#")]
    smp = [r["query_id"] for r in csv.DictReader(
        open("results/phase4/sample_294.csv")) if r["pool"] == "hitab_arith"]
    return {
        "population_175": collections.Counter(
            d[i].aggregation or "none" for i in pop).most_common(),
        "sample_60": collections.Counter(
            d[i].aggregation or "none" for i in smp).most_common()}


def main() -> int:
    inv = inventory()
    jobs = smoke_jobs(n=N)
    torch.manual_seed(SEED)
    llm = LocalQwenLLM(model_name=MODEL, dtype="float16", quantization="4bit")
    tok = llm.tokenizer
    rows = []
    for j in jobs:
        user = f"CONTEXT:\n{j['ctx']}\n\nQUESTION: {j['query']}\n\nJSON:"
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYS_EXTRACT},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ins = tok(prompt, return_tensors="pt").to(llm.device)
        t0 = time.time()
        with torch.inference_mode():
            out = llm.model.generate(**ins, max_new_tokens=MAXNEW,
                                     do_sample=False,
                                     pad_token_id=tok.eos_token_id)
        gen = out[0, ins["input_ids"].shape[1]:]
        raw = tok.decode(gen, skip_special_tokens=True)
        val, op, why = parse(raw)
        rows.append(dict(query_id=j["query_id"], query=j["query"],
                         gold_answer=j["gold_answer"],
                         phase4_raw=j["phase4_raw"],
                         prompt_tokens=int(ins["input_ids"].shape[1]),
                         n_generated=int(len(gen)),
                         hit_cap=int(len(gen)) >= MAXNEW,
                         raw=raw, parsed_op=op, parse_reason=why,
                         computed=val,
                         is_correct=em(val, j["gold_answer"]) if val else 0,
                         latency_sec=round(time.time() - t0, 3)))
    res = {"model": MODEL, "revision": REV, "max_new_tokens_trial": MAXNEW,
           "ops_supported": list(OPS), "aggregation_inventory": inv,
           "n": len(rows),
           "n_parse_fail": sum(1 for r in rows if r["parse_reason"] != "ok"),
           "parse_fail_rate": round(
               sum(1 for r in rows if r["parse_reason"] != "ok") / len(rows), 4),
           "gen_tokens_min": min(r["n_generated"] for r in rows),
           "gen_tokens_max": max(r["n_generated"] for r in rows),
           "gen_tokens_mean": round(
               sum(r["n_generated"] for r in rows) / len(rows), 1),
           "n_hit_cap": sum(1 for r in rows if r["hit_cap"]),
           "rows": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"},
                     ensure_ascii=False, indent=2))
    for r in rows:
        print("---", r["query_id"], "| gold", r["gold_answer"], "| gen",
              r["n_generated"], "|", r["parse_reason"], "| computed",
              r["computed"], "| EM", r["is_correct"])
        print(r["raw"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
