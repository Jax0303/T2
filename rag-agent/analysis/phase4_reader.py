#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Phase 4 Task C -- the local reader over the frozen sample_294 contexts.

Three conditions per query: the two retrieved contexts Task B already filled
greedily to 4096 Qwen tokens, and gold_cell, which skips retrieval and injects
the gold cell sentences directly (every gold cell for a multi-gold query). One
row per (query, condition); 294 x 3 = 882.

Appends to a jsonl as it goes, so a crash costs the current query, not the run.
Re-running skips whatever the jsonl already holds.

  PYTHONPATH=. .venv/bin/python analysis/phase4_reader.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis"))

import corpus_dump_vs_cell as cdv                                    # noqa: E402
from baseline_comparison_llm import Budget                           # noqa: E402
from header_path_coverage import load_corpus                         # noqa: E402
from qwen_equiv_k import MODEL, REV, SYS, args_for                   # noqa: E402
from rag_agent.llm.local_qwen import LocalQwenLLM                    # noqa: E402
from stratified_recall import build_chunks                           # noqa: E402

SEED, MAXNEW = 42, 32
# PREREGISTER rev4: run pool by pool so an interrupted run leaves whole pools
# finished rather than a ragged slice of every pool, and cap gold_cell -- it is
# the reader ceiling, which needs far less n than the condition comparison.
ORDER = ["hitab_arith", "rhb_num", "aitqa", "rhb_fact", "hitab_lookup"]
GOLD_N = 30
# rev5 extends hitab_lookup to 189 and runs gold_cell on all of it
GOLD_CAP = {"hitab_lookup": 10 ** 9}
RET = [Path("results/phase4/retrieval_294"),
       Path("results/phase4/retrieval_ext129")]
OUT = Path("results/phase4/reader_records.jsonl")
POOL_DS = {"hitab_lookup": ("hitab", "hitab_dev_lookup_all"),
           "hitab_arith": ("hitab", "hitab_dev_corpus_arith"),
           "aitqa": ("aitqa", "aitqa"),
           "rhb_fact": ("realhitbench", "rhb_lookup_all"),
           "rhb_num": ("realhitbench", "rhb_lookup_all")}

# --- EM normalisation. Exactly the five rules the spec fixes, nothing else. ---
_CUR = re.compile(r"[$€£¥%]")
_THOU = re.compile(r"(?<=\d),(?=\d)")


def norm_em(s: str) -> str:
    s = _CUR.sub("", str(s))
    s = _THOU.sub("", s)                      # thousands comma only: digit,digit
    s = " ".join(s.split()).strip().lower()
    if re.fullmatch(r"-?\d+\.\d+", s):        # trailing zeros: numbers only
        s = s.rstrip("0").rstrip(".")
    return s


def em(pred: str, gold: str) -> int:
    """A multi-gold query is still scored on the ONE answer string the dataset
    ships: multiple gold cells change what gets retrieved/injected, not what
    counts as right. No per-cell partial credit."""
    return int(norm_em(pred) == norm_em(gold))


def main() -> int:
    import torch
    torch.manual_seed(SEED)

    bud = Budget("BAAI/bge-small-en-v1.5")
    # gold cell sentence for the gold_cell condition = the P4 chunk text
    cell_text, corpora = {}, {}
    for ds, pop in (("hitab", "hitab_dev_lookup_all"), ("aitqa", "aitqa"),
                    ("realhitbench", "rhb_lookup_all")):
        C = corpora[ds] = load_corpus(args_for(ds, pop))
        by = defaultdict(dict)
        for n, (t, i, j) in enumerate(C.cell_owner):
            by[t][(i, j)] = n
        chunks, owner = build_chunks(C, "P4_path_cell", bud, by, "S3c")
        for c, (tid, cells) in zip(chunks, owner):
            for (i, j) in cells:
                cell_text[(tid, i, j)] = c.text
        print(f"[cells] {ds} {len(chunks)}", flush=True)

    # every (query, condition) to run, built off the Task B files
    import random
    jobs = []
    for pool in ORDER:
        per = {}
        for pol in ("P1_fixed_512", "P4_path_cell"):
            for rd in RET:
                f = rd / f"{pol}_{pool}.jsonl"
                if not f.exists():
                    continue
                for line in open(f):
                    d = json.loads(line)
                    jobs.append((pol, d))
                    per.setdefault(d["query_id"], d)
        cap = GOLD_CAP.get(pool, GOLD_N)
        pick = random.Random(SEED).sample(sorted(per), min(cap, len(per)))
        jobs += [("gold_cell", per[q]) for q in pick]

    done = set()
    if OUT.exists():
        for line in open(OUT):
            r = json.loads(line)
            done.add((r["policy"], r["pool"], r["query_id"]))
    print(f"[jobs] {len(jobs)} total, {len(done)} already done", flush=True)

    import csv as _csv
    sample = {}
    for f in ("results/phase4/sample_294.csv",
              "results/phase4/sample_hitab_lookup_ext129.csv"):
        for r in _csv.DictReader(open(f)):
            sample[(r["pool"], r["query_id"])] = r

    llm = LocalQwenLLM(model_name=MODEL, quantization="4bit")
    tok = llm.tokenizer
    fh = open(OUT, "a")
    t_start = time.time()
    for n, (pol, d) in enumerate(jobs, 1):
        pool, qid = d["pool"], d["query_id"]
        if (pol, pool, qid) in done:
            continue
        srow = sample[(pool, qid)]
        gold = [tuple(g) for g in d["gold_cells"]]
        if pol == "gold_cell":
            keep = []
            ctx = "\n".join(cell_text[(t, i, j)] for (t, i, j) in gold)
            n_used, cap, in_top, grank = len(gold), False, True, 0
        else:
            keep = [c for c in d["topk"] if c["used"]]
            ctx = "\n".join(c["text"] for c in keep)
            n_used, cap = d["n_chunks_used"], d["n_chunks_used"] == 200
            used_cells = {tuple(x) for c in keep for x in
                          [[c["table_id"]] + list(cc) for cc in c["cells"]]}
            in_top = any(g in used_cells for g in gold)
            rk = [c["rank"] for c in d["topk"]
                  for cc in c["cells"] if (c["table_id"], cc[0], cc[1]) in gold]
            grank = min(rk) if rk else None
        user = f"CONTEXT:\n{ctx}\n\nQUESTION: {d['question']}\n\nAnswer:"
        ptok = len(tok(tok.apply_chat_template(
            [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False),
            add_special_tokens=False)["input_ids"])
        t0 = time.time()
        raw = llm.complete(system=SYS, user=user, max_tokens=MAXNEW, temperature=0.0)
        lat = time.time() - t0
        parsed = raw.strip().splitlines()[0].strip() if raw.strip() else ""
        fh.write(json.dumps({
            "query_id": qid, "query_type": srow["query_type"], "dataset": d["dataset"],
            "pool": pool, "policy": pol, "n_chunks_used": n_used,
            "hit_chunk_cap": cap, "query": d["question"],
            "gold_answer": srow["gold_answer"], "gold_table_id": srow["gold_table_id"],
            "gold_cell": json.dumps([list(g) for g in gold]),
            "retrieved_topk": json.dumps(
                [{"rank": c["rank"], "table_id": c["table_id"],
                  "chunk_id": c["chunk_id"], "cells": c["cells"],
                  "text": c["text"]} for c in keep]),
            "gold_in_topk": in_top,
            "gold_all_in_topk": (all(g in used_cells for g in gold)
                                 if pol != "gold_cell" else True),
            "gold_rank": grank, "pred_answer_raw": raw, "pred_parsed": parsed,
            "is_correct": em(parsed, srow["gold_answer"]),
            "prompt_tokens": ptok, "latency_sec": round(lat, 3)}) + "\n")
        fh.flush()
        if n % 25 == 0:
            el = time.time() - t_start
            print(f"  {n}/{len(jobs)}  {el/60:.1f}min elapsed", flush=True)
    fh.close()
    print("-> " + str(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
