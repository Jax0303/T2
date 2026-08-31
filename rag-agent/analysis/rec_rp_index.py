#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Task C -- index HiTab's cell sentences with the RECONSTRUCTED row path.

Everything else is held: same corpus, same column paths (gold), same values,
same S3c template, same alpha, same greedy fill, same reader, same scorer.
Only pt["gold_rp"] -> pt["rec_rp"] moves. Writes to its own files; nothing
under results/phase4 is touched.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis"))

import numpy as np                                                   # noqa: E402
import corpus_dump_vs_cell as cdv                                    # noqa: E402
from baseline_comparison_llm import Budget                           # noqa: E402
from header_path_coverage import TEMPLATE, load_corpus               # noqa: E402
from phase4_summary import em                                        # noqa: E402
from point3_reconstruction_cost import build_table_paths             # noqa: E402
from qwen_equiv_k import (B_READER, MODEL, REV, SYS, args_for,       # noqa: E402
                          count_all, prompt_text)
from rag_agent.bench.hitab import load_queries                       # noqa: E402
from rag_agent.llm.local_qwen import LocalQwenLLM                    # noqa: E402
from rag_agent.retrieve.encoders import default_encoder              # noqa: E402
from rag_agent.retrieve.hybrid_index import HybridIndex, _minmax     # noqa: E402
from rag_agent.serialization.caption import caption_sentence         # noqa: E402
from stratified_recall import Chunk                                  # noqa: E402
from transformers import AutoTokenizer                               # noqa: E402

ALPHA, EMBED, TOPN, SCHEME = 0.7, "BAAI/bge-small-en-v1.5", 200, "S3c"
SEED, MAXNEW = 42, 32
OUT = Path("results/audit/rec_rp")


def main() -> int:
    import torch
    torch.manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    tokz = AutoTokenizer.from_pretrained(MODEL, revision=REV)

    C = load_corpus(args_for("hitab", "hitab_dev_lookup_all"))
    by_table = defaultdict(dict)
    for n, (t, i, j) in enumerate(C.cell_owner):
        by_table[t][(i, j)] = n

    # rec_rp per (table, row)
    _, tables = load_queries("data/hitab", "dev")
    raw_dir = Path("data/hitab/data/tables/raw")
    rec = {}
    for tid in C.tids:
        bt = tables.get(tid)
        f = raw_dir / f"{tid}.json"
        if bt is None or not f.exists():
            continue
        pt = build_table_paths(json.load(open(f)), bt)
        if pt is None:
            continue
        for i in range(pt["n_r"]):
            rec[(tid, i)] = list(pt["rec_rp"][i])
    print(f"[rec_rp] {len(rec)} rows", flush=True)

    chunks, owner, n_no_rec = [], [], 0
    for tid in C.tids:
        for (i, j), n in sorted(by_table[tid].items()):
            _rp, cp, v = C.cell_paths[n]
            rp = rec.get((tid, i))
            if rp is None:
                rp = list(_rp)
                n_no_rec += 1
            txt = caption_sentence(C.title.get(tid, ""), rp, cp, v,
                                   template=TEMPLATE[SCHEME])
            chunks.append(Chunk(table_id=tid, chunk_id=f"{tid}::cell::{i}:{j}",
                                text=txt, scheme="P4_path_cell", kind="chunk"))
            owner.append((tid, {(i, j)}))
    print(f"[chunks] {len(chunks)} (no rec_rp fallback: {n_no_rec})", flush=True)

    holds = defaultdict(list)
    for n, (tid, cells) in enumerate(owner):
        for (i, j) in cells:
            holds[(tid, i, j)].append(n)

    enc = cdv._CachedEncoder(default_encoder(model_name=EMBED),
                             ".cache/corpus_dump_vs_cell",
                             f"hitab_dev_recrp_{EMBED}")
    t0 = time.time()
    ix = HybridIndex(chunks, encoder=enc, alpha=0.5)
    qt = count_all(tokz, [c.text + "\n" for c in chunks])
    print(f"[index] {time.time()-t0:.0f}s", flush=True)

    sample = [r for r in csv.DictReader(open("results/phase4/sample_294.csv"))
              if r["pool"] == "hitab_lookup"]
    sample += list(csv.DictReader(
        open("results/phase4/sample_hitab_lookup_ext129.csv")))
    qmap = {str(q["query_id"]): q for q in C.queries}

    llm = LocalQwenLLM(model_name=MODEL, dtype="float16", quantization="4bit")
    fh = open(OUT / "records.jsonl", "w")
    t0 = time.time()
    for n, r in enumerate(sample, 1):
        qid = str(r["query_id"])
        q = qmap[qid]
        bm, dn = ix._bm25_scores(q["question"]), ix._dense_scores(q["question"])
        s = ALPHA * _minmax(dn) + (1 - ALPHA) * _minmax(bm)
        n_top = min(TOPN, len(s))
        top = np.argpartition(-s, n_top - 1)[:n_top]
        top = top[np.argsort(-s[top])]
        ov = len(tokz(prompt_text(tokz, "", q["question"]),
                      add_special_tokens=False)["input_ids"])
        cum, keep = ov, []
        for p in top:
            if cum + int(qt[p]) > B_READER:
                break
            cum += int(qt[p])
            keep.append(int(p))
        kept = set(keep)
        ctx = "\n".join(chunks[p].text for p in keep)
        ptok = len(tokz(prompt_text(tokz, ctx, q["question"]),
                        add_special_tokens=False)["input_ids"])
        gold = [tuple(g) for g in json.loads(r["gold_cells"])]
        hit = int(all(any(c in kept for c in holds.get((str(t), i, j), []))
                      for (t, i, j) in gold)) if gold else 0
        raw = llm.complete(SYS, f"CONTEXT:\n{ctx}\n\nQUESTION: {q['question']}"
                                f"\n\nAnswer:", max_tokens=MAXNEW,
                           temperature=0.0)
        parsed = raw.strip().splitlines()[0].strip() if raw.strip() else ""
        fh.write(json.dumps({
            "query_id": qid, "index": "rec_rp", "policy": "P4_path_cell",
            "question": q["question"], "gold_answer": r["gold_answer"],
            "gold_cells": [list(g) for g in gold],
            "n_chunks_used": len(keep), "prompt_tokens": ptok,
            "gold_in_topk": bool(hit), "pred_answer_raw": raw,
            "pred_parsed": parsed,
            "is_correct": em(parsed, r["gold_answer"])}) + "\n")
        fh.flush()
        if n % 40 == 0:
            print(f"  {n}/{len(sample)} {time.time()-t0:.0f}s", flush=True)
    fh.close()

    rows = [json.loads(l) for l in open(OUT / "records.jsonl")]
    res = {"n": len(rows),
           "recall": round(sum(r["gold_in_topk"] for r in rows) / len(rows), 6),
           "EM": round(sum(r["is_correct"] for r in rows) / len(rows), 6),
           "n_hit": sum(r["gold_in_topk"] for r in rows),
           "n_correct": sum(r["is_correct"] for r in rows),
           "chunks_used_mean": round(
               sum(r["n_chunks_used"] for r in rows) / len(rows), 2)}
    (OUT / "summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
