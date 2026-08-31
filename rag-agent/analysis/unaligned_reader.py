#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The 80 placeable queries on unaligned tables, read for real.

Same index (857 P1 chunks), same greedy fill, same prompt, same decoding and
same scorer as Phase 4. P4_path_cell has no sentence to index here, so only
P1_fixed_512 runs.
"""
from __future__ import annotations

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
from baseline_comparison_llm import Budget, markdown_table           # noqa: E402
from header_path_coverage import chunks_for, load_corpus             # noqa: E402
from phase4_summary import em                                        # noqa: E402
from point3_reconstruction_cost import build_table_paths             # noqa: E402
from qwen_equiv_k import (B_READER, MODEL, REV, SYS, args_for,       # noqa: E402
                          count_all, prompt_text)
from rag_agent.bench.hitab import load_queries                       # noqa: E402
from rag_agent.llm.local_qwen import LocalQwenLLM                    # noqa: E402
from rag_agent.retrieve.encoders import default_encoder              # noqa: E402
from rag_agent.retrieve.hybrid_index import HybridIndex, _minmax     # noqa: E402
from stratified_recall import Chunk                                  # noqa: E402
from transformers import AutoTokenizer                               # noqa: E402

ALPHA, EMBED, TOPN, POLICY = 0.7, "BAAI/bge-small-en-v1.5", 200, "P1_fixed_512"
SEED, MAXNEW = 42, 32
OUT = Path("results/audit/unaligned_reader.jsonl")


def main() -> int:
    import torch
    torch.manual_seed(SEED)
    tokz = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    bud = Budget(EMBED)

    queries, tables = load_queries("data/hitab", "dev")
    raw_dir = Path("data/hitab/data/tables/raw")
    built = {}
    for tid, bt in tables.items():
        f = raw_dir / f"{tid}.json"
        if not f.exists():
            continue
        try:
            raw = json.load(open(f))
        except Exception:                                        # noqa: BLE001
            continue
        if build_table_paths(raw, bt) is not None:
            continue
        nhr = int(raw.get("top_header_rows_num") or 0)
        nhc = int(raw.get("left_header_columns_num") or 0)
        texts = raw.get("texts") or []
        if not texts or nhr <= 0 or nhc <= 0:
            continue
        lines = markdown_table(raw, nhr)
        w = max(len(r) for r in texts)
        n_r, n_c = len(texts) - nhr, w - nhc
        if not lines or n_r <= 0 or n_c <= 0:
            continue
        built[tid] = {"lines": lines, "shape": (n_r, n_c),
                      "dims_match": (n_r, n_c) == (bt.n_rows, bt.n_cols)}

    C = load_corpus(args_for("hitab", "hitab_dev_lookup_all"))
    by_table = defaultdict(dict)
    for n, (t, i, j) in enumerate(C.cell_owner):
        by_table[t][(i, j)] = n
    for tid, v in built.items():
        C.md_lines[tid] = v["lines"]
        C.shape[tid] = v["shape"]
        for i in range(v["shape"][0]):
            for j in range(v["shape"][1]):
                by_table[tid][(i, j)] = -1

    chunks, owner = [], []
    for tid in list(C.tids) + sorted(built):
        have = by_table.get(tid)
        if not have:
            continue
        for ch in chunks_for(C, tid, POLICY, bud, have, "S3c"):
            chunks.append(Chunk(table_id=tid, chunk_id=ch["chunk_id"],
                                text=ch["text"], scheme=POLICY, kind="chunk"))
            owner.append((tid, ch["cells"]))
    holds = defaultdict(list)
    for n, (tid, cells) in enumerate(owner):
        for (i, j) in cells:
            holds[(tid, i, j)].append(n)

    enc = cdv._CachedEncoder(default_encoder(model_name=EMBED),
                             ".cache/corpus_dump_vs_cell", f"hitab_dev_{EMBED}")
    ix = HybridIndex(chunks, encoder=enc, alpha=0.5)
    qt = count_all(tokz, [c.text + "\n" for c in chunks])
    print(f"[index] {len(chunks)} chunks", flush=True)

    scoreable = {t for t, v in built.items() if v["dims_match"]}
    pop = [q for q in queries
           if q.gold_table_id in scoreable and len(q.gold_operands) == 1]
    print(f"[pop] {len(pop)}", flush=True)

    llm = LocalQwenLLM(model_name=MODEL, dtype="float16", quantization="4bit")
    fh = open(OUT, "w")
    t0 = time.time()
    for n, q in enumerate(pop, 1):
        o = q.gold_operands[0]
        tid, i, j = q.gold_table_id, o.row, o.col
        bm, dn = ix._bm25_scores(q.question), ix._dense_scores(q.question)
        s = ALPHA * _minmax(dn) + (1 - ALPHA) * _minmax(bm)
        n_top = min(TOPN, len(s))
        top = np.argpartition(-s, n_top - 1)[:n_top]
        top = top[np.argsort(-s[top])]
        ov = len(tokz(prompt_text(tokz, "", q.question),
                      add_special_tokens=False)["input_ids"])
        cum, keep = ov, []
        for p in top:
            if cum + int(qt[p]) > B_READER:
                break
            cum += int(qt[p])
            keep.append(int(p))
        ctx = "\n".join(chunks[p].text for p in keep)
        ptok = len(tokz(prompt_text(tokz, ctx, q.question),
                        add_special_tokens=False)["input_ids"])
        raw = llm.complete(SYS, f"CONTEXT:\n{ctx}\n\nQUESTION: {q.question}"
                                f"\n\nAnswer:", max_tokens=MAXNEW,
                           temperature=0.0)
        parsed = raw.strip().splitlines()[0].strip() if raw.strip() else ""
        gold = str(q.answer)
        fh.write(json.dumps({
            "query_id": str(q.query_id), "pool": "hitab_lookup_unaligned",
            "policy": POLICY, "table": tid, "row": i, "col": j,
            "question": q.question, "gold_answer": gold,
            "n_chunks_used": len(keep), "prompt_tokens": ptok,
            "gold_in_context": int(any(c in set(keep)
                                       for c in holds.get((tid, i, j), []))),
            "pred_answer_raw": raw, "pred_parsed": parsed,
            "is_correct": em(parsed, gold)}) + "\n")
        fh.flush()
        if n % 20 == 0:
            print(f"  {n}/{len(pop)} {time.time()-t0:.0f}s", flush=True)
    fh.close()
    rows = [json.loads(l) for l in open(OUT)]
    print(json.dumps({"n": len(rows),
                      "EM": round(sum(r["is_correct"] for r in rows) / len(rows), 6),
                      "n_correct": sum(r["is_correct"] for r in rows),
                      "recall": round(sum(r["gold_in_context"] for r in rows)
                                      / len(rows), 6)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
