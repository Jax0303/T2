#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Can P1_fixed_512 run on the 116 tables alignment drops?

A. does the markdown build for them at all, and does P1 chunk it
B. the 195 single-operand queries on those tables, retrieved for real
C. the arithmetic that follows

P4_path_cell is NOT run: its index unit is the cell sentence, whose row/col
header paths come from the aligned header tree. With no alignment there is no
verified path, so there is no sentence to index -- not a slower arm, an absent
one.
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
from point3_reconstruction_cost import build_table_paths             # noqa: E402
from qwen_equiv_k import (B_READER, MODEL, REV, args_for, count_all,  # noqa: E402
                          prompt_text)
from rag_agent.bench.hitab import load_queries                       # noqa: E402
from rag_agent.retrieve.encoders import default_encoder              # noqa: E402
from rag_agent.retrieve.hybrid_index import HybridIndex, _minmax     # noqa: E402
from stratified_recall import Chunk                                  # noqa: E402
from transformers import AutoTokenizer                               # noqa: E402

ALPHA, EMBED, TOPN, POLICY = 0.7, "BAAI/bge-small-en-v1.5", 200, "P1_fixed_512"
OUT = Path("results/audit/unaligned_p1.json")


def unaligned_tables():
    queries, tables = load_queries("data/hitab", "dev")
    raw_dir = Path("data/hitab/data/tables/raw")
    bad = {}
    for tid, bt in tables.items():
        f = raw_dir / f"{tid}.json"
        if not f.exists():
            bad[tid] = (bt, None)
            continue
        try:
            raw = json.load(open(f))
        except Exception:                                        # noqa: BLE001
            bad[tid] = (bt, None)
            continue
        if build_table_paths(raw, bt) is None:
            bad[tid] = (bt, raw)
    return queries, bad


def main() -> int:
    tokz = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    bud = Budget(EMBED)
    queries, bad = unaligned_tables()
    res = {"n_unaligned": len(bad)}

    # ---- Task A ----
    built, fail = {}, []
    for tid, (bt, raw) in bad.items():
        if raw is None:
            fail.append({"table": tid, "why": "no raw json"})
            continue
        nhr = int(raw.get("top_header_rows_num") or 0)
        nhc = int(raw.get("left_header_columns_num") or 0)
        texts = raw.get("texts") or []
        if not texts or nhr <= 0 or nhc <= 0:
            fail.append({"table": tid, "why": f"nhr={nhr} nhc={nhc} "
                                              f"rows={len(texts)}"})
            continue
        lines = markdown_table(raw, nhr)
        w = max(len(r) for r in texts)
        n_r, n_c = len(texts) - nhr, w - nhc
        if not lines or n_r <= 0 or n_c <= 0:
            fail.append({"table": tid, "why": f"empty data area {n_r}x{n_c}"})
            continue
        built[tid] = {"lines": lines, "shape": (n_r, n_c), "nhr": nhr,
                      "nhc": nhc, "bt": (bt.n_rows, bt.n_cols),
                      "dims_match": (n_r, n_c) == (bt.n_rows, bt.n_cols)}
    res["A1"] = {"n_markdown_built": len(built), "n_failed": len(fail),
                 "failures": fail[:20],
                 "n_dims_match_bt": sum(1 for v in built.values()
                                        if v["dims_match"])}

    # ---- extended corpus: the 424 aligned tables plus whatever built ----
    C = load_corpus(args_for("hitab", "hitab_dev_lookup_all"))
    by_table = defaultdict(dict)
    for n, (t, i, j) in enumerate(C.cell_owner):
        by_table[t][(i, j)] = n
    base_tids = set(C.tids)
    # every table that yields a markdown enters the haystack -- indexing only
    # the ones whose gold can be placed would hand the new queries an easier
    # corpus than the real one. Only the placeable ones get SCORED.
    add = dict(built)
    scoreable = {t for t, v in built.items() if v["dims_match"]}
    for tid, v in add.items():
        C.md_lines[tid] = v["lines"]
        C.shape[tid] = v["shape"]
        for i in range(v["shape"][0]):
            for j in range(v["shape"][1]):
                by_table[tid][(i, j)] = -1        # P1 needs the key, not a path

    chunks, owner = [], []
    n_chunk_base = 0
    for tid in list(C.tids) + sorted(add):
        have = by_table.get(tid)
        if not have:
            continue
        for ch in chunks_for(C, tid, POLICY, bud, have, "S3c"):
            chunks.append(Chunk(table_id=tid, chunk_id=ch["chunk_id"],
                                text=ch["text"], scheme=POLICY, kind="chunk"))
            owner.append((tid, ch["cells"]))
        if tid in base_tids:
            n_chunk_base = len(chunks)
    res["A3"] = {"n_chunks_total": len(chunks),
                 "n_chunks_aligned_424": n_chunk_base,
                 "n_chunks_added": len(chunks) - n_chunk_base,
                 "n_tables_added": len(add),
                 "added_tables_with_zero_chunks":
                     sorted(t for t in add
                            if not any(o[0] == t for o in owner))}

    # ---- Task B ----
    holds = defaultdict(list)
    for n, (tid, cells) in enumerate(owner):
        for (i, j) in cells:
            holds[(tid, i, j)].append(n)
    enc = cdv._CachedEncoder(default_encoder(model_name=EMBED),
                             ".cache/corpus_dump_vs_cell", f"hitab_dev_{EMBED}")
    t0 = time.time()
    ix = HybridIndex(chunks, encoder=enc, alpha=0.5)
    qt = count_all(tokz, [c.text + "\n" for c in chunks])
    print(f"[index] {len(chunks)} chunks in {time.time()-t0:.0f}s", flush=True)

    pop_all = [q for q in queries
               if q.gold_table_id in add and len(q.gold_operands) == 1]
    pop = [q for q in pop_all if q.gold_table_id in scoreable]
    hit, used, rows = [], [], []
    n_off_grid = 0
    for q in pop:
        o = q.gold_operands[0]
        tid, i, j = q.gold_table_id, o.row, o.col
        nr, nc = C.shape[tid]
        if not (0 <= i < nr and 0 <= j < nc):
            n_off_grid += 1
            continue
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
        kept = set(keep)
        h = int(any(c in kept for c in holds.get((tid, i, j), [])))
        hit.append(h)
        used.append(len(keep))
        rows.append({"query_id": str(q.query_id), "table": tid, "row": i,
                     "col": j, "n_chunks_used": len(keep), "gold_in_context": h})
    res["B"] = {"n_queries_on_built_tables": len(pop_all),
                "n_queries_eligible": len(pop), "n_scored": len(hit),
                "n_unscoreable_dims_mismatch": len(pop_all) - len(pop),
                "n_gold_off_grid": n_off_grid,
                "recall": round(float(np.mean(hit)), 6) if hit else None,
                "n_hit": int(sum(hit)),
                "chunks_used_mean": round(float(np.mean(used)), 2) if used else None,
                "chunks_used_min": int(min(used)) if used else None,
                "chunks_used_max": int(max(used)) if used else None,
                "index": "P1 chunks of the 424 aligned tables PLUS the added ones",
                "P4": "not run -- no verified header path, so no cell sentence"}
    res["B_rows"] = rows
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k != "B_rows"},
                     indent=2, ensure_ascii=False)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
