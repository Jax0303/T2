#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Task D -- HiTab P4 with reconstructed paths on both axes, and P1 with a
guessed header boundary. Writes under results/audit/taskd/; nothing existing
is overwritten."""
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
from baseline_comparison_llm import Budget, markdown_table           # noqa: E402
from header_path_coverage import TEMPLATE, cell_spans, load_corpus   # noqa: E402
from phase4_summary import em                                        # noqa: E402
from point3_reconstruction_cost import build_table_paths             # noqa: E402
from qwen_equiv_k import (B_READER, MODEL, REV, SYS, args_for,       # noqa: E402
                          count_all, prompt_text)
from rag_agent.bench.hitab import load_queries                       # noqa: E402
from rag_agent.llm.local_qwen import LocalQwenLLM                    # noqa: E402
from rag_agent.reconstruct.header_grid import (guess_n_header_cols,  # noqa: E402
                                               guess_n_header_rows)
from rag_agent.retrieve.encoders import default_encoder              # noqa: E402
from rag_agent.retrieve.hybrid_index import HybridIndex, _minmax     # noqa: E402
from rag_agent.serialization.caption import caption_sentence         # noqa: E402
from stratified_recall import Chunk                                  # noqa: E402
from transformers import AutoTokenizer                               # noqa: E402

ALPHA, EMBED, TOPN, SCHEME = 0.7, "BAAI/bge-small-en-v1.5", 200, "S3c"
SEED, MAXNEW = 42, 32
OUT = Path("results/audit/taskd")
ARMS = [("P4_recon_row", "rec", "gold", "P4"),      # rec_rp + gold_cp
        ("P4_recon", "rec", "rec", "P4"),           # rec_rp + rec_cp
        ("P1_guessed_boundary", None, None, "P1")]


def samples():
    look = [r for r in csv.DictReader(open("results/phase4/sample_294.csv"))
            if r["pool"] == "hitab_lookup"]
    look += list(csv.DictReader(
        open("results/phase4/sample_hitab_lookup_ext129.csv")))
    ari = [r for r in csv.DictReader(open("results/phase4/sample_294.csv"))
           if r["pool"] == "hitab_arith"]
    return {"hitab_lookup": look, "hitab_arith": ari}


def main() -> int:
    import torch
    torch.manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    tokz = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    bud = Budget(EMBED)

    C = load_corpus(args_for("hitab", "hitab_dev_lookup_all"))
    Ca = load_corpus(args_for("hitab", "hitab_dev_corpus_arith"))
    qmap = {"hitab_lookup": {str(q["query_id"]): q for q in C.queries},
            "hitab_arith": {str(q["query_id"]): q for q in Ca.queries}}
    by_table = defaultdict(dict)
    for n, (t, i, j) in enumerate(C.cell_owner):
        by_table[t][(i, j)] = n

    _, tables = load_queries("data/hitab", "dev")
    raw_dir = Path("data/hitab/data/tables/raw")
    rec_r, rec_c, boundary = {}, {}, {}
    for tid in C.tids:
        bt, f = tables.get(tid), raw_dir / f"{tid}.json"
        if bt is None or not f.exists():
            continue
        raw = json.load(open(f))
        pt = build_table_paths(raw, bt)
        if pt is None:
            continue
        for i in range(pt["n_r"]):
            rec_r[(tid, i)] = list(pt["rec_rp"][i])
        for j in range(pt["n_c"]):
            rec_c[(tid, j)] = list(pt["rec_cp"][j])
        texts = [[str(x) if x is not None else "" for x in r]
                 for r in raw["texts"]]
        w = max(len(r) for r in texts)
        texts = [r + [""] * (w - len(r)) for r in texts]
        gc = guess_n_header_cols(texts)
        gr = max(1, min(guess_n_header_rows(texts, n_header_cols=gc),
                        len(texts) - 1))
        gc = guess_n_header_cols(texts, n_header_rows=gr)
        boundary[tid] = {"oracle_nhr": max(1, len(raw["texts"]) - pt["n_r"]),
                         "guess_nhr": gr, "guess_nhc": gc,
                         "n_r_oracle": pt["n_r"], "n_c_oracle": pt["n_c"],
                         "n_r_guess": len(texts) - gr, "n_c_guess": w - gc,
                         "raw": raw}
    agree = sum(1 for v in boundary.values()
                if v["oracle_nhr"] == v["guess_nhr"])
    print(f"[boundary] nhr guess == oracle on {agree}/{len(boundary)} tables",
          flush=True)

    smp = samples()
    llm = None
    prev = (json.loads((OUT / "summary.json").read_text())
            if (OUT / "summary.json").exists() else {})
    summary = {"boundary_nhr_agree": agree, "n_tables": len(boundary),
               "arms": prev.get("arms", {})}

    for arm, rsrc, csrc, kind in ARMS:
        if all((OUT / f"{arm}_{p}.jsonl").exists()
               for p in ("hitab_lookup", "hitab_arith")):
            print(f"[skip] {arm} complete", flush=True)
            continue
        t0 = time.time()
        chunks, owner = [], []
        if kind == "P4":
            for tid in C.tids:
                for (i, j), n in sorted(by_table[tid].items()):
                    grp, gcp, v = C.cell_paths[n]
                    rp = rec_r.get((tid, i), list(grp)) if rsrc == "rec" else list(grp)
                    cp = rec_c.get((tid, j), list(gcp)) if csrc == "rec" else list(gcp)
                    chunks.append(Chunk(
                        table_id=tid, chunk_id=f"{tid}::cell::{i}:{j}",
                        text=caption_sentence(C.title.get(tid, ""), rp, cp, v,
                                              template=TEMPLATE[SCHEME]),
                        scheme="P4_path_cell", kind="chunk"))
                    owner.append((tid, {(i, j)}))
        else:
            n_tok = 512
            for tid in C.tids:
                b = boundary.get(tid)
                if b is None:
                    continue
                lines = markdown_table(b["raw"], b["guess_nhr"])
                nr, nc = b["n_r_guess"], b["n_c_guess"]
                if nr <= 0 or nc <= 0 or not lines:
                    continue
                # a throwaway view so cell_spans reads the GUESSED layout
                class V:
                    md_lines = {tid: lines}
                    shape = {tid: (nr, nc)}
                have = {(i, j): 0 for i in range(nr) for j in range(nc)}
                spans = cell_spans(V, tid, have)
                text = "\n".join(lines)
                offs = bud.tok(text, add_special_tokens=False,
                               return_offsets_mapping=True)["offset_mapping"]
                for k in range(0, len(offs), n_tok):
                    win = offs[k:k + n_tok]
                    a, bnd = win[0][0], win[-1][1]
                    chunks.append(Chunk(
                        table_id=tid,
                        chunk_id=f"{tid}::fixed{n_tok}::{k // n_tok}",
                        text=text[a:bnd], scheme="P1_fixed_512", kind="chunk"))
                    owner.append((tid, {ij for ij, (s, e) in spans.items()
                                        if a <= s and e <= bnd}))
        holds = defaultdict(list)
        for n, (tid, cells) in enumerate(owner):
            for (i, j) in cells:
                holds[(tid, i, j)].append(n)
        enc = cdv._CachedEncoder(default_encoder(model_name=EMBED),
                                 ".cache/corpus_dump_vs_cell",
                                 f"hitab_dev_{arm}_{EMBED}")
        ix = HybridIndex(chunks, encoder=enc, alpha=0.5)
        qt = count_all(tokz, [c.text + "\n" for c in chunks])
        print(f"[{arm}] {len(chunks)} chunks, {time.time()-t0:.0f}s", flush=True)

        if llm is None:
            llm = LocalQwenLLM(model_name=MODEL, dtype="float16",
                               quantization="4bit")
        for pool in ("hitab_lookup", "hitab_arith"):
            f_out = OUT / f"{arm}_{pool}.jsonl"
            if f_out.exists():
                print(f"[skip] {f_out} exists", flush=True)
                continue
            fh = open(f_out, "w")
            gcell_hit, gcell_n, corr, used = 0, 0, 0, []
            t1 = time.time()
            for n, r in enumerate(smp[pool], 1):
                qid = str(r["query_id"])
                q = qmap[pool][qid]
                bm = ix._bm25_scores(q["question"])
                dn = ix._dense_scores(q["question"])
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
                gold = [(str(g[0]), int(g[1]), int(g[2]))
                        for g in json.loads(r["gold_cells"])]
                hits = [int(any(c in kept for c in holds.get(g, [])))
                        for g in gold]
                gcell_hit += sum(hits)
                gcell_n += len(hits)
                raw = llm.complete(
                    SYS, f"CONTEXT:\n{ctx}\n\nQUESTION: {q['question']}"
                         f"\n\nAnswer:", max_tokens=MAXNEW, temperature=0.0)
                parsed = raw.strip().splitlines()[0].strip() if raw.strip() else ""
                ok = em(parsed, r["gold_answer"])
                corr += ok
                used.append(len(keep))
                fh.write(json.dumps({
                    "query_id": qid, "arm": arm, "pool": pool,
                    "question": q["question"], "gold_answer": r["gold_answer"],
                    "gold_cells": [list(g) for g in gold],
                    "n_chunks_used": len(keep),
                    "gold_in_topk": bool(hits and all(hits)),
                    "n_gold_hit": sum(hits), "n_gold": len(hits),
                    "pred_answer_raw": raw, "pred_parsed": parsed,
                    "is_correct": ok}) + "\n")
                fh.flush()
                if n % 50 == 0:
                    print(f"  {arm}/{pool} {n}/{len(smp[pool])} "
                          f"{time.time()-t1:.0f}s", flush=True)
            fh.close()
            summary["arms"][f"{arm}|{pool}"] = {
                "n_queries": len(smp[pool]), "n_gold_cells": gcell_n,
                "recall": round(gcell_hit / gcell_n, 6) if gcell_n else None,
                "n_gold_hit": gcell_hit,
                "EM": round(corr / len(smp[pool]), 6), "n_correct": corr,
                "chunks_used_mean": round(float(np.mean(used)), 2)}
            (OUT / "summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False))
            print(json.dumps(summary["arms"][f"{arm}|{pool}"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
