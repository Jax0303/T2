#!/usr/bin/env python3
"""Full RAG pipeline: retrieve -> chunk -> compute, measured on NUMERIC accuracy.

Addresses the most critical review point (Liner #2): the structure-loss claim is
about RAG chunking, so the eval must contain a real retrieval stage. Here a table
is split into row-chunks; BM25 retrieves the top-k chunks for the question
(retrieval is held IDENTICAL across conditions); the retrieved chunks are then
serialized to the LLM at three structure-preservation levels, and the LLM computes
the answer. This isolates: within a real retrieve->chunk pipeline, does
structure-preserving serialization of the retrieved fragment protect numeric
computation?

  chunk_rows : rows per chunk (smaller = more fragmentation / more boundary loss)
  top_k      : chunks fed to the LLM (smaller = lossier retrieval)
  retrieval text = flat_leaf serialization of each chunk (same for all conditions)
  conditions = flat_values | flat_leaf | header_path  (serialization of retrieved chunks)

Reports, per condition: numeric|exact accuracy. Plus retrieval recall (did the
retrieved chunks contain a gold answer value?) so we know retrieval is non-trivial.
seed=42, Groq LLM. Reuses codegen_chunk_eval serialization + builders.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                                    # noqa: E402
from rank_bm25 import BM25Okapi                                       # noqa: E402
from rag_agent.eval.metrics import numeric_match, exact_match         # noqa: E402
from rag_agent.llm.factory import build_llm                           # noqa: E402
from codegen_chunk_eval import (serialize, build_hier, build_flat, ANSWER_SYS,  # noqa: E402
                                build_prompt, clean_pred, complete_retry, boot_ci)

SEED = 42
CONDS = ["flat_values", "flat_leaf", "header_path"]
_TOK = re.compile(r"[a-z0-9.]+")


def tok(s):
    return _TOK.findall(str(s).lower())


def chunks_of(n_rows, chunk_rows):
    return [list(range(i, min(i + chunk_rows, n_rows))) for i in range(0, n_rows, chunk_rows)]


def gold_values(answers):
    out = set()
    for a in answers:
        s = str(a).strip().lower()
        out.add(s)
        try:
            out.add(str(float(str(a).replace(",", ""))))
        except (ValueError, TypeError):
            pass
    return out


def row_has_gold(ot, r, gv):
    for v in ot.data[r]:
        s = str(v).strip().lower()
        if s in gv:
            return True
        try:
            if str(float(str(v).replace(",", ""))) in gv:
                return True
        except (ValueError, TypeError):
            pass
    return False


def retrieve(ot, question, chunk_rows, top_k, mode="bm25", gv=None):
    """Return (union of retrieved row indices, retrieved chunk indices, all chunks).
    mode=bm25           : top-k chunks by BM25(question, chunk_text)  [baseline RAG]
    mode=operand_oracle : the top-k chunks that actually CONTAIN a gold operand cell
                          (upper bound on what operand-complete retrieval can reach;
                          isolates 'is retrieval completeness the bottleneck?')."""
    chs = chunks_of(ot.n_rows, chunk_rows)
    if mode == "operand_oracle" and gv:
        scored = [(sum(1 for r in ch if row_has_gold(ot, r, gv)), -ci, ci) for ci, ch in enumerate(chs)]
        scored.sort(reverse=True)
        order = [ci for hits, _, ci in scored if hits > 0][:top_k]
        if not order:                       # no gold chunk -> fall back to first k
            order = list(range(min(top_k, len(chs))))
    else:
        texts = [serialize(ot, "flat_leaf", rows=ch) for ch in chs]
        bm = BM25Okapi([tok(t) for t in texts])
        scores = bm.get_scores(tok(question))
        order = list(np.argsort(-scores)[:top_k])
    rows = sorted({r for ci in order for r in chs[ci]})
    return rows, order, chs


def run(name, samples, llm, args):
    print(f"\n=== {name}: n={len(samples)} | chunk_rows={args.chunk_rows} top_k={args.top_k} ===", flush=True)
    per = {c: [] for c in CONDS}
    recall = []
    rows_out = []
    t0 = time.time()
    for i, s in enumerate(samples, 1):
        ot = s["ot"]
        gv = gold_values(s["answers"])
        rrows, order, chs = retrieve(ot, s["question"], args.chunk_rows, args.top_k,
                                     mode=args.retrieval, gv=gv)
        got_gold = any(row_has_gold(ot, r, gv) for r in rrows)
        recall.append(int(got_gold))
        rec = {"id": s["id"], "q": s["question"], "gold": s["answers"],
               "n_chunks": len(chs), "retrieved_rows": rrows, "got_gold": got_gold,
               "preds": {}, "ok": {}}
        for cond in CONDS:
            text = serialize(ot, cond, rows=rrows)
            prompt = build_prompt(getattr(ot, "title", "") or "", s["question"], text)
            try:
                pred = clean_pred(complete_retry(llm, ANSWER_SYS, prompt, args.max_tokens))
            except Exception as e:  # noqa: BLE001
                pred = ""
                rec.setdefault("err", {})[cond] = type(e).__name__
            ok = numeric_match(pred, s["answers"]) or exact_match(pred, s["answers"])
            rec["preds"][cond] = pred
            rec["ok"][cond] = bool(ok)
            per[cond].append(int(bool(ok)))
            time.sleep(args.sleep)
        rows_out.append(rec)
        if i % 10 == 0 or i == len(samples):
            acc = {c: np.mean(per[c]) for c in CONDS}
            print(f"  {i}/{len(samples)} R@chunk={np.mean(recall):.2f} " +
                  " ".join(f"{c}={acc[c]:.2f}" for c in CONDS) + f"  {time.time()-t0:.0f}s", flush=True)
    return per, recall, rows_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--splits", default="hier")
    ap.add_argument("--chunk-rows", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--retrieval", choices=["bm25", "operand_oracle"], default="bm25")
    ap.add_argument("--llm", default="groq:llama-3.1-8b-instant")
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--sleep", type=float, default=0.25)
    ap.add_argument("--out", default="results/codegen/rag_chunk.json")
    args = ap.parse_args()

    llm = build_llm(args.llm)
    builders = {"flat": build_flat, "hier": build_hier}
    out = {"config": vars(args), "splits": {}}
    for name in args.splits.split(","):
        name = name.strip()
        samples = builders[name](args.n)
        per, recall, rows_out = run(name, samples, llm, args)
        acc = {c: round(float(np.mean(per[c])), 4) for c in CONDS}
        contrasts = {}
        for a, b in [("flat_leaf", "header_path"), ("flat_values", "header_path")]:
            d = [y - x for x, y in zip(per[a], per[b])]
            m, lo, hi = boot_ci(d)
            contrasts[f"{b}-{a}"] = {"delta": round(m, 4), "ci": [round(lo, 4), round(hi, 4)],
                                     "sig": bool(lo > 0 or hi < 0)}
        out["splits"][name] = {"n": len(samples), "acc": acc, "chunk_recall": round(float(np.mean(recall)), 4),
                               "contrasts": contrasts, "rows": rows_out}
        print(f"  [{name}] chunk_recall={np.mean(recall):.3f} acc={acc}")
        for k, v in contrasts.items():
            print(f"    {k}: {v['delta']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]{'*' if v['sig'] else ''}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
