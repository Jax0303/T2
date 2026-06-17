#!/usr/bin/env python3
"""F2+F4 — does structure loss in the RAG representation break NUMERIC computation,
and does a structure-preserving serialization restore it?

Gap (verified against TableRAG/GTR/STC/HD-RAG, 2024-2026): structure-preserving
table chunkers are evaluated on retrieval recall@k only; arithmetic-capable table
systems sidestep chunking by offloading to code. Nobody closes the loop by
MEASURING a chunking/serialization scheme on END-TO-END numeric-answer accuracy
over hierarchical tables. This harness does exactly that.

Design (retrieval held fixed → answer stage isolated, like method_grounded):
  the SAME table cells are serialized to text at three levels of structure
  preservation, then the LLM reads the text and computes the answer directly
  (this is how RAG QA actually works: retrieved text -> answer).

  serialization condition (independent variable = degree of structure kept):
    S0 flat_values : data values only, no headers      (a mid-table chunk whose
                                                         header row was lost)
    S1 flat_leaf   : leaf column header + leaf row label (a header-aware chunker
                                                         that flattens hierarchy)
    S2 header_path : FULL hierarchical header path on every row/column
                                                        (structure-preserving)

  table type (stratifier): flat (WikiSQL, exact SQL gold) vs hier (HiTab).
  hypothesis: S2 >> S1 >> S0 on HIER and the S2-S1 gap GROWS with hierarchy,
              while on FLAT the conditions converge (no hierarchy to lose) —
              i.e. the benefit is caused by structure, not by verbosity.

  metric = numeric_match | exact_match of the model's computed answer.
  paired across conditions (same question), paired bootstrap CI, seed=42.
  LLM = Groq (CPU-friendly). No gold answer ever shown to the model.

usage:
  python scripts/codegen_chunk_eval.py --n 40 --llm groq:llama-3.3-70b-versatile
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                                    # noqa: E402
from rag_agent.eval.metrics import numeric_match, exact_match         # noqa: E402
from rag_agent.llm.factory import build_llm                           # noqa: E402
from rag_agent.data.loader import load_samples, load_table            # noqa: E402
from rag_agent.stores.original_store import build_original_table      # noqa: E402
from codegen_flat_eval import load_wikisql, flat_table, complete_retry  # noqa: E402

SEED = 42
CONDS = ["flat_values", "flat_leaf", "header_path"]


# ───────────────────────── serialization (the 3 levels) ─────────────────────────
def _leaf(path):
    return path[-1] if path else ""


def _full(path):
    return " > ".join(p for p in path if p)


def serialize(ot, cond, max_rows=60):
    """Render the table to text at a given structure-preservation level.
    Rows are capped (chunk budget) deterministically from the top."""
    n = min(ot.n_rows, max_rows)
    col_leaf = [_leaf(ot.col_path(c)) for c in range(ot.n_cols)]
    col_full = [_full(ot.col_path(c)) for c in range(ot.n_cols)]
    lines = []
    if cond == "flat_values":
        # no headers at all — a chunk that lost its header row
        for r in range(n):
            lines.append(" | ".join(str(v) for v in ot.data[r]))
        return "\n".join(lines)
    if cond == "flat_leaf":
        # leaf headers only — hierarchy parents flattened away
        lines.append("row | " + " | ".join(col_leaf))
        for r in range(n):
            lab = _leaf(ot.row_path(r))
            lines.append(f"{lab} | " + " | ".join(str(v) for v in ot.data[r]))
        return "\n".join(lines)
    # header_path — full hierarchical path on every column and every row
    lines.append("columns: " + " ;; ".join(f"[{cf}]" for cf in col_full))
    for r in range(n):
        rp = _full(ot.row_path(r))
        cells = [f"{col_full[c]} = {ot.data[r][c]}" for c in range(ot.n_cols)]
        lines.append(f"({rp}) :: " + " ; ".join(cells))
    return "\n".join(lines)


ANSWER_SYS = (
    "You answer a question using ONLY the table excerpt provided. "
    "Read the structure carefully, compute if needed (sum/min/max/count/difference), "
    "and output ONLY the final answer value (a number or short string), no explanation.")


def build_prompt(title, question, table_text):
    return (f"Table title: {title}\n\nTable excerpt:\n{table_text}\n\n"
            f"Question: {question}\n\nFinal answer (value only):")


def clean_pred(txt):
    t = txt.strip().splitlines()
    t = t[-1] if t else ""
    for pre in ("answer:", "final answer:", "the answer is"):
        if t.lower().startswith(pre):
            t = t[len(pre):].strip()
    return t.strip().strip(".").strip()


# ───────────────────────── data builders ─────────────────────────
def build_flat(n, split="dev"):
    """WikiSQL with exact SQL gold; cap rows so the table fits a chunk budget."""
    exs = load_wikisql(n * 2, SEED, split=split)
    out = []
    for e in exs:
        ot = flat_table(e["id"], "", e["table"]["header"], e["table"]["rows"])
        out.append({"id": e["id"], "question": e["question"], "answers": e["answers"], "ot": ot})
        if len(out) >= n:
            break
    return out


def build_hier(n, data_dir="data/hitab", split="dev"):
    raw_samples = load_samples(data_dir, split)
    rng = random.Random(SEED)
    rng.shuffle(raw_samples)
    tcache, out = {}, []
    for s in raw_samples:
        tid, ans = s.get("table_id"), s.get("answer")
        if not tid or not ans:
            continue
        if tid not in tcache:
            raw = load_table(tid, data_dir)
            tcache[tid] = build_original_table(raw) if raw else None
        ot = tcache[tid]
        if ot is None or ot.n_rows == 0:
            continue
        out.append({"id": s["id"], "question": s["question"],
                    "answers": [str(a) for a in ans], "ot": ot,
                    "agg": (s.get("aggregation") or ["none"])[0]})
        if len(out) >= n:
            break
    return out


def run_split(name, samples, llm, max_tokens, sleep, max_rows):
    print(f"\n=== {name}: n={len(samples)} ===", flush=True)
    per = {c: [] for c in CONDS}
    rows = []
    t0 = time.time()
    for i, s in enumerate(samples, 1):
        rec = {"id": s["id"], "q": s["question"], "gold": s["answers"], "preds": {}, "ok": {}}
        for cond in CONDS:
            text = serialize(s["ot"], cond, max_rows=max_rows)
            prompt = build_prompt(getattr(s["ot"], "title", "") or "", s["question"], text)
            try:
                raw = complete_retry(llm, ANSWER_SYS, prompt, max_tokens)
                pred = clean_pred(raw)
            except Exception as e:  # noqa: BLE001
                pred = ""
                rec.setdefault("err", {})[cond] = type(e).__name__
            ok = numeric_match(pred, s["answers"]) or exact_match(pred, s["answers"])
            rec["preds"][cond] = pred
            rec["ok"][cond] = bool(ok)
            per[cond].append(int(bool(ok)))
            time.sleep(sleep)
        rows.append(rec)
        if i % 10 == 0 or i == len(samples):
            acc = {c: np.mean(per[c]) for c in CONDS}
            print(f"  {i}/{len(samples)} " +
                  " ".join(f"{c}={acc[c]:.2f}" for c in CONDS) +
                  f"  {time.time()-t0:.0f}s", flush=True)
    return per, rows


def boot_ci(diff, iters=10000, seed=SEED):
    rng = np.random.default_rng(seed)
    arr = np.array(diff, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    s = arr[rng.integers(0, len(arr), (iters, len(arr)))].mean(1)
    return float(arr.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--splits", default="flat,hier")
    ap.add_argument("--llm", default="groq:llama-3.3-70b-versatile")
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--max-rows", type=int, default=60, help="chunk budget (rows serialized)")
    ap.add_argument("--sleep", type=float, default=0.35)
    ap.add_argument("--out", default="results/codegen/chunk_struct.json")
    args = ap.parse_args()

    llm = build_llm(args.llm)
    builders = {"flat": build_flat, "hier": build_hier}
    out = {"config": vars(args), "splits": {}}
    for name in args.splits.split(","):
        name = name.strip()
        samples = builders[name](args.n)
        per, rows = run_split(name, samples, llm, args.max_tokens, args.sleep, args.max_rows)
        acc = {c: round(float(np.mean(per[c])), 4) for c in CONDS}
        contrasts = {}
        for a, b in [("flat_values", "flat_leaf"), ("flat_leaf", "header_path"),
                     ("flat_values", "header_path")]:
            d = [y - x for x, y in zip(per[a], per[b])]
            m, lo, hi = boot_ci(d)
            contrasts[f"{b}-{a}"] = {"delta": round(m, 4), "ci": [round(lo, 4), round(hi, 4)],
                                     "sig": bool(lo > 0 or hi < 0)}
        out["splits"][name] = {"n": len(samples), "acc": acc, "contrasts": contrasts, "rows": rows}
        print(f"  [{name}] acc={acc}")
        for k, v in contrasts.items():
            print(f"    {k}: {v['delta']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]"
                  f"{'*' if v['sig'] else ''}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
