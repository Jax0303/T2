#!/usr/bin/env python3
"""GATE: how accurately does the deterministic HPIR decompose a question into the
header paths it actually refers to?  This is the single point of failure of the
operand-complete-retrieval pipeline — if HPIR picks the wrong header paths, the
wrong operand cells are retrieved 'completely' and the answer is confidently wrong.

Two independent reviews said: measure this FIRST. <60% -> the pivot is dead;
>80% -> proceed. No LLM, no rate limits.

Gold (no coordinate mapping needed): HiTab `linked_cells.entity_link` keys are the
real header strings the question links to — `top` = column headers, `left` = row
headers (excluding the '[ANSWER]' marker). HPIR predicts col_paths / row_paths.
Metric: header-path recall — fraction of gold headers covered by any predicted path
of that axis (token-Jaccard >= thr or substring). seed=42.
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np                                                    # noqa: E402
from rag_agent.data.loader import load_samples, load_table           # noqa: E402
from rag_agent.stores.original_store import build_original_table     # noqa: E402
from rag_agent.query.header_path_resolver import resolve_against_table  # noqa: E402

SEED = 42
_W = re.compile(r"[a-z0-9]+")
_STOP = {"the","of","a","an","in","on","to","and","or","is","was","were","for","by",
         "with","that","this","which","they","their","at","as","from","be"}


def toks(s):
    return {w for w in _W.findall(str(s).lower()) if w not in _STOP and len(w) > 1}


def covered(gold, pred_paths, thr=0.5):
    g = toks(gold)
    if not g:
        return True
    for p in pred_paths:
        pt = toks(" ".join(p))
        if not pt:
            continue
        inter = len(g & pt)
        if inter / len(g) >= thr or inter >= 2:   # >=50% of gold tokens, or >=2 shared
            return True
    return False


def gold_headers(linked):
    el = (linked or {}).get("entity_link", {}) if isinstance(linked, dict) else {}
    cols = [k for k in (el.get("top") or {}) if k != "[ANSWER]"]
    rows = [k for k in (el.get("left") or {}) if k != "[ANSWER]"]
    return rows, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--data", default="data/hitab")
    ap.add_argument("--top-n-rows", type=int, default=4)
    ap.add_argument("--top-n-cols", type=int, default=3)
    args = ap.parse_args()

    import random
    ss = load_samples(args.data, args.split)
    random.Random(SEED).shuffle(ss)
    tcache = {}
    row_hit, col_hit, both_hit = [], [], []
    n_used = 0
    for s in ss:
        tid = s.get("table_id")
        g_rows, g_cols = gold_headers(s.get("linked_cells"))
        if not tid or (not g_rows and not g_cols):
            continue
        if tid not in tcache:
            raw = load_table(tid, args.data)
            tcache[tid] = build_original_table(raw) if raw else None
        ot = tcache[tid]
        if ot is None:
            continue
        intent = resolve_against_table(s["question"], ot,
                                       top_n_cols=args.top_n_cols, top_n_rows=args.top_n_rows)
        rok = all(covered(g, intent.row_paths) for g in g_rows) if g_rows else None
        cok = all(covered(g, intent.col_paths) for g in g_cols) if g_cols else None
        if rok is not None:
            row_hit.append(int(rok))
        if cok is not None:
            col_hit.append(int(cok))
        # 'both' = every gold header on every present axis covered
        present = [x for x in (rok, cok) if x is not None]
        both_hit.append(int(all(present)))
        n_used += 1
        if n_used >= args.n:
            break

    print(f"HPIR decomposition gate | n={n_used} | top_n_rows={args.top_n_rows} top_n_cols={args.top_n_cols}")
    print(f"  ROW header recall : {np.mean(row_hit):.3f}  (n={len(row_hit)})")
    print(f"  COL header recall : {np.mean(col_hit):.3f}  (n={len(col_hit)})")
    print(f"  BOTH-axes correct : {np.mean(both_hit):.3f}  (n={len(both_hit)})  <-- the gate number")


if __name__ == "__main__":
    main()
