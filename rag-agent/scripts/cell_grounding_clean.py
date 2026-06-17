#!/usr/bin/env python3
"""Clean LLM-free cell-grounding: does header-path expansion help *locate* the
answer cell, and does the effect grow with table complexity?  (Milestone:
fixes the flat confound of cell_retrieval_eval by using WikiSQL, whose SQL gives
the EXACT gold cell — sel column × WHERE-matched row — instead of fragile
value-matching.)

  complexity in {flat (WikiSQL, exact gold), hier (HiTab, formula gold)}
  condition  in {A = raw cell value, B = value + header path}
  metric     = cell recall@k within one tiny per-table index (BGE-small cosine)

No generative LLM is used anywhere — purely retrieval/representation quality.
seed=42. Outputs results/cell/clean_grounding.json
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from cell_retrieval_eval import CellTable, hitab_table, norm  # noqa: E402
from rag_agent.data.loader import load_samples, load_table     # noqa: E402

SEED = 42


def _isnum(v):
    try: float(str(v).replace(",", "")); return True
    except (ValueError, TypeError): return False
def _num(v): return float(str(v).replace(",", ""))


def wikisql_celltable(header, rows):
    cells, a, b, idx_of = [], [], [], {}
    for r, row in enumerate(rows):
        rowkey = str(row[0]) if row else ""
        for c, v in enumerate(row):
            v = "" if v is None else str(v)
            if not v.strip():
                continue
            colhdr = str(header[c]) if c < len(header) else ""
            idx_of[(r, c)] = len(cells)
            cells.append((r, c, v)); a.append(v)
            head = " > ".join(p for p in (rowkey, colhdr) if p and p != v)
            b.append(f"{head} | {v}" if head else v)
    ct = CellTable("wikisql", cells, a, b); ct.idx_of = idx_of
    return ct


def build_flat_wikisql(n, wikisql_dir="/tmp/WikiSQL/data", split="dev"):
    base = Path(wikisql_dir)
    tables = {json.loads(l)["id"]: json.loads(l) for l in open(base / f"{split}.tables.jsonl")}
    exs = [json.loads(l) for l in open(base / f"{split}.jsonl")]
    random.Random(SEED).shuffle(exs)
    tcache, samples, excl = {}, [], {"agg": 0, "no_match": 0, "multi": 0}
    for e in exs:
        sql = e["sql"]
        if sql["agg"] != 0:          # single-cell lookups only (apples-to-apples with hier)
            excl["agg"] += 1; continue
        t = tables.get(e["table_id"])
        if not t or not t.get("rows"):
            continue
        def match(row):
            for ci, op, val in sql["conds"]:
                cv = str(row[ci]).strip().lower(); vv = str(val).strip().lower()
                if op == 0 and cv != vv: return False
                if op == 1 and not (_isnum(row[ci]) and _isnum(val) and _num(row[ci]) > _num(val)): return False
                if op == 2 and not (_isnum(row[ci]) and _isnum(val) and _num(row[ci]) < _num(val)): return False
            return True
        mrows = [r for r, row in enumerate(t["rows"]) if match(row)]
        if not mrows: excl["no_match"] += 1; continue
        if len(mrows) > 2: excl["multi"] += 1; continue
        tid = e["table_id"]
        if tid not in tcache:
            tcache[tid] = wikisql_celltable(t["header"], t["rows"])
        ct = tcache[tid]
        gold = [ct.idx_of[(r, sql["sel"])] for r in mrows if (r, sql["sel"]) in ct.idx_of]
        if not gold:
            continue
        samples.append({"id": f"{tid}_{len(samples)}", "question": e["question"],
                        "table_id": tid, "gold_idx": gold})
        if len(samples) >= n:
            break
    return samples, tcache, excl


def build_hier(n, data_dir="/tmp/HiTab", split="dev"):
    samples_raw = load_samples(data_dir, split=split)
    random.Random(SEED).shuffle(samples_raw)
    tcache, samples, excl = {}, [], {"no_gold": 0, "multi": 0}
    for s in samples_raw:
        tid, ans = s.get("table_id"), s.get("answer") or []
        if not tid or not ans:
            continue
        if tid not in tcache:
            raw = load_table(tid, data_dir)
            tcache[tid] = hitab_table(raw) if raw else None
        ct = tcache[tid]
        if ct is None:
            continue
        gold = ct.gold_indices(ans)
        if not gold: excl["no_gold"] += 1; continue
        if len(gold) > 2: excl["multi"] += 1; continue
        samples.append({"id": s["id"], "question": s["question"], "table_id": tid, "gold_idx": gold})
        if len(samples) >= n:
            break
    return samples, tcache, excl


def run_condition(model, samples, tcache, cond, ks):
    attr = "a_texts" if cond == "A" else "b_texts"
    needed = {s["table_id"] for s in samples}
    offsets, flat = {}, []
    for tid in needed:
        ct = tcache[tid]; offsets[tid] = (len(flat), len(ct.cells)); flat.extend(getattr(ct, attr))
    cemb = model.encode(flat, batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    qemb = model.encode([s["question"] for s in samples], batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    hits = {k: [] for k in ks}
    for si, s in enumerate(samples):
        off, nlen = offsets[s["table_id"]]
        scores = cemb[off:off + nlen] @ qemb[si]
        order = list(np.argsort(-scores)[:max(ks)])
        gold = set(s["gold_idx"])
        for k in ks:
            hits[k].append(int(any(j in gold for j in order[:k])))
    return hits


def boot_ci(diff, iters=10000, seed=SEED):
    rng = np.random.default_rng(seed); n = len(diff); arr = np.array(diff)
    s = arr[rng.integers(0, n, (iters, n))].mean(1)
    return float(arr.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--ks", default="1,3,5")
    ap.add_argument("--out", default="results/cell/clean_grounding.json")
    args = ap.parse_args()
    ks = [int(k) for k in args.ks.split(",")]
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model, device="cpu")

    print("=== load (exact gold) ===", flush=True)
    flat_s, flat_tc, flat_excl = build_flat_wikisql(args.n)
    hier_s, hier_tc, hier_excl = build_hier(args.n)
    print(f"flat (WikiSQL): {len(flat_s)} | excl {flat_excl}")
    print(f"hier (HiTab):   {len(hier_s)} | excl {hier_excl}")

    res = {"n": {"flat": len(flat_s), "hier": len(hier_s)}, "ks": ks, "point": {}, "contrasts": {}}
    cells = {}
    for name, s, tc in [("flat", flat_s, flat_tc), ("hier", hier_s, hier_tc)]:
        for cond in ["A", "B"]:
            h = run_condition(model, s, tc, cond, ks)
            cells[(name, cond)] = h
            res["point"][f"{name}-{cond}"] = {f"R@{k}": round(float(np.mean(h[k])), 4) for k in ks}
            print(f"  {name}-{cond}: " + " ".join(f"R@{k}={np.mean(h[k]):.3f}" for k in ks), flush=True)

    print("\n=== contrasts (B-A) + difference-in-differences ===", flush=True)
    for name in ["flat", "hier"]:
        for k in ks:
            d = [b - a for a, b in zip(cells[(name, "A")][k], cells[(name, "B")][k])]
            m, lo, hi = boot_ci(d)
            sig = lo > 0 or hi < 0
            res["contrasts"][f"{name} B-A R@{k}"] = {"delta": round(m, 4), "ci": [round(lo, 4), round(hi, 4)], "sig": sig}
            print(f"  {name} B-A R@{k}: {m:+.3f} [{lo:+.3f},{hi:+.3f}]{'*' if sig else ''}", flush=True)
    # DiD needs equal n; report on min length aligned by index is not paired across datasets, so report gap of deltas
    res["did_note"] = "flat and hier are different query sets; compare B-A magnitudes, not a paired DiD"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
