#!/usr/bin/env python3
"""In-table cell retrieval: does header-path preprocessing help locate the
answer cell, and does the effect grow with table complexity (flat -> hier)?

Diagnosis-first design (no framing, numbers only):

  complexity in {flat (WikiTableQuestions), hier (HiTab)}
  condition  in {A = raw cell value, B = value + full header path}
  metric     = cell recall@k (k=1,3,5), within-table search (one tiny index
               per table, BGE-small cosine)

The 2x2 plus the difference-in-differences (hierB-hierA) - (flatB-flatA)
isolates "what header-path expansion buys" from raw complexity difficulty.

Ground-truth answer cell (both datasets, kept identical on purpose so the
complexity axis is apples-to-apples): the data cell whose normalized value
*completely matches* a normalized answer string. Questions with 0 or >2
matching cells are excluded (ambiguous); the exclusion count is reported.
This restricts the study to single-cell *lookup* questions and sidesteps
HiTab's brittle full-grid coordinate system (verified: some aggregate rows
are absent from the left-header tree, so coord->cell mapping is unsafe).

seed=42 everywhere. Outputs under --out-dir:
  cell_sample_{flat,hier}.json   cell_recall_raw.jsonl
  cell_recall_stats.json         cell_cases_for_review.md
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_agent.stores.original_store import _parse_paths  # noqa: E402

SEED = 42
_WS = re.compile(r"\s+")
_INTLIKE = re.compile(r"-?\d+\.0+$")


def norm(x) -> str:
    s = str(x).strip().lower().replace(",", "")
    if _INTLIKE.fullmatch(s):
        s = s.split(".")[0]
    return _WS.sub(" ", s).strip()


# --------------------------------------------------------------------------- #
# table abstraction: a grid of cells + per-cell (A, B) text                   #
# --------------------------------------------------------------------------- #

class CellTable:
    """A table reduced to data cells, each with raw value + header path."""

    def __init__(self, table_id, cells, a_texts, b_texts):
        self.table_id = table_id
        self.cells = cells          # list[(r, c, value_str)]
        self.a_texts = a_texts      # condition A text per cell
        self.b_texts = b_texts      # condition B text per cell

    def gold_indices(self, answers):
        ans = {norm(a) for a in answers if norm(a)}
        return [i for i, (_, _, v) in enumerate(self.cells) if norm(v) in ans]


def hitab_table(raw: dict) -> CellTable:
    data = raw.get("data") or []
    _, top_by_idx = _parse_paths(raw.get("top_root") or {})
    _, left_by_idx = _parse_paths(raw.get("left_root") or {})
    cells, a, b = [], [], []
    for r, row in enumerate(data):
        rpath = " ".join(left_by_idx.get(r, []))
        for c, cell in enumerate(row):
            v = cell.get("value") if isinstance(cell, dict) else cell
            v = "" if v is None else str(v)
            if not v.strip():
                continue
            cpath = " ".join(top_by_idx.get(c, []))
            cells.append((r, c, v))
            a.append(v)
            head = " > ".join(p for p in (rpath, cpath) if p)
            b.append(f"{head} | {v}" if head else v)
    return CellTable(raw.get("table_id", "?"), cells, a, b)


def wtq_table(table_id: str, header, rows) -> CellTable:
    cells, a, b = [], [], []
    for r, row in enumerate(rows):
        rowkey = str(row[0]) if row else ""
        for c, v in enumerate(row):
            v = "" if v is None else str(v)
            if not v.strip():
                continue
            colhdr = str(header[c]) if c < len(header) else ""
            cells.append((r, c, v))
            a.append(v)
            head = " > ".join(p for p in (rowkey, colhdr) if p and p != v)
            b.append(f"{head} | {v}" if head else v)
    return CellTable(table_id, cells, a, b)


# --------------------------------------------------------------------------- #
# dataset loading + seeded sampling (gold = complete-match data cell)         #
# --------------------------------------------------------------------------- #

def build_hier(data_dir: str, split: str, n: int):
    from rag_agent.data.loader import load_samples, load_table

    samples = load_samples(data_dir, split=split)
    rng = random.Random(SEED)
    rng.shuffle(samples)
    kept, excl_nogold, excl_multi, tcache = [], 0, 0, {}
    for s in samples:
        tid = s.get("table_id")
        ans = s.get("answer") or []
        if not tid or not ans:
            continue
        if tid not in tcache:
            raw = load_table(tid, data_dir)
            tcache[tid] = hitab_table(raw) if raw else None
        ct = tcache[tid]
        if ct is None:
            continue
        gold = ct.gold_indices(ans)
        if len(gold) == 0:
            excl_nogold += 1
            continue
        if len(gold) > 2:
            excl_multi += 1
            continue
        kept.append({"id": s["id"], "question": s["question"], "answer": ans,
                     "table_id": tid, "n_gold": len(gold)})
        if len(kept) >= n:
            break
    return kept, tcache, {"no_gold": excl_nogold, "multi_gold": excl_multi}


def build_flat(n: int):
    from datasets import load_dataset

    ds = None
    for name in ("stanfordnlp/wikitablequestions", "wikitablequestions"):
        try:
            ds = load_dataset(name, split="validation")
            break
        except Exception as e:  # noqa: BLE001
            print(f"  load_dataset({name}) failed: {e}", file=sys.stderr)
    if ds is None:
        raise SystemExit("could not load WikiTableQuestions")

    idx = list(range(len(ds)))
    random.Random(SEED).shuffle(idx)
    kept, excl_nogold, excl_multi, tcache = [], 0, 0, {}
    for i in idx:
        ex = ds[i]
        tbl = ex.get("table") or {}
        header, rows = tbl.get("header") or [], tbl.get("rows") or []
        ans = ex.get("answers") or ex.get("answer") or []
        tid = ex.get("id") or f"wtq-{i}"
        if not header or not rows or not ans:
            continue
        if tid not in tcache:
            tcache[tid] = wtq_table(tid, header, rows)
        gold = tcache[tid].gold_indices(ans)
        if len(gold) == 0:
            excl_nogold += 1
            continue
        if len(gold) > 2:
            excl_multi += 1
            continue
        kept.append({"id": tid, "question": ex["question"], "answer": list(ans),
                     "table_id": tid, "n_gold": len(gold)})
        if len(kept) >= n:
            break
    return kept, tcache, {"no_gold": excl_nogold, "multi_gold": excl_multi}


# --------------------------------------------------------------------------- #
# retrieval + metrics                                                         #
# --------------------------------------------------------------------------- #

def run_condition(model, samples, tcache, cond, ks):
    import numpy as np

    texts_attr = "a_texts" if cond == "A" else "b_texts"
    # encode all cells across the sampled tables once
    needed = {s["table_id"] for s in samples}
    offsets, flat_texts = {}, []
    for tid in needed:
        ct = tcache[tid]
        offsets[tid] = (len(flat_texts), len(ct.cells))
        flat_texts.extend(getattr(ct, texts_attr))
    cell_emb = model.encode(flat_texts, batch_size=256, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False)
    q_emb = model.encode([s["question"] for s in samples], batch_size=128,
                         normalize_embeddings=True, convert_to_numpy=True,
                         show_progress_bar=False)

    maxk = max(ks)
    per_case, hits = [], {k: [] for k in ks}
    for si, s in enumerate(samples):
        tid = s["table_id"]
        off, nlen = offsets[tid]
        ct = tcache[tid]
        scores = cell_emb[off:off + nlen] @ q_emb[si]
        order = list(np.argsort(-scores)[:maxk])
        gold = set(ct.gold_indices(s["answer"]))
        topk_cells = [list(ct.cells[j][:2]) for j in order]
        rec = {"id": s["id"], "condition": cond, "table_id": tid,
               "n_gold": len(gold), "gold_cells": [list(ct.cells[j][:2]) for j in gold],
               "topk_cells": topk_cells}
        for k in ks:
            hit = int(any(j in gold for j in order[:k]))
            hits[k].append(hit)
            rec[f"hit@{k}"] = hit
        per_case.append(rec)
    return hits, per_case


def paired_bootstrap(a, b, n_iters=10000, seed=SEED):
    """95% CI of mean(a)-mean(b), paired, resampling the index."""
    import random as _r
    assert len(a) == len(b)
    n = len(a)
    mean = (sum(a) - sum(b)) / n
    rng = _r.Random(seed)
    deltas = []
    for _ in range(n_iters):
        s = 0
        for _ in range(n):
            i = rng.randrange(n)
            s += a[i] - b[i]
        deltas.append(s / n)
    deltas.sort()
    return mean, deltas[int(0.025 * n_iters)], deltas[int(0.975 * n_iters)]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hitab-dir", default="/tmp/HiTab")
    p.add_argument("--hitab-split", default="dev")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--device", default="cpu")
    p.add_argument("--ks", default="1,3,5")
    p.add_argument("--boot-iters", type=int, default=10000)
    p.add_argument("--out-dir", default="results/cell")
    p.add_argument("--skip-flat", action="store_true",
                   help="hier only (e.g. when WTQ download is unavailable)")
    args = p.parse_args()

    ks = [int(k) for k in args.ks.split(",")]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- GATE-1: sampling + exclusion report --------------------------------
    print("=== GATE-1: data load + sampling ===")
    hier_s, hier_tc, hier_excl = build_hier(args.hitab_dir, args.hitab_split, args.n)
    json.dump(hier_s, open(out / "cell_sample_hier.json", "w"), indent=1)
    print(f"hier (HiTab {args.hitab_split}): kept {len(hier_s)} | "
          f"excluded no_gold={hier_excl['no_gold']} multi_gold={hier_excl['multi_gold']}")
    if hier_s:
        ex = hier_tc[hier_s[0]["table_id"]]
        print(f"  hier table[0] {ex.table_id}: {len(ex.cells)} cells; "
              f"example B = {ex.b_texts[0]!r}")

    flat_s, flat_tc = [], {}
    flat_excl = {"no_gold": 0, "multi_gold": 0}
    if not args.skip_flat:
        flat_s, flat_tc, flat_excl = build_flat(args.n)
        json.dump(flat_s, open(out / "cell_sample_flat.json", "w"), indent=1)
        tot = len(flat_s) + flat_excl["no_gold"] + flat_excl["multi_gold"]
        rate = (flat_excl["no_gold"] + flat_excl["multi_gold"]) / max(tot, 1)
        print(f"flat (WTQ valid): kept {len(flat_s)} | "
              f"excluded no_gold={flat_excl['no_gold']} multi_gold={flat_excl['multi_gold']} "
              f"(exclusion rate {rate:.3f})")

    # ---- embedding model ----------------------------------------------------
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model, device=args.device)

    # ---- GATE-3: 2x2 x k recall --------------------------------------------
    print("\n=== GATE-3: recall@k (4 cells x k) ===")
    raw_fh = open(out / "cell_recall_raw.jsonl", "w")
    cells_data, point = {}, {}
    blocks = [("hier", hier_s, hier_tc)]
    if flat_s:
        blocks.append(("flat", flat_s, flat_tc))
    for comp, samples, tc in blocks:
        for cond in ("A", "B"):
            hits, per_case = run_condition(model, samples, tc, cond, ks)
            cells_data[(comp, cond)] = hits
            for rec in per_case:
                rec["complexity"] = comp
                raw_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            row = {f"R@{k}": round(sum(hits[k]) / len(hits[k]), 4) for k in ks}
            point[(comp, cond)] = row
            print(f"  {comp}-{cond}: " + " ".join(f"R@{k}={row[f'R@{k}']}" for k in ks)
                  + f"  (n={len(samples)})")
    raw_fh.close()

    # ---- GATE-4: paired bootstrap contrasts --------------------------------
    print("\n=== GATE-4: paired bootstrap CI (10k, seed=42) ===")
    stats = {"point_estimates": {f"{c}-{cd}": point[(c, cd)] for (c, cd) in point},
             "exclusions": {"hier": hier_excl, "flat": flat_excl},
             "n": {"hier": len(hier_s), "flat": len(flat_s)},
             "contrasts": {}}

    def contrast(name, a_key, b_key):
        stats["contrasts"][name] = {}
        for k in ks:
            mean, lo, hi = paired_bootstrap(cells_data[a_key][k], cells_data[b_key][k],
                                            n_iters=args.boot_iters)
            sig = lo > 0 or hi < 0
            stats["contrasts"][name][f"R@{k}"] = {
                "delta": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)],
                "significant": sig}
            print(f"  {name} R@{k}: {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                  + (" *" if sig else ""))

    contrast("hierB - hierA", ("hier", "B"), ("hier", "A"))
    if flat_s:
        contrast("flatB - flatA", ("flat", "B"), ("flat", "A"))
        # difference-in-differences: paired by index requires equal n; report as
        # difference of the two deltas with a bootstrap over the pooled question
        # sets (independent samples, so resample each separately).
        print("  --- difference-in-differences ---")
        stats["contrasts"]["DiD (hierB-hierA)-(flatB-flatA)"] = {}
        import random as _r
        for k in ks:
            hb, ha = cells_data[("hier", "B")][k], cells_data[("hier", "A")][k]
            fb, fa = cells_data[("flat", "B")][k], cells_data[("flat", "A")][k]
            did = (sum(hb) - sum(ha)) / len(hb) - (sum(fb) - sum(fa)) / len(fb)
            rng = _r.Random(SEED)
            ds = []
            for _ in range(args.boot_iters):
                nh, nf = len(hb), len(fb)
                sh = sum(hb[(i := rng.randrange(nh))] - ha[i] for _ in range(nh)) / nh
                sf = sum(fb[(j := rng.randrange(nf))] - fa[j] for _ in range(nf)) / nf
                ds.append(sh - sf)
            ds.sort()
            lo, hi = ds[int(0.025 * args.boot_iters)], ds[int(0.975 * args.boot_iters)]
            sig = lo > 0 or hi < 0
            stats["contrasts"]["DiD (hierB-hierA)-(flatB-flatA)"][f"R@{k}"] = {
                "delta": round(did, 4), "ci95": [round(lo, 4), round(hi, 4)],
                "significant": sig}
            print(f"  DiD R@{k}: {did:+.4f} [{lo:+.4f}, {hi:+.4f}]" + (" *" if sig else ""))

    json.dump(stats, open(out / "cell_recall_stats.json", "w"), indent=1)

    # ---- GATE-5: case dump (hier, k=5) -------------------------------------
    cases = out / "cell_cases_for_review.md"
    by_id = {}
    for rec in (json.loads(l) for l in open(out / "cell_recall_raw.jsonl")):
        if rec["complexity"] != "hier":
            continue
        by_id.setdefault(rec["id"], {})[rec["condition"]] = rec
    saved, still = 0, 0
    with open(cases, "w") as f:
        f.write("# hier cell-retrieval cases (k=5, researcher fills the blanks)\n\n")
        qmap = {s["id"]: s for s in hier_s}
        f.write("## Rescued by header path (A miss -> B hit @5)\n\n")
        for qid, d in by_id.items():
            if "A" in d and "B" in d and not d["A"]["hit@5"] and d["B"]["hit@5"]:
                saved += 1
                _dump_case(f, qmap[qid], d)
        f.write("\n## Still wrong under B (@5)\n\n")
        for qid, d in by_id.items():
            if "B" in d and not d["B"]["hit@5"]:
                still += 1
                _dump_case(f, qmap[qid], d)
    print("\n=== GATE-5: case dump ===")
    print(f"  rescued (A miss -> B hit @5): {saved} | still wrong under B @5: {still}")
    print(f"  wrote {cases}")
    print(f"\nartifacts in {out}/")


def _dump_case(f, s, d):
    f.write(f"- **Q** ({s['id']}): {s['question']}\n")
    f.write(f"  - answer: {s['answer']}  gold_cells: {d.get('B', d.get('A'))['gold_cells']}\n")
    if "A" in d:
        f.write(f"  - A top5: {d['A']['topk_cells']}\n")
    if "B" in d:
        f.write(f"  - B top5: {d['B']['topk_cells']}\n")
    f.write("  - (classification: ____)\n\n")


if __name__ == "__main__":
    main()
