#!/usr/bin/env python3
"""CELL_RETRIEVAL GATE-3 + GATE-4 + GATE-5. 표내 셀검색 recall@k. 숫자만.

표마다 그 표의 셀들로만 인덱스 구성(표 내 검색). 질문 임베딩 → 셀 코사인 → top-k.
recall@k = top-k에 정답셀 1개라도 포함된 질문 비율. 4칸 × k=1,3,5.
paired bootstrap CI(질문단위, 10000, seed=42) + 대조 3종.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL = "BAAI/bge-small-en-v1.5"
QINSTR = "Represent this sentence for searching relevant passages: "
KS = (1, 3, 5)
SEED = 42
RES = Path("results")


def eval_condition(model, samples, cond):
    """각 표 내 셀검색. 반환: {k: [hit per question]} (0/1 벡터)."""
    # 셀 텍스트 평탄화 + 표 경계
    all_cells, bounds = [], []
    for s in samples:
        st = len(all_cells)
        all_cells.extend(c[cond] for c in s["cells"])
        bounds.append((st, len(all_cells)))
    cell_emb = model.encode(all_cells, batch_size=512, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False)
    q_emb = model.encode([QINSTR + s["question"] for s in samples], batch_size=256,
                         normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cell_t = torch.tensor(cell_emb, device=dev)
    hits = {k: [] for k in KS}
    rec_rows = []
    for si, s in enumerate(samples):
        st, en = bounds[si]
        if en == st:
            for k in KS:
                hits[k].append(0)
            continue
        sims = torch.tensor(q_emb[si], device=dev) @ cell_t[st:en].T
        order = torch.topk(sims, min(max(KS), en - st)).indices.cpu().numpy()
        ranked_coords = [tuple(s["cells"][j]["coord"]) for j in order]
        gold = {tuple(g) for g in s["gold_cells"]}
        topk_hit = {}
        for k in KS:
            hit = int(any(tuple(c) in gold for c in ranked_coords[:k]))
            hits[k].append(hit); topk_hit[k] = hit
        rec_rows.append({"id": s["id"], "condition": cond,
                         "gold_cells": s["gold_cells"],
                         "topk_cells": [list(c) for c in ranked_coords[:max(KS)]],
                         "hit@1": topk_hit[1], "hit@3": topk_hit[3], "hit@5": topk_hit[5],
                         "n_gold": len(gold)})
    return hits, rec_rows


def boot_ci(vec, B=10000):
    a = np.array(vec); rng = np.random.default_rng(SEED); n = len(a)
    bs = np.array([a[rng.integers(0, n, n)].mean() for _ in range(B)])
    return float(a.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def paired(va, vb, B=10000):
    d = np.array(vb) - np.array(va); rng = np.random.default_rng(SEED); n = len(d)
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"delta": float(d.mean()), "ci95": [float(lo), float(hi)], "sig": bool(lo > 0 or hi < 0)}


def did(dh_a, dh_b, df_a, df_b, B=10000):  # (hierB-hierA)-(flatB-flatA)
    dh = np.array(dh_b) - np.array(dh_a); df = np.array(df_b) - np.array(df_a)
    rng = np.random.default_rng(SEED)
    bs = np.array([dh[rng.integers(0, len(dh), len(dh))].mean()
                   - df[rng.integers(0, len(df), len(df))].mean() for _ in range(B)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"delta": float(dh.mean() - df.mean()), "ci95": [float(lo), float(hi)],
            "sig": bool(lo > 0 or hi < 0)}


def main():
    hier = json.loads(open(RES / "cell_sample_hier.json").read())
    flat = json.loads(open(RES / "cell_sample_flat.json").read())
    model = SentenceTransformer(MODEL, device="cuda" if torch.cuda.is_available() else "cpu")

    data = {"hier": hier, "flat": flat}
    H, raw_all = {}, []
    for comp in ("flat", "hier"):
        for cond in ("A", "B"):
            h, rows = eval_condition(model, data[comp], cond)
            H[f"{comp}-{cond}"] = h
            for r in rows:
                r["complexity"] = comp
            raw_all.extend(rows)

    with open(RES / "cell_recall_raw.jsonl", "w", encoding="utf-8") as f:
        for r in raw_all:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- GATE-3: 12 recall ----
    print("=" * 64); print("GATE-3  표내 셀검색 recall@k  (셀=정답셀 hit 비율)")
    print(f"{'cell':10s}" + "".join(f"  R@{k:<5d}" for k in KS))
    for key in ("flat-A", "flat-B", "hier-A", "hier-B"):
        print(f"{key:10s}" + "".join(f"  {np.mean(H[key][k]):.3f} " for k in KS))

    # ---- GATE-4: 대조 + CI ----
    stats = {"model": MODEL, "seed": SEED, "n_hier": len(hier), "n_flat": len(flat),
             "cells": {}, "contrasts": {}}
    for key in H:
        stats["cells"][key] = {k: {"recall": (m := boot_ci(H[key][k]))[0], "ci95": m[1]} for k in KS}
    print("\n" + "=" * 64); print("GATE-4  대조 (점추정 + 95% CI)")
    for k in KS:
        c1 = paired(H["hier-A"][k], H["hier-B"][k])
        c2 = paired(H["flat-A"][k], H["flat-B"][k])
        c3 = did(H["hier-A"][k], H["hier-B"][k], H["flat-A"][k], H["flat-B"][k])
        stats["contrasts"][f"k{k}"] = {"hier_B_minus_A": c1, "flat_B_minus_A": c2, "diff_in_diff": c3}
        print(f"  k={k}:")
        print(f"    hier B-A : {c1['delta']:+.3f} CI{[round(x,3) for x in c1['ci95']]} sig={c1['sig']}")
        print(f"    flat B-A : {c2['delta']:+.3f} CI{[round(x,3) for x in c2['ci95']]} sig={c2['sig']}")
        print(f"    diff-in-diff: {c3['delta']:+.3f} CI{[round(x,3) for x in c3['ci95']]} sig={c3['sig']}")
    json.dump(stats, open(RES / "cell_recall_stats.json", "w"), indent=2)

    # ---- GATE-5: 케이스 덤프 (hier, k=5 기준) ----
    byid = {}
    for r in raw_all:
        if r["complexity"] == "hier":
            byid.setdefault(r["id"], {})[r["condition"]] = r
    revived, still_wrong = [], []
    for hid, cc in byid.items():
        a, b = cc.get("A"), cc.get("B")
        if not a or not b:
            continue
        if a["hit@5"] == 0 and b["hit@5"] == 1:
            revived.append((hid, a, b))
        elif b["hit@5"] == 0:
            still_wrong.append((hid, a, b))
    qmap = {s["id"]: s for s in hier}
    md = ["# 셀검색 케이스 (연구자 검토용) — 자동분류 금지\n",
          f"hier, recall@5 기준. 살아난(A틀림→B맞음)={len(revived)} 여전히틀림(B@5=0)={len(still_wrong)}\n"]
    md.append("\n## A에선 틀렸는데 B에서 살아난 케이스\n")
    for hid, a, b in revived:
        s = qmap[hid]
        md.append(f"### {hid}\n질문: {s['question']}\n정답셀: {a['gold_cells']}\n"
                  f"A top5: {a['topk_cells']}\nB top5: {b['topk_cells']}\n[분류: ___ ]\n")
    md.append("\n## B에서도 여전히 틀린 케이스\n")
    for hid, a, b in still_wrong:
        s = qmap[hid]
        md.append(f"### {hid}\n질문: {s['question']}\n정답셀: {a['gold_cells']}\n"
                  f"A top5: {a['topk_cells']}\nB top5: {b['topk_cells']}\n[분류: ___ ]\n")
    (RES / "cell_cases_for_review.md").write_text("\n".join(md), encoding="utf-8")
    print("\n" + "=" * 64); print("GATE-5  케이스")
    print(f"  살아난(A@5틀림→B@5맞음): {len(revived)} | B@5에서도 틀림: {len(still_wrong)}")
    print(f"  -> results/cell_cases_for_review.md")


if __name__ == "__main__":
    main()
