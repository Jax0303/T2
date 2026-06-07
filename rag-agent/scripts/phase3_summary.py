#!/usr/bin/env python3
"""Phase 3 합성·통계 — 라우팅 ablation 결과 → routing.csv + 층화 + Gate 1·3.

입력 (codegen_eval 산출, hard-class dev, 540-pool):
  results/phase3_FULL.json        adaptive (라우터 ON)        = FULL
  results/phase3_A1_vector.json   always-codegen (VDB→codegen) = A1 always-vector
  results/phase3_A2_structured.json always-original           = A2 always-structured
  results/phase3_A3_nostruct.json always-keyword (구조/verifier 신호 OFF) = A3 ref

A4 oracle-router = per-query max(A1, A2)  (정답 경로를 아는 상한)
통계: paired bootstrap 95% CI + 2-sided p (1000 resample, seed=42)
       (Smucker et al. 2007 권고: bootstrap; Wilcoxon/sign 지양)
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
rng = np.random.default_rng(SEED)

import os
_SUF = os.environ.get("PHASE3_SUFFIX", "_groq")  # "_groq" (재실행) | "" (로컬)
FILES = {
    "FULL": f"phase3_FULL{_SUF}.json",
    "A1_vector": f"phase3_A1_vector{_SUF}.json",
    "A2_structured": f"phase3_A2_structured{_SUF}.json",
    "A3_nostruct": f"phase3_A3_nostruct{_SUF}.json",
}


def load_rows(fn):
    d = json.loads((RES / fn).read_text())
    by_q = {}
    for r in d["rows"]:
        by_q[(r["class"], r["query"])] = int(bool(r["correct"]))
    return by_q


def paired_bootstrap(a, b, B=1000):
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = len(a); base = float(b.mean() - a.mean())
    idx = rng.integers(0, n, size=(B, n))
    diffs = np.sort((b[idx] - a[idx]).mean(axis=1))
    lo, hi = float(diffs[int(0.025 * B)]), float(diffs[int(0.975 * B)])
    # 2-sided p: proportion of resampled diffs on the opposite side of 0 ×2
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"delta": base, "ci": [lo, hi], "p": float(min(p, 1.0)), "sig": bool(lo > 0 or hi < 0)}


def main():
    data = {k: load_rows(v) for k, v in FILES.items() if (RES / v).exists()}
    if "A1_vector" not in data or "A2_structured" not in data:
        print("need A1 and A2 at minimum"); return
    # 공통 질의 집합 (모든 ablation 교집합)
    keys = set.intersection(*[set(d) for d in data.values()])
    keys = sorted(keys)
    n = len(keys)
    print(f"공통 질의 {n}개 (ablation 교집합)")

    # per-query correctness 벡터
    vec = {k: [d[q] for q in keys] for k, d in data.items()}
    # A4 oracle-router = max(A1, A2)
    vec["A4_oracle_router"] = [max(a, b) for a, b in zip(vec["A1_vector"], vec["A2_structured"])]

    overall = {k: float(np.mean(v)) for k, v in vec.items()}

    # 층화
    classes = [k[0] for k in keys]
    cls_set = sorted(set(classes))
    by_class = {}
    for c in cls_set:
        idx = [i for i, cc in enumerate(classes) if cc == c]
        by_class[c] = {"n": len(idx),
                       **{k: round(float(np.mean([vec[k][i] for i in idx])), 4) for k in vec}}

    # paired stats
    pairs = [("FULL", "A1_vector"), ("A4_oracle_router", "A1_vector"),
             ("A4_oracle_router", "FULL"), ("A2_structured", "A1_vector"),
             ("A2_structured", "A3_nostruct")]
    stats = {}
    for a, b in pairs:
        if a in vec and b in vec:
            stats[f"{b}_vs_{a}"] = paired_bootstrap(vec[a], vec[b])  # delta = b - a

    # ---- CSV ----
    with open(RES / "phase3_routing.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["ablation", "n", "NM"])
        for k in ["FULL", "A1_vector", "A2_structured", "A3_nostruct", "A4_oracle_router"]:
            if k in overall:
                w.writerow([k, n, round(overall[k], 4)])

    # ---- markdown ----
    L = ["# Phase 3 — 라우팅 ablation (codegen 경로, hard-class dev, 540-pool)", ""]
    L.append(f"공통 질의 n={n} · 정답기준 NM(±2% 허용) · seed={SEED}")
    L.append("> 주: Phase 2(3597-pool, 직접답변)와 별개 regime. codegen 경로는 hard-class 한정.")
    L.append("")
    L.append("| ablation | NM |")
    L.append("|---|---|")
    names = {"FULL": "FULL (adaptive 라우터)", "A1_vector": "A1 always-vector",
             "A2_structured": "A2 always-structured", "A3_nostruct": "A3 구조/verifier OFF",
             "A4_oracle_router": "A4 oracle-router (상한)"}
    for k in ["A4_oracle_router", "FULL", "A2_structured", "A1_vector", "A3_nostruct"]:
        if k in overall:
            L.append(f"| {names[k]} | {overall[k]:.4f} |")
    L.append("")
    L.append("## paired bootstrap (Δ=후자−전자, 95% CI, 2-sided p; 1000 resample)")
    L.append("| 비교 (b vs a) | Δ(b−a) | CI95 | p | sig |")
    L.append("|---|---|---|---|---|")
    for name, s in stats.items():
        L.append(f"| {name} | {s['delta']:+.4f} | [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}] | {s['p']:.3f} | {s['sig']} |")
    L.append("")
    L.append("## 난이도 층화 (NM, 셀별 n)")
    hdr_keys = [k for k in ["A1_vector", "A2_structured", "FULL", "A4_oracle_router", "A3_nostruct"] if k in vec]
    L.append("| class | n | " + " | ".join(hdr_keys) + " |")
    L.append("|---|---|" + "|".join(["---"] * len(hdr_keys)) + "|")
    for c in cls_set:
        row = by_class[c]
        L.append(f"| {c} | {row['n']} | " + " | ".join(f"{row[k]:.3f}" for k in hdr_keys) + " |")
    L.append("")

    # ---- Gate / Decision ----
    g3 = overall.get("A4_oracle_router", 0) >= overall.get("FULL", 0)
    L.append("## Gate 3 / Decision Gate 1")
    L.append(f"- Gate 3 (A4 oracle-router ≥ FULL): **{g3}** "
             f"(A4={overall.get('A4_oracle_router',0):.4f}, FULL={overall.get('FULL',0):.4f})")
    s_dg1 = stats.get("A1_vector_vs_A4_oracle_router")  # delta = A1 - A4 (음수면 A4 우위)
    # Decision Gate 1: oracle-router vs always-vector(A1) +2~3%p 미만 & p≥0.05 → 라우팅 기여 철회
    a4 = overall.get("A4_oracle_router", 0); a1 = overall.get("A1_vector", 0)
    gain = a4 - a1
    s = stats.get("A4_oracle_router_vs_A1_vector") or {}
    verdict = ("라우팅 기여 철회 (gain<+0.02~0.03 & p≥0.05)"
               if (gain < 0.03 and s.get("p", 1) >= 0.05) else
               "라우팅 기여 유지 (oracle-router가 always-vector 유의 상회)")
    L.append(f"- Decision Gate 1: A4−A1 = {gain:+.4f}, p={s.get('p',float('nan')):.3f} → **{verdict}**")

    (RES / "phase3_summary.md").write_text("\n".join(L))
    (RES / "phase3_routing.json").write_text(json.dumps(
        {"n": n, "overall": overall, "by_class": by_class, "paired": stats}, ensure_ascii=False, indent=2))
    print("wrote phase3_routing.csv / phase3_summary.md / phase3_routing.json")
    print("\n".join(L))


if __name__ == "__main__":
    main()
