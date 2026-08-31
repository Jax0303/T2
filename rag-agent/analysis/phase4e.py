#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Phase 4e -- Holm correction, hitab_lookup subset stability, and what the
P1-only-correct rows had in their P4 context."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402
from phase4_summary import em                                        # noqa: E402
from scipy.stats import binomtest, norm as znorm                     # noqa: E402

POOLS = ["hitab_lookup", "hitab_arith", "aitqa", "rhb_fact", "rhb_num"]
ALPHA = 0.05


def bc(d):
    p1 = d[d.policy == "P1_fixed_512"].set_index("query_id").is_correct
    p4 = d[d.policy == "P4_path_cell"].set_index("query_id").is_correct.reindex(p1.index)
    return p1, p4, int(((p4 == 1) & (p1 == 0)).sum()), int(((p4 == 0) & (p1 == 1)).sum())


def main() -> int:
    df = pd.DataFrame([json.loads(l) for l in
                       open("results/phase4/reader_records.jsonl")])
    df["is_correct"] = [em(p, g) for p, g in zip(df.pred_parsed, df.gold_answer)]
    L = ["# Phase 4e", ""]

    # ---- Task A: Holm-Bonferroni over the 5 pool p-values ----
    ps = []
    for pool in POOLS:
        _, _, b, c = bc(df[df.pool == pool])
        ps.append((pool, b, c, binomtest(b, b + c, 0.5).pvalue if b + c else 1.0))
    order = sorted(range(5), key=lambda i: ps[i][3])
    m = len(ps)
    rej, prev = {}, True
    L += ["## Task A -- Holm-Bonferroni (m=5, alpha=0.05)", "",
          "| 순위 | pool | b:c | 원 p | 보정 임계값 alpha/(m-i+1) | 판정 |",
          "|---:|---|---|---:|---:|---|"]
    for k, i in enumerate(order):
        pool, b, c, p = ps[i]
        thr = ALPHA / (m - k)
        prev = prev and (p < thr)
        rej[pool] = prev
        L.append(f"| {k+1} | {pool} | {b}:{c} | {p:.4e} | {thr:.6f} | "
                 f"{'기각 (유의)' if prev else '기각 실패'} |")
    L += ["", "순차 절차이므로 순위 k에서 기각에 실패하면 이후 순위는 모두 기각 실패로 "
          "처리했다.", ""]

    # ---- Task B: hitab_lookup, 60 vs 129 ----
    base = {r["query_id"] for r in csv.DictReader(
        open("results/phase4/sample_294.csv")) if r["pool"] == "hitab_lookup"}
    ext = {r["query_id"] for r in csv.DictReader(
        open("results/phase4/sample_hitab_lookup_ext129.csv"))}
    rec = {("results/phase4/taskB.json", "기존 60"): base,
           ("results/phase4/taskB_retrieval_ext129.json", "추가 129"): ext}
    L += ["## Task B -- hitab_lookup 부분집합 안정성", "",
          "| 부분집합 | n | P1 EM | P4 EM | delta | b (P4만) | c (P1만) | "
          "Recall P1 | Recall P4 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    stat = {}
    for (f, name), ids in rec.items():
        d = df[(df.pool == "hitab_lookup") & (df.query_id.isin(ids))]
        p1, p4, b, c = bc(d)
        stat[name] = (p1.values, p4.values)
        r = {e["policy"]: e["recall"] for e in json.load(open(f))["b2"]
             if e["pool"] == "hitab_lookup"}
        L.append(f"| {name} | {len(p1)} | {p1.mean():.4f} | {p4.mean():.4f} | "
                 f"{p4.mean()-p1.mean():+.4f} | {b} | {c} | "
                 f"{r['P1_fixed_512']:.4f} | {r['P4_path_cell']:.4f} |")

    a1, a4 = stat["기존 60"]
    b1, b4 = stat["추가 129"]
    d_a, d_b = (a4 - a1), (b4 - b1)
    diff = d_b.mean() - d_a.mean()
    se = np.sqrt(d_a.var(ddof=1) / len(d_a) + d_b.var(ddof=1) / len(d_b))
    z = znorm.ppf(0.975)
    L += ["", f"두 부분집합 delta 차이 (추가 129 − 기존 60): {diff:+.4f}, "
          f"SE={se:.4f}, 95% CI [{diff - z*se:+.4f}, {diff + z*se:+.4f}] "
          "(독립 두 표본, 쿼리별 차이값 d=P4−P1의 정규근사).", ""]

    # ---- Task C: the c rows ----
    for pool in ("aitqa", "hitab_lookup"):
        d = df[df.pool == pool]
        p1, p4, b, c = bc(d)
        cid = list(p1.index[(p4 == 0) & (p1 == 1)])
        g = d[(d.policy == "P4_path_cell") & (d.query_id.isin(cid))]
        hit = int(g.gold_in_topk.sum())
        L += [f"## Task C -- {pool}: c (P1만 정답) {len(cid)}건의 P4 조건", "",
              "| query_id | P4 n_chunks_used | P4 gold_in_topk | P4 gold_rank | "
              "P4 pred_parsed | gold |", "|---|---:|---|---:|---|---|"]
        for r in g.sort_values("query_id").itertuples():
            L.append(f"| {r.query_id} | {r.n_chunks_used} | {r.gold_in_topk} | "
                     f"{r.gold_rank} | `{r.pred_parsed}` | `{r.gold_answer}` |")
        L += ["", f"gold_in_topk=True 인데 오답: {hit} / {len(cid)}",
              f"gold_in_topk=False: {len(cid)-hit} / {len(cid)}", ""]

    out = Path("results/phase4/phase4e.md")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
