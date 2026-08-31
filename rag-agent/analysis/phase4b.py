#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Phase 4b -- P4 vs P1 significance, the gold-missing-but-correct rows, and
what the reader actually emitted on the arithmetic pool it failed."""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis"))

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402
from header_path_coverage import load_corpus                         # noqa: E402
from phase4_summary import em, gold_parts, norm_em                   # noqa: E402
from qwen_equiv_k import args_for                                    # noqa: E402
from scipy.stats import binomtest                                    # noqa: E402

POOLS = ["hitab_lookup", "hitab_arith", "aitqa", "rhb_fact", "rhb_num"]


def paired_boot(a, b, B=10000, seed=42):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = rng.integers(0, len(a), size=(B, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> int:
    rows = [json.loads(l) for l in open("results/phase4/reader_records.jsonl")]
    df = pd.DataFrame(rows)
    df["is_correct"] = [em(p, g) for p, g in zip(df.pred_parsed, df.gold_answer)]

    print("## Task A -- P4_path_cell vs P1_fixed_512, McNemar exact (paired)\n")
    print("| pool | n | P1 EM | P4 EM | b (P4만) | c (P1만) | delta | 95% CI | p |")
    print("|---|---:|---:|---:|---:|---:|---:|---|---:|")
    for pool in POOLS:
        d = df[df["pool"] == pool]
        p1 = d[d.policy == "P1_fixed_512"].set_index("query_id").is_correct
        p4 = d[d.policy == "P4_path_cell"].set_index("query_id").is_correct
        p4 = p4.reindex(p1.index)
        b = int(((p4 == 1) & (p1 == 0)).sum())
        c = int(((p4 == 0) & (p1 == 1)).sum())
        lo, hi = paired_boot(p4.values, p1.values)
        p = (binomtest(b, b + c, 0.5).pvalue if b + c else None)
        print(f"| {pool} | {len(p1)} | {p1.mean():.4f} | {p4.mean():.4f} | {b} | {c} "
              f"| {p4.mean()-p1.mean():+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
              f"{'n/a (b+c=0)' if p is None else f'{p:.4f}'} |")
    print("\np는 다중비교 보정 전 값이다. 보정하지 않았다.\n")

    # --- cell value lookup, for Task B ---
    val = {}
    for ds, pop in (("hitab", "hitab_dev_lookup_all"), ("aitqa", "aitqa"),
                    ("realhitbench", "rhb_lookup_all")):
        C = load_corpus(args_for(ds, pop))
        for (t, i, j), (rp, cp, v) in zip(C.cell_owner, C.cell_paths):
            val[(t, i, j)] = v

    print("\n## Task B -- gold_in_topk=False 인데 정답인 행 전건\n")
    sel = df[(df.policy != "gold_cell") & (~df.gold_in_topk) & (df.is_correct == 1)]
    print(f"n = {len(sel)}\n")
    for r in sel.itertuples():
        top = json.loads(r.retrieved_topk)
        g = norm_em(gold_parts(r.gold_answer)[0]) if len(
            gold_parts(r.gold_answer)) == 1 else None
        found = []
        for c in top:
            for (i, j) in c["cells"]:
                v = val.get((c["table_id"], i, j))
                if v is not None and g is not None and norm_em(v) == g:
                    found.append((c["table_id"], i, j))
        print(f"- [{r.policy} / {r.pool}] query_id={r.query_id}")
        print(f"  query: {r.query}")
        print(f"  gold_answer={r.gold_answer!r}  pred_parsed={r.pred_parsed!r}")
        print(f"  gold_table_id={r.gold_table_id}  gold_cell={r.gold_cell}")
        print(f"  같은 값 셀 컨텍스트 내 존재: {'Y' if found else 'N'}"
              f"  (n={len(found)})")
        if found:
            print(f"  위치: {found[:20]}")
        print()

    print("\n## Task C -- hitab_arith / gold_cell 오답 전건\n")
    d = df[(df.pool == "hitab_arith") & (df.policy == "gold_cell")]
    bad = d[d.is_correct == 0]
    print(f"n = {len(bad)} / {len(d)}\n")
    print("| query_id | gold | pred_answer_raw | 주입 셀 값 | 분류 |")
    print("|---|---|---|---|---|")
    cnt = defaultdict(int)
    for r in bad.itertuples():
        cells = [tuple(x) for x in json.loads(r.gold_cell)]
        vals = [str(val.get(tuple(c))) for c in cells]
        pr = norm_em(r.pred_parsed)
        if any(pr == norm_em(v) for v in vals):
            k = "(i) 셀 값 그대로"
        elif re.fullmatch(r"-?[\d.]+", pr or "x"):
            k = "(ii) 계산했으나 틀림"
        else:
            k = "(iii) 형식 이상"
        cnt[k] += 1
        print(f"| {r.query_id[:8]} | {r.gold_answer} | {r.pred_answer_raw!r} | "
              f"{vals} | {k} |")
    print()
    for k in ("(i) 셀 값 그대로", "(ii) 계산했으나 틀림", "(iii) 형식 이상"):
        print(f"- {k}: {cnt[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
