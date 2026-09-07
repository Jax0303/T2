#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""3단계 보드 — 리더 사다리 (단일 → 다중 → 산술), 조건별 EM 과 짝지은 검정. 리더 재호출 없음.

입력은 `analysis/retrieved_answer_em.py` 의 jsonl. 조건마다 EM / gold 전부 주입률 /
EM|주입됨 / EM|안됨 / 프롬프트 토큰 중앙을 찍고, `top10` 을 기준으로 다른 조건과
exact McNemar 를 낸다. 산술에서 `EM|안됨` 은 정의상 0 이어야 한다(셀이 빠지면 오답) —
0 이 아니면 gold 라벨이나 채점기를 의심할 것.

  PYTHONPATH=analysis:. .venv/bin/python analysis/stage3_board.py results/stage3/*.jsonl \
      --out results/stage3/BOARD.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest

from phase4_summary import _one_num, em, gold_parts


def conv_em(p, g):
    """진단 전용: em 이거나, 부호 또는 x100(퍼센트<->비율) 표기만 다른 같은 수 (1% 허용).
    HiTab 의 `div` gold 는 비율(0.0706)인데 질문은 '몇 퍼센트'라 묻고, `opposite`/`diff` 는 감소를
    양수로 적는다. 값을 맞게 계산하고 표기만 다른 답을 세는 열이며 주지표가 아니다."""
    if em(p, g):
        return 1
    gs = gold_parts(g)
    if len(gs) != 1:
        return 0
    pv, gv = _one_num(p), _one_num(gs[0])
    if pv is None or gv is None or gv == 0:
        return 0
    return int(any(abs(abs(pv) - abs(gv) * k) / abs(gv * k) < 0.01 for k in (1, 100, 0.01)))

ORDER = ["gold", "orc10", "orc20", "top1", "top3", "top10", "top20"]


def load(path):
    R = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        r["ok"] = em(r["pred_parsed"], r["gold_answer"])
        r["cok"] = conv_em(r["pred_parsed"], r["gold_answer"])
        R[r["cond"]][r["query_id"]] = r
    return R


def mcnemar(a, b):
    qs = sorted(set(a) & set(b))
    x = sum(1 for q in qs if a[q]["ok"] and not b[q]["ok"])
    y = sum(1 for q in qs if not a[q]["ok"] and b[q]["ok"])
    p = binomtest(min(x, y), x + y, 0.5).pvalue if x + y else 1.0
    return x, y, p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="results/stage3/BOARD.md")
    a = ap.parse_args()
    L = ["# 3단계 보드 — 리더 사다리 (단일 셀 → 다중 셀 → 산술)", "",
         "계측기 `analysis/stage3_board.py`, 원본 `analysis/retrieved_answer_em.py` 출력.",
         "리더 Qwen2.5-7B-Instruct 4-bit NF4 temp=0 seed=42 max_new_tokens=32, 채점 `phase4_summary.em`.",
         "arm `models/bge-base-cell-ft-p0` α=0.8 `--title-mode page`. 사전 기록 `PREREG-2026-09-05-reader-ladder.md`.", ""]
    for f in a.files:
        R = load(f)
        conds = [c for c in ORDER if c in R]
        n = len(next(iter(R.values())))
        L += [f"## `{Path(f).stem}`  (n={n}, 출처 `{f}`)", "",
              "| 조건 | EM | EM(표기허용) | gold 전부 주입 | EM\\|주입됨 | EM\\|안됨 | ptok 중앙 | vs top10 (앞만:뒤만, p) |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
        base = R.get("top10")
        for c in conds:
            rs = list(R[c].values())
            full = [r for r in rs if r["gold_in_ctx"] == r["m"]]
            miss = [r for r in rs if r["gold_in_ctx"] < r["m"]]
            rate = lambda s: f"{sum(r['ok'] for r in s) / len(s):.4f}" if s else "—"
            pt = sorted(r["prompt_tokens"] for r in rs)[len(rs) // 2]
            test = ""
            if base is not None and c != "top10":
                x, y, p = mcnemar(R[c], base)
                test = f"{x}:{y}, p={p:.3g}"
            crate = f"{sum(r['cok'] for r in rs) / len(rs):.4f}"
            L.append(f"| `{c}` | {rate(rs)} | {crate} | {len(full) / len(rs):.4f} | {rate(full)} | "
                     f"{rate(miss)} | {pt} | {test} |")
        L.append("")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
