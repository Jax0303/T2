#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""답변 정확도 보드 — 한 표에 모든 실행, 그리고 검색몫/리더몫 분해. 리더 재호출 없음.

입력은 `analysis/retrieved_answer_em.py` 의 jsonl (stage3/ · answer_ret/). 각 (실행, 조건) 마다

    EM = Cov · EM|cov + (1 - Cov) · EM|¬cov

이 항등식이 성립하는지 assert 한다 (Cov = gold 셀 전부 주입률 = all-covered@k).
**arm·모집단은 여기서 안 적는다** — 파일 이름만 적고 라벨은 `METRICS.md` 가 갖는다
(`stage3_board.py:69` 이 arm 을 하드코딩해 틀린 전례가 있다).

  PYTHONPATH=analysis:. python3 analysis/answer_board.py \
      results/stage3/*.jsonl results/answer_ret/[ap]0_*.jsonl --out results/stage3/ANSWER_BOARD.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from phase4_summary import em

ORDER = ["gold", "orc50", "orc20", "orc10", "top20", "top10", "top5", "top3", "top1"]


def rows(path):
    R = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        R[r["cond"]].append((em(r["pred_parsed"], r["gold_answer"]), r["gold_in_ctx"] == r["m"]))
    out = []
    for c in [c for c in ORDER if c in R]:
        rs = R[c]
        cov = [ok for ok, c_ in rs if c_]
        unc = [ok for ok, c_ in rs if not c_]
        acc = sum(ok for ok, _ in rs) / len(rs)
        p_cov, p_unc = len(cov) / len(rs), len(unc) / len(rs)
        e_cov = sum(cov) / len(cov) if cov else 0.0
        e_unc = sum(unc) / len(unc) if unc else 0.0
        assert abs(p_cov * e_cov + p_unc * e_unc - acc) < 1e-9, (path, c)   # 분해 항등식
        out.append((c, len(rs), p_cov, e_cov, e_unc if unc else None, acc))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="results/stage3/ANSWER_BOARD.md")
    a = ap.parse_args()
    R = {f: rows(f) for f in a.files}
    L = ["# 정확도 보드 — 맞다 / 틀리다", "",
         "계측기 `analysis/answer_board.py`. 리더 재호출 없이 저장된 예측만 다시 채점했다.",
         "arm·모집단 라벨은 `METRICS.md` §0 에 있다 — 파일 이름으로 대조할 것.", "",
         "## 1. 정확도 (배포 조건 `top10`)", "",
         "문맥에 넣는 셀 10개는 **시스템 설정이지 지표 축이 아니다.** 질의마다 맞다/틀리다 하나씩이다.", "",
         "| 실행 | n | 검색 정확도 | 답변 정확도 | 리더 정확도 |",
         "|---|---:|---:|---:|---:|"]
    for f in a.files:
        for c, n, p_cov, e_cov, e_unc, acc in R[f]:
            if c == "top10":
                L.append(f"| `{Path(f).stem}` | {n} | {p_cov:.4f} | **{acc:.4f}** | {e_cov:.4f} |")
    L += ["", "검색 = 정답 셀이 문맥에 **전부** 들어왔나 · 답변 = 최종 답이 맞았나 ·",
          "리더 = 정답 셀을 다 줬을 때 맞혔나 (검색 실패를 분모에서 뺀 값).", "",
          "## 2. 분해 — 조건별 (진단)", "",
          "`EM = Cov · EM|cov + (1−Cov) · EM|¬cov` 가 전 행에서 성립한다 (`assert`).",
          "⚠️ `orc*` 는 오라클이다. 성능 주장에 쓰지 말 것.", "",
          "| 실행 | 조건 | n | Cov (검색) | EM\\|cov (리더) | EM\\|¬cov (누출) | **EM** (종단) |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for f in a.files:
        for c, n, p_cov, e_cov, e_unc, acc in R[f]:
            L.append(f"| `{Path(f).stem}` | `{c}` | {n} | {p_cov:.4f} | {e_cov:.4f} | "
                     f"{'—' if e_unc is None else f'{e_unc:.4f}'} | **{acc:.4f}** |")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
