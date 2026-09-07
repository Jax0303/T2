#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""표준 IR 지표 보드 — stage1 `*_ranks.jsonl` 을 Recall/MRR/nDCG 와 나란히 적는다.

`analysis/stage1_board.py` 와 **같은 입력·같은 모집단·같은 gold** 이고 채점 규칙만
문헌 표준으로 바꾼다 (교수 규칙: all-covered 를 단독으로 보고하지 않는다). 리더 없음.
지표 정의는 `scripts/standard_ir_metrics_from_records.py` 것을 그대로 가져다 쓴다 —
정의가 두 벌이 되면 안 된다. 그쪽은 1-based rank, stage1 덤프는 0-based 라 +1 한다.

`all-covered@k` 열은 `BOARD.md` 의 `전체@k` 와 같은 값이어야 한다 — 아래에서 assert 한다.

  PYTHONPATH=.:scripts:analysis python3 analysis/ir_metrics_board.py \
      results/stage1_clean/*_ranks.jsonl --out results/stage1_clean/IR_BOARD.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "analysis")]

import standard_ir_metrics_from_records as ir                        # noqa: E402
from stage1_board import POP, load                                   # noqa: E402

KS = (10, 20)
ir.KS = KS


def board(paths):
    rows = []
    for p in sorted(paths):
        R = load(p)
        name = (POP.match(Path(p).name).group(1) + ("_clean" if "_clean_" in p else ""))
        per_q = {r["query_id"]: [x + 1 for x in r["ranks"]] for r in R}
        assert len(per_q) == len(R), f"{name}: query_id 중복"
        s = ir.summarize(per_q)
        m = sorted(r["m"] for r in R)
        n = len(R)
        # R-Precision — 컷오프를 쿼리마다 k=m 으로. m 분포가 다른 모집단끼리 비교할 때 쓴다.
        # 같은 컷오프의 "전부 아니면 0" 판이 BOARD.md 의 c=1 열이다.
        s["r_prec"] = round(sum(sum(x < r["m"] for x in r["ranks"]) / r["m"] for r in R) / n, 4)
        # cell-F1@k — HotpotQA Sup F1 의 셀 단위 대응. |G∩R_k| 가 P·R·F1 을 다 정하므로
        # 쿼리 단위로는 Recall@k 의 재척도다 (새 신호 아님, 리뷰어가 기대하는 이름).
        for k in KS:
            s[f"f1@{k}"] = round(sum(2 * sum(x < k for x in r["ranks"]) / (k + r["m"])
                                     for r in R) / n, 4)
        # 같은 파일에서 나온 stage1_board 의 전체@k 와 정의가 어긋나면 여기서 죽는다.
        for k in KS:
            assert s[f"set_em@{k}"] == round(sum(max(r["ranks"]) < k for r in R) / len(R), 4)
        rows.append((name, len(R), m[len(m) // 2], m[-1], s))
    return rows


K = KS[-1]


def render(rows):
    head = ["모집단", "n", "m 중앙/최대", "MRR", "R-Prec",
            f"hit@{K}", f"Recall@{K}", f"nDCG@{K}", f"F1@{K}", f"**all-cov@{K}**",
            "(사전등록) all-cov@10"]
    L = ["# 표준 IR 지표 보드 — 같은 검색 결과, 채점 규칙만 다름", "",
         "계측기 `analysis/ir_metrics_board.py`. 입력·모집단·gold 는 `BOARD.md` 와 동일하고",
         "`all-cov@k` 열은 거기 `전체@k` 와 같은 값이다 (계측기가 assert 로 확인).", "",
         "- `hit@k` gold 셀이 **하나라도** 상위 k — 가장 관대",
         "- `Recall@k` gold 셀 중 상위 k 에 든 **비율** — 부분 점수",
         "- `nDCG@k` 이진 관련도, 다중 gold 이상 랭킹 기준 — 순위 위치까지 반영",
         "- `MRR` **첫** gold 셀 역순위 — 나머지 m−1 개를 아예 안 본다",
         "- `F1@k` |G∩R_k| 의 P·R 조화평균 — HotpotQA Sup F1 의 셀 단위 대응",
         "- `all-cov@k` gold 셀이 **전부** 상위 k — 전부 아니면 0 (주지표)",
         "- `R-Prec` 컷오프를 쿼리마다 k=m 으로 준 Recall — m 분포가 다른 모집단끼리 비교할 때",
         "  (같은 컷오프의 전부-아니면-0 판이 `BOARD.md` 의 `c=1` 열이다. **오라클이다** — m 을 알아야 한다)", "",
         "| " + " | ".join(head) + " |",
         "|" + "---|" + "---:|" * (len(head) - 1)]
    for name, n, mmed, mmax, s in rows:
        L.append("| " + " | ".join(
            [f"`{name}`", str(n), f"{mmed}/{mmax}", f"{s['mrr']:.4f}", f"{s['r_prec']:.4f}",
             f"{s[f'hit@{K}']:.4f}", f"{s[f'recall@{K}']:.4f}", f"{s[f'ndcg@{K}']:.4f}",
             f"{s[f'f1@{K}']:.4f}", f"**{s[f'set_em@{K}']:.4f}**", f"{s['set_em@10']:.4f}"]) + " |")
    L += ["", "m=1 모집단에서는 hit = Recall = all-cov 가 **정의상 같은 값**이다 — 지표가",
          "갈라지는 것은 m≥2 부터고, 그 간격이 곧 '관대한 지표가 가리는 불완전성'이다.",
          "m=1 에서는 `R-Prec` 도 정답셀 1등률과 같은 값이 된다 (k=m=1).", "",
          "Precision@k 는 싣지 않는다 — m≤k 이면 `Recall@k × m/k` 라 쿼리 단위로 새 신호가 없고,",
          "상한이 m/k 라 m=1 조회는 어떤 검색기로도 @20 에서 .05 를 못 넘는다."]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ranks", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = board(a.ranks)
    md = render(rows)
    if a.out:
        Path(a.out).write_text(md, encoding="utf-8")
        json.dump({n: s for n, _, _, _, s in rows},
                  open(Path(a.out).with_suffix(".json"), "w"), ensure_ascii=False, indent=1)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
