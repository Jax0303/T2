#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""정답 셀만 줬는데 왜 틀렸나 — 채점기 탓과 리더 탓을 가른다.

`gold` 조건은 리더에게 그 질의의 정답 셀만 넣는다. 그래도 1,245건 중 190건이
틀린다. 그 190건이 전부 "리더가 못 읽었다"는 아니다. HiTab 의 정답 표기 관행과
공식 EM 의 엄격함이 만드는 몫이 섞여 있고, 둘은 처방이 정반대다:

  채점기 몫  리더를 바꿔도 안 없어진다. 사람이 보면 맞은 답이다.
             - 순서만 다름            gold [91.7, 96.6] / pred "96.6, 91.7"
             - 스케일 100배           gold 0.691446 (비율) / pred "69.2%"
             - 반올림                 gold 2.177774 / pred "2.2"
  리더 몫    검색으로도 채점기로도 못 고친다. 리더를 바꿔야 한다.
             - 집계 미수행            gold 60 / pred "27.0,33.0" (27+33=60)
             - 오독·환각

이 파일이 내는 "관대 채점" 수치는 **진단 전용**이다. 공식 EM 이 아니므로
논문·명세의 성능 수치로 쓰지 말 것 (`CLAUDE.md` §8: NM/em_norm 은 진단 전용).

  PYTHONPATH=. .venv/bin/python analysis/reader_ceiling_anatomy.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
D = ROOT / "results/retrieval_accuracy"
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def floats(x):
    out = []
    for m in NUM.finditer(str(x)):
        try:
            out.append(float(m.group(0).replace(",", "")))
        except ValueError:
            pass
    return out


def gold_floats(g):
    vals = g if isinstance(g, list) else [g]
    out = []
    for v in vals:
        f = floats(v)
        out.extend(f if f else [])
    return out


def close(a, b, tol=0.011):
    return abs(a - b) <= tol * max(abs(b), 1e-9) or abs(a - b) < 1e-6


def is_rounded(pred, gold):
    """pred 가 gold 를 자릿수만 줄여 쓴 것인가.

    HiTab 은 계산 답을 float 전체 자릿수로 싣는다(2.177774). 모델이 "2.2"라고
    쓰면 공식 EM 은 오답이지만 같은 수다. 소수 N자리 반올림과 유효숫자 N자리
    반올림을 N<=4 까지 본다 — 임의의 허용오차가 아니라 "자릿수를 줄였는가"다.
    """
    from decimal import Decimal
    for n in range(5):
        if abs(round(gold, n) - pred) < 1e-9:
            return True
    for n in range(1, 5):                     # 유효숫자
        if gold == 0:
            break
        try:
            r = float(f"%.{n}g" % gold)
        except (ValueError, OverflowError):
            break
        if abs(r - pred) < 1e-9:
            return True
    return False


def diagnose(pred, gold):
    """(분류, 관대채점에서는 맞다고 볼 것인가)"""
    p, g = floats(pred), gold_floats(gold)
    if not str(pred).strip():
        return "빈 응답", False
    if not g:
        return "gold 가 숫자 아님", False
    if not p:
        return "숫자 아닌 응답", False

    # 순서만 다름 — 집합은 같고 순서가 다르다
    if len(p) == len(g) and sorted(p) != p and sorted(p) == sorted(g):
        return "순서만 다름", True
    if len(p) == len(g) and all(close(a, b) for a, b in zip(sorted(p), sorted(g))) \
            and not all(close(a, b) for a, b in zip(p, g)):
        return "순서만 다름", True

    # 값 하나짜리 비교
    if len(g) == 1 and len(p) == 1:
        a, b = p[0], g[0]
        if close(a, b) or is_rounded(a, b):
            return "반올림·자릿수", True
        for k in (100.0, 0.01):
            if close(a, b * k) or is_rounded(a, b * k):
                return "스케일 100배 (비율 대 퍼센트)", True
        # 집계 미수행: 예측한 여러 값이 아니라 하나인데, gold 가 그 값들의 연산 결과
        return "오독 — 다른 값을 답함", False

    # 예측이 여러 값인데 gold 는 하나 — 피연산자를 나열하고 집계를 안 한 경우
    if len(g) == 1 and len(p) >= 2:
        b = g[0]
        if close(sum(p), b):
            return "집계 미수행 — 합을 안 냄", False
        for x, y in combinations(p, 2):
            if close(abs(x - y), b) or (y and close(x / y, b)) or (x and close(y / x, b)):
                return "집계 미수행 — 연산을 안 냄", False
        return "오독 — 다른 값을 답함", False

    # gold 가 여러 값
    if len(g) >= 2:
        if len(p) < len(g):
            if all(any(close(a, b) for b in g) for a in p):
                return "값 일부 누락", False
            return "오독 — 다른 값을 답함", False
        if len(p) == len(g) and all(close(a, b) for a, b in zip(p, g)):
            return "반올림·자릿수", True
        return "오독 — 다른 값을 답함", False
    return "오독 — 다른 값을 답함", False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leg", default="gold", choices=["gold", "retrieved"])
    args = ap.parse_args()

    rows = [json.loads(l) for l in (D / f"t_s3c_hybrid_answer_{args.leg}.jsonl").open()]
    rows = [r for r in rows if r["mode"] == "all"]
    wrong = [r for r in rows if not r["answer_correct"]]
    n = len(rows)
    print(f"\n조건 `{args.leg}` — {n}건 중 오답 {len(wrong)}건 "
          f"(공식 EM {1 - len(wrong)/n:.4f})\n")

    cls, lenient_ok, ex = Counter(), 0, defaultdict(list)
    for r in wrong:
        c, ok = diagnose(r["pred"], r["answer"])
        cls[c] += 1
        lenient_ok += int(ok)
        if len(ex[c]) < 3:
            ex[c].append((r["answer"], r["pred"][:44].replace("\n", " ")))

    SCORER = {"순서만 다름", "반올림·자릿수", "스케일 100배 (비율 대 퍼센트)"}
    s_cnt = sum(v for k, v in cls.items() if k in SCORER)
    print(f"{'분류':32} {'건수':>5} {'오답 중':>7}  처방")
    print("-" * 74)
    for k, v in cls.most_common():
        who = "채점기" if k in SCORER else "리더"
        print(f"{k:32} {v:5} {v/len(wrong):>6.1%}  {who}")
    print("-" * 74)
    print(f"{'채점기 몫 (사람이 보면 정답)':32} {s_cnt:5} {s_cnt/len(wrong):>6.1%}")
    print(f"{'리더 몫 (진짜 오답)':32} {len(wrong)-s_cnt:5} "
          f"{(len(wrong)-s_cnt)/len(wrong):>6.1%}")
    print(f"\n공식 EM                       : {1 - len(wrong)/n:.4f}")
    print(f"관대 채점(순서·반올림·스케일 허용): {1 - (len(wrong)-s_cnt)/n:.4f}  "
          f"<- 진단 전용, 성능 수치로 쓰지 말 것")

    print("\n예시:")
    for k, v in cls.most_common():
        print(f"\n  [{k}]")
        for g, p in ex[k]:
            print(f"    gold={g}  pred={p!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
