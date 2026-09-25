"""판정자 두 명의 예/아니오 판정 일치도 (Cohen's κ). 사전등록: PREREG-2026-09-26-judging-criteria.md §5.
열 이름은 앞부분만 줘도 된다("(가)" -> "(가) 질문만 ..."). 두 파일의 query_id 순서가 같아야 하고, 판정 값은 예/아니오만.
95% CI = 문항 부트스트랩 2,000회(seed 20260926), 백분위. 한쪽 판정이 한 값뿐인 재표본(κ 정의 안 됨)은 빼고 그 수를 적는다.
실행: .venv/bin/python scripts/judge_kappa.py A.csv B.csv "(가)" ["(다)" ...]
"""
import csv
import json
import math
import random
import sys

from sklearn.metrics import cohen_kappa_score

VALUES = ["예", "아니오"]


def column(rows, prefix):
    (name,) = [k for k in rows[0] if k.startswith(prefix)]
    vals = [r[name].strip() for r in rows]
    bad = [i for i, v in enumerate(vals, 1) if v not in VALUES]
    if bad:
        raise SystemExit(f"{prefix}: 예/아니오가 아닌 행 번호 {bad[:10]}")
    return vals


def kappa(a, b):
    if len(set(a)) == len(set(b)) == 1:
        return 1.0 if a == b else float("nan")
    return float(cohen_kappa_score(a, b, labels=VALUES))


def report(a, b, n_boot=2000, seed=20260926):
    rng, n = random.Random(seed), len(a)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(kappa([a[i] for i in idx], [b[i] for i in idx]))
    ok = sorted(k for k in boots if not math.isnan(k))
    return {"query_count": n, "agree": sum(x == y for x, y in zip(a, b)),
            "agreement": round(sum(x == y for x, y in zip(a, b)) / n, 4), "kappa": round(kappa(a, b), 4),
            "ci95": [round(ok[int(.025 * len(ok))], 4), round(ok[int(.975 * len(ok)) - 1], 4)],
            "bootstrap_undefined": n_boot - len(ok),
            "table_A_by_B": {f"{x}|{y}": sum(p == x and q == y for p, q in zip(a, b)) for x in VALUES for y in VALUES}}


if __name__ == "__main__":
    fa, fb, *cols = sys.argv[1:]
    A, B = (list(csv.DictReader(open(f, encoding="utf-8-sig"))) for f in (fa, fb))
    assert [r["query_id"] for r in A] == [r["query_id"] for r in B], "두 파일의 query_id 순서가 다르다"
    print(json.dumps({"A": fa, "B": fb, **{c: report(column(A, c), column(B, c)) for c in cols}},
                     indent=1, ensure_ascii=False))
