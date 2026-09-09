#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""짝지은 이진 결과의 검정 — McNemar 정확검정, Holm 보정, paired bootstrap.

조건 넷을 같은 질의 집합에서 재므로 비교는 전부 **짝지음**이다. 독립 표본 검정을
쓰면 같은 질의가 두 번 세어져 p 가 작아진다.

  PYTHONPATH=. .venv/bin/python analysis/answer_stats.py     # 자체 검사
"""
from __future__ import annotations

from math import comb
from typing import Dict, List, Sequence, Tuple

import numpy as np


def mcnemar_exact(a: Sequence[int], b: Sequence[int]) -> Tuple[int, int, float]:
    """``(n01, n10, p)`` — a 만 맞은 수, b 만 맞은 수, 양측 정확검정 p.

    불일치쌍만 정보를 갖는다. 이항(n, 0.5)의 양측이므로 작은 쪽 꼬리를 두 배 하고
    1 에서 자른다. 근사(카이제곱)를 쓰지 않는 이유는 불일치쌍이 20~30 인 칸이
    실제로 나오기 때문이다.
    """
    if len(a) != len(b):
        raise ValueError("짝지음이 깨졌다: 길이가 다르다")
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    tail = sum(comb(n, i) for i in range(min(n01, n10) + 1))
    return n01, n10, min(1.0, 2 * tail / 2 ** n)


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm–Bonferroni. 단조성을 강제한 조정 p 를 이름별로 돌려준다."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def paired_bootstrap(a: Sequence[int], b: Sequence[int], n_boot: int = 10000,
                     seed: int = 42, alpha: float = 0.05) -> Dict[str, float]:
    """b − a 차이의 백분위 CI. **질의 단위로** 재표집해 짝지음을 유지한다."""
    if len(a) != len(b):
        raise ValueError("짝지음이 깨졌다: 길이가 다르다")
    A, B = np.asarray(a, float), np.asarray(b, float)
    n = len(A)
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    d = B[idx].mean(1) - A[idx].mean(1)
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"diff": float(B.mean() - A.mean()), "ci_lo": float(lo), "ci_hi": float(hi)}


def bootstrap_ci(v: Sequence[int], n_boot: int = 10000, seed: int = 42,
                 alpha: float = 0.05) -> Tuple[float, float]:
    """한 조건 EM 자체의 95% CI (표에 싣는 값)."""
    V = np.asarray(v, float)
    if V.size == 0:
        return (float("nan"), float("nan"))
    idx = np.random.default_rng(seed).integers(0, V.size, size=(n_boot, V.size))
    m = V[idx].mean(1)
    lo, hi = np.percentile(m, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _selfcheck() -> None:
    # 완전 일치하면 불일치쌍이 없고 p=1
    assert mcnemar_exact([1, 0, 1], [1, 0, 1]) == (0, 0, 1.0)
    # a 만 맞은 것이 5, b 만 맞은 것이 0 -> 양측 p = 2 * (1/2)^5
    n01, n10, p = mcnemar_exact([1] * 5 + [0], [0] * 5 + [0])
    assert (n01, n10) == (5, 0) and abs(p - 2 * 0.5 ** 5) < 1e-12, (n01, n10, p)
    # 대칭이면 방향이 바뀌어도 p 는 같다
    assert mcnemar_exact([1, 0], [0, 1])[2] == mcnemar_exact([0, 1], [1, 0])[2]
    # Holm: 가장 작은 p 에 m 배, 단조 증가
    h = holm({"a": 0.01, "b": 0.02, "c": 0.9})
    assert abs(h["a"] - 0.03) < 1e-12 and abs(h["b"] - 0.04) < 1e-12 and h["c"] == 0.9
    assert h["a"] <= h["b"] <= h["c"]
    # 보정 p 는 원래 p 이상이고 1 을 넘지 않는다
    h2 = holm({"x": 0.6, "y": 0.7})
    assert h2["x"] >= 0.6 and h2["y"] <= 1.0
    # bootstrap: 차이가 0 이면 CI 가 0 을 포함한다
    r = paired_bootstrap([1, 0] * 50, [1, 0] * 50, n_boot=500)
    assert r["diff"] == 0.0 and r["ci_lo"] <= 0 <= r["ci_hi"]
    # 차이가 크면 CI 가 0 을 넘지 않는다
    r2 = paired_bootstrap([0] * 100, [1] * 100, n_boot=500)
    assert r2["diff"] == 1.0 and r2["ci_lo"] > 0
    # 같은 seed 면 같은 값
    assert paired_bootstrap([1, 0] * 20, [0, 1] * 20, n_boot=300) == \
           paired_bootstrap([1, 0] * 20, [0, 1] * 20, n_boot=300)
    lo, hi = bootstrap_ci([1] * 60 + [0] * 40)
    assert lo < 0.6 < hi
    print("answer_stats 자체 검사 통과")


if __name__ == "__main__":
    _selfcheck()
