"""Retrieval metrics + paired bootstrap for the preprocessing experiment.

Per-query unit of analysis: the rank of the gold table under each condition.
Significance between two conditions is a paired bootstrap (resample queries,
10k iterations, fixed seed) on the difference of hit@k indicators — same
procedure as scripts/bootstrap_ci.py, generalized to arbitrary rank lists.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple


def recall_at_k(ranks: Sequence[Optional[int]], k: int) -> float:
    """``ranks`` holds the 1-based gold rank per query (None = not in top-N)."""
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr(ranks: Sequence[Optional[int]]) -> float:
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks if r is not None) / len(ranks)


def paired_delta_bootstrap(
    ranks_a: Sequence[Optional[int]],
    ranks_b: Sequence[Optional[int]],
    k: int,
    n_iters: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """95% CI of recall@k(A) − recall@k(B), paired by query.

    Returns (mean_delta, ci_low, ci_high). The interval excluding 0 means
    the two conditions differ significantly at the 5% level (two-sided).
    """
    assert len(ranks_a) == len(ranks_b), "paired bootstrap needs aligned queries"
    n = len(ranks_a)
    hits_a = [1 if r is not None and r <= k else 0 for r in ranks_a]
    hits_b = [1 if r is not None and r <= k else 0 for r in ranks_b]
    mean_delta = (sum(hits_a) - sum(hits_b)) / n

    rng = random.Random(seed)
    deltas = []
    for _ in range(n_iters):
        s = 0
        for _ in range(n):
            i = rng.randrange(n)
            s += hits_a[i] - hits_b[i]
        deltas.append(s / n)
    deltas.sort()
    return (mean_delta,
            deltas[int(0.025 * n_iters)],
            deltas[int(0.975 * n_iters)])


def summarize_condition(ranks: Sequence[Optional[int]], ks: Sequence[int]) -> Dict:
    out = {f"R@{k}": round(recall_at_k(ranks, k), 4) for k in ks}
    out["MRR"] = round(mrr(ranks), 4)
    out["n"] = len(ranks)
    return out
