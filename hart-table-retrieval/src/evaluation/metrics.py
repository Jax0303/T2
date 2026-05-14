import math
from typing import List, Set, Union


def _to_set(relevant) -> Set[str]:
    if isinstance(relevant, str):
        return {relevant}
    return set(relevant)


def recall_at_k(predicted: List[str], relevant, k: int) -> float:
    rel = _to_set(relevant)
    return 1.0 if any(p in rel for p in predicted[:k]) else 0.0


def ndcg_at_k(predicted: List[str], relevant, k: int) -> float:
    rel = _to_set(relevant)
    dcg = 0.0
    for i, p in enumerate(predicted[:k]):
        if p in rel:
            dcg += 1.0 / math.log2(i + 2)
    idcg = 1.0 / math.log2(2)
    return dcg / idcg


def hit_at_k(predicted: List[str], relevant, k: int = 1) -> float:
    rel = _to_set(relevant)
    return 1.0 if predicted and predicted[0] in rel else 0.0


def mrr(predicted: List[str], relevant) -> float:
    rel = _to_set(relevant)
    for i, p in enumerate(predicted):
        if p in rel:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_batch(
    all_predictions: List[List[str]],
    all_relevants: List,
    ks: List[int] = [1, 5, 10],
) -> dict:
    n = len(all_predictions)
    if n == 0:
        return {}

    results = {}
    for k in ks:
        results[f"Recall@{k}"] = sum(
            recall_at_k(p, r, k) for p, r in zip(all_predictions, all_relevants)
        ) / n
        results[f"nDCG@{k}"] = sum(
            ndcg_at_k(p, r, k) for p, r in zip(all_predictions, all_relevants)
        ) / n

    results["Hit@1"] = sum(
        hit_at_k(p, r, 1) for p, r in zip(all_predictions, all_relevants)
    ) / n
    results["MRR"] = sum(
        mrr(p, r) for p, r in zip(all_predictions, all_relevants)
    ) / n

    return results
