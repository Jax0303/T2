"""Combine vector score and verification confidence. Supports filter and rerank modes."""
from __future__ import annotations

from typing import Dict, List, Tuple


def filter_only(hits: List[Dict], threshold: float = 0.3) -> List[Dict]:
    """Drop hits whose verification.confidence < threshold; preserve vector order."""
    kept: List[Dict] = []
    for h in hits:
        v = h.get("verification") or {}
        conf = v.get("confidence", 0.0) if h.get("verification_status") == "ok" else 0.0
        if conf >= threshold:
            kept.append({**h, "verify_conf": conf, "kept_by_filter": True})
        # else dropped silently
    return kept


def rerank(
    hits: List[Dict], w_vector: float = 0.7, w_verify: float = 0.3
) -> List[Dict]:
    out: List[Dict] = []
    for h in hits:
        v = h.get("verification") or {}
        conf = v.get("confidence", 0.0) if h.get("verification_status") == "ok" else 0.0
        fused = w_vector * float(h.get("score", 0.0)) + w_verify * float(conf)
        out.append({**h, "verify_conf": conf, "fused_score": fused})
    out.sort(key=lambda x: -x["fused_score"])
    return out


def filter_then_rerank(
    hits: List[Dict],
    threshold: float = 0.2,
    w_vector: float = 0.7,
    w_verify: float = 0.3,
) -> List[Dict]:
    kept = filter_only(hits, threshold=threshold)
    return rerank(kept, w_vector=w_vector, w_verify=w_verify)


def disagreement(hits_a: List[Dict], hits_b: List[Dict], top: int = 3) -> Dict:
    def rank_map(hs):
        return {h["table_id"]: i for i, h in enumerate(hs[:top])}

    ra, rb = rank_map(hits_a), rank_map(hits_b)
    top_set_a = set(ra.keys())
    top_set_b = set(rb.keys())
    overlap = top_set_a & top_set_b
    only_a = top_set_a - top_set_b
    only_b = top_set_b - top_set_a
    rank_shifts: List[Tuple[str, int]] = [(tid, ra[tid] - rb[tid]) for tid in overlap]
    return {
        "overlap@top": len(overlap),
        "vector_only": list(only_a),
        "verified_only": list(only_b),
        "rank_shifts": rank_shifts,
    }
