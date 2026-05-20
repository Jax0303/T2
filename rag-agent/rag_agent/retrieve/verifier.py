"""Cross-verify a vector hit against the OriginalStore (NOT the chunk it came from).

This is the "원본과 벡터DB 동시에 비교/검증" step. The OriginalStore holds the
2D structure (header tree + cell values), so we can ask question-grounded
checks like "do this query's numbers appear in the actual cells?" and
"do this query's keywords appear in the header paths?" — both signals the
serialized chunk alone cannot reliably answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Set

from ..stores.original_store import OriginalTable, _to_float


_NUM_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+\.?\d*")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]+")
_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "for", "to", "from", "by", "with",
    "and", "or", "is", "was", "were", "be", "been", "are", "as", "it", "this",
    "that", "what", "which", "who", "where", "when", "why", "how", "many",
    "much", "do", "does", "did", "has", "have", "had", "than", "then", "into",
    "about", "over", "under", "between", "per", "out", "if", "would", "could",
    "should", "will", "can", "may", "more", "most", "less", "least", "any",
    "some", "all", "each", "their", "they", "them", "we", "i", "you",
}


def _keywords(text: str, min_len: int = 3) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")
            if len(w) >= min_len and w.lower() not in _STOPWORDS]


def _numbers(text: str) -> List[float]:
    out = []
    for m in _NUM_RE.findall(text or ""):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _header_tokens(t: OriginalTable) -> Set[str]:
    toks: Set[str] = set(_keywords(t.title))
    for p in t.top_paths + t.left_paths:
        for seg in p:
            toks.update(_keywords(seg))
    return toks


def _numeric_cells(t: OriginalTable) -> Set[float]:
    out: Set[float] = set()
    for row in t.data:
        for v in row:
            f = _to_float(v)
            if f is not None:
                out.add(f)
    return out


@dataclass
class VerifyResult:
    table_id: str
    keyword_overlap: float
    numeric_overlap: float
    matched_keywords: List[str]
    matched_numbers: List[float]
    confidence: float            # composite

    def to_dict(self) -> dict:
        return {
            "table_id": self.table_id,
            "keyword_overlap": round(self.keyword_overlap, 3),
            "numeric_overlap": round(self.numeric_overlap, 3),
            "matched_keywords": self.matched_keywords,
            "matched_numbers": self.matched_numbers,
            "confidence": round(self.confidence, 3),
        }


def verify_against_original(
    query: str,
    table: OriginalTable,
    w_keyword: float = 0.6,
    w_number: float = 0.4,
) -> VerifyResult:
    q_kws = set(_keywords(query))
    q_nums = _numbers(query)

    t_kws = _header_tokens(table)
    matched_kws = sorted(q_kws & t_kws)
    kw_ov = (len(matched_kws) / len(q_kws)) if q_kws else 0.0

    t_nums = _numeric_cells(table)
    matched_nums = [n for n in q_nums if any(abs(n - tn) < 1e-6 for tn in t_nums)]
    # No-numbers questions: don't penalise — set neutral 1.0 and let kw decide.
    num_ov = (len(matched_nums) / len(q_nums)) if q_nums else 1.0
    conf = (w_keyword * kw_ov + w_number * num_ov) if q_nums else kw_ov

    return VerifyResult(
        table_id=table.table_id,
        keyword_overlap=kw_ov,
        numeric_overlap=num_ov,
        matched_keywords=matched_kws,
        matched_numbers=matched_nums,
        confidence=conf,
    )


def rerank(
    query: str,
    hits: list,
    original_store,
    w_vector: float = 0.7,
    w_verify: float = 0.3,
) -> list:
    """Blend vector score and verifier confidence. Returns new list of dicts."""
    out = []
    for h in hits:
        rec = original_store.get(h.table_id)
        if rec is None:
            out.append({"table_id": h.table_id, "vector_score": h.score,
                        "verify_confidence": 0.0, "final_score": h.score * w_vector,
                        "verification": None})
            continue
        v = verify_against_original(query, rec)
        final = w_vector * h.score + w_verify * v.confidence
        out.append({
            "table_id": h.table_id, "vector_score": h.score,
            "verify_confidence": v.confidence, "final_score": final,
            "verification": v.to_dict(),
        })
    out.sort(key=lambda d: -d["final_score"])
    return out
