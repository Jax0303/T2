"""Query-aware verification: check that the QUERY's evidence appears in the candidate table.

v1 was tautological — it parsed numbers out of the retrieved chunk (which by
definition came from the candidate table) and re-matched them back to the same
table. That always scored ~1.0 regardless of relevance.

v2 verifies in two directions:
  - query→table  : do numbers / keyword spans from the QUERY appear in this table?
  - answer→table : (optional) does a candidate answer trace to a real cell?
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from ..store.table_store import TableRecord, TableStore

_NUM_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+\.?\d*")

_STOPWORDS: Set[str] = {
    "a", "an", "the", "of", "in", "on", "at", "for", "to", "from", "by",
    "with", "and", "or", "is", "was", "were", "be", "been", "being",
    "are", "as", "it", "this", "that", "these", "those", "what", "which",
    "who", "whom", "whose", "where", "when", "why", "how", "many", "much",
    "do", "does", "did", "has", "have", "had", "than", "then", "into",
    "about", "over", "under", "between", "per", "out", "if", "would",
    "could", "should", "will", "shall", "can", "may", "might", "there",
    "their", "they", "them", "us", "we", "i", "you", "your", "yours",
    "more", "most", "less", "least", "any", "some", "all", "each",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]+")


def _parse_numbers(text: str) -> List[float]:
    out: List[float] = []
    for m in _NUM_RE.findall(text or ""):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            continue
    return out


def _keywords(text: str, min_len: int = 3) -> List[str]:
    return [
        w.lower()
        for w in _WORD_RE.findall(text or "")
        if len(w) >= min_len and w.lower() not in _STOPWORDS
    ]


def _table_text_tokens(rec: TableRecord) -> Set[str]:
    """Collect all words from title + every header path of the table."""
    tokens: Set[str] = set()
    tokens.update(_keywords(rec.title))
    for path in rec.top_header_paths + rec.left_header_paths:
        for h in path:
            tokens.update(_keywords(str(h)))
    return tokens


def _table_numeric_cells(rec: TableRecord) -> Set[float]:
    out: Set[float] = set()
    for r in range(rec.df.shape[0]):
        for c in range(rec.df.shape[1]):
            v = rec.df.iat[r, c]
            if v is None:
                continue
            if isinstance(v, (int, float)):
                out.add(float(v))
            else:
                try:
                    out.add(float(str(v).replace(",", "")))
                except ValueError:
                    continue
    return out


@dataclass
class QueryVerification:
    table_id: str
    keyword_overlap: float        # 0..1 — Jaccard between query terms and table text
    numeric_overlap: float        # 0..1 — fraction of query numbers found in table cells
    matched_keywords: List[str]
    matched_numbers: List[float]
    missing_numbers: List[float]
    confidence: float             # composite (see weight args)

    def to_dict(self) -> Dict:
        return {
            "table_id": self.table_id,
            "keyword_overlap": round(self.keyword_overlap, 3),
            "numeric_overlap": round(self.numeric_overlap, 3),
            "matched_keywords": self.matched_keywords,
            "matched_numbers": self.matched_numbers,
            "missing_numbers": self.missing_numbers,
            "confidence": round(self.confidence, 3),
        }


def verify_query(
    query: str,
    rec: TableRecord,
    w_keywords: float = 0.6,
    w_numbers: float = 0.4,
) -> QueryVerification:
    q_kws = set(_keywords(query))
    q_nums = _parse_numbers(query)

    t_tokens = _table_text_tokens(rec)
    matched_kws = sorted(q_kws & t_tokens)
    kw_overlap = (len(matched_kws) / len(q_kws)) if q_kws else 0.0

    t_nums = _table_numeric_cells(rec)
    matched_nums = [n for n in q_nums if n in t_nums]
    missing_nums = [n for n in q_nums if n not in t_nums]
    num_overlap = (len(matched_nums) / len(q_nums)) if q_nums else 1.0  # no nums → not penalised

    # If the query has no numbers, only keyword overlap matters.
    if not q_nums:
        confidence = kw_overlap
    else:
        confidence = w_keywords * kw_overlap + w_numbers * num_overlap

    return QueryVerification(
        table_id=rec.table_id,
        keyword_overlap=kw_overlap,
        numeric_overlap=num_overlap,
        matched_keywords=matched_kws,
        matched_numbers=matched_nums,
        missing_numbers=missing_nums,
        confidence=confidence,
    )


def verify_hits(
    query: str,
    store: TableStore,
    hits: List[Dict],
    **kwargs,
) -> List[Dict]:
    out: List[Dict] = []
    for h in hits:
        rec = store.get(h["table_id"])
        if rec is None:
            out.append({**h, "verification": None, "verification_status": "table_missing"})
            continue
        v = verify_query(query, rec, **kwargs)
        out.append({**h, "verification": v.to_dict(), "verification_status": "ok"})
    return out
