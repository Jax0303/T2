"""Paper-aligned evaluation metrics for HiTab table QA.

Retrieval (HiTab paper / DTR, Herzig et al. NAACL 2021)
-------------------------------------------------------
  Recall@k     fraction of queries whose gold table is in the top-k.
  MRR          mean reciprocal rank of the gold table.
  nDCG@k       binary relevance (gold=1) on log2(i+1) discount.

Answer accuracy (HiTab paper, §5)
---------------------------------
  Exact Match (EM)        string equality after light normalisation.
  Numeric Match (NM)      relative-tolerance numeric equality with the
                           paper-customary variants (×100 percent form,
                           ÷100 fraction form, abs() for opposite/sign).
                           We default to ±2% rel-tol — the same threshold
                           used in the existing hard-query bench.

Formula execution accuracy (added by us)
----------------------------------------
  An evaluation that DOES NOT EXIST verbatim in the HiTab paper but is
  inspired by HiTab Table 9 ("execution accuracy of seq2seq with formula
  supervision"). Reports the fraction of arithmetic-class queries whose
  *symbolic* answer (deterministic eval over extracted cells) matched the
  gold answer — orthogonal to whether the natural-language reader also
  produced the answer. Use it to separate routing failures from arithmetic
  failures.

Difficulty stratification
-------------------------
  ``difficulty_class(sample)`` mirrors the HiTab paper's appendix tags
  (aggregation array + Excel formula op count). Kept here so the eval
  harness is self-contained.
"""
from __future__ import annotations

import math
import re
from typing import Iterable, List


_NUM_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+\.?\d*")
_OP_RE = re.compile(r"[+\-*/]")


HARD_CLASSES: List[str] = [
    "multi_op_formula",
    "arithmetic_agg",
    "pair_or_topk_arg",
    "single_arg",
    "comparison_or_count",
    "single_op_formula",
    "simple_lookup",
]


# --- normalisation helpers -------------------------------------------------

def _to_nums(s) -> List[float]:
    if isinstance(s, (int, float)):
        return [float(s)]
    if isinstance(s, list):
        out: List[float] = []
        for x in s:
            out.extend(_to_nums(x))
        return out
    if s is None:
        return []
    out = []
    for m in _NUM_RE.findall(str(s)):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            continue
    return out


def _flatten_strs(g) -> List[str]:
    if isinstance(g, list):
        return [str(x) for x in g]
    return [str(g)] if g is not None else []


# --- answer-side metrics ---------------------------------------------------

def numeric_match(pred, gold, rel_tol: float = 0.02) -> bool:
    """Tolerant numeric / substring match (matches the existing eval).

    For numeric gold: accept exact, ±rel_tol, ×100 (% form), ÷100 (fraction),
    and abs() (covers HiTab "opposite" / sign-change cases). For string
    gold: case-insensitive substring either direction (handles list-style
    gold like ``['quebec']``).
    """
    if pred is None:
        return False
    pred_s = str(pred).strip().lower()
    g_nums = _to_nums(gold)
    p_nums = _to_nums(pred)
    if g_nums:
        p_variants = [
            {round(x, 2) for x in p_nums},
            {round(x * 100, 2) for x in p_nums},
            {round(x / 100, 4) for x in p_nums},
            {round(abs(x), 2) for x in p_nums},
        ]
        for g in g_nums:
            g_cands = [round(g, 2), round(g * 100, 2), round(g / 100, 4), round(abs(g), 2)]
            ok = False
            for gc in g_cands:
                for pv in p_variants:
                    if gc in pv:
                        ok = True
                        break
                    for pn in pv:
                        if abs(pn - gc) / max(abs(gc), 1e-9) < rel_tol:
                            ok = True
                            break
                    if ok:
                        break
                if ok:
                    break
            if not ok:
                return False
        return True
    for gs in (s.strip().lower() for s in _flatten_strs(gold) if s.strip()):
        if gs in pred_s or pred_s in gs:
            return True
    return False


def exact_match(pred, gold) -> bool:
    pred_s = str(pred or "").strip().lower()
    for gs in _flatten_strs(gold):
        if pred_s == gs.strip().lower():
            return True
    return False


# --- retrieval-side metrics ------------------------------------------------

def recall_at_k(ranked_ids: List[str], gold_id: str, k: int) -> int:
    return int(gold_id in ranked_ids[:k])


def mrr(ranked_ids: List[str], gold_id: str) -> float:
    for i, x in enumerate(ranked_ids, 1):
        if x == gold_id:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_ids: List[str], gold_id: str, k: int = 10) -> float:
    """Binary nDCG@k with single-gold relevance."""
    dcg = 0.0
    for i, x in enumerate(ranked_ids[:k], 1):
        if x == gold_id:
            dcg += 1.0 / math.log2(i + 1)
            break
    return dcg  # ideal DCG = 1.0 (single gold)


# --- difficulty stratification ---------------------------------------------

def _formula_ops(sample: dict) -> int:
    fs = sample.get("answer_formulas") or []
    if not fs:
        return 0
    return max(len(_OP_RE.findall(f.lstrip("="))) for f in fs)


def difficulty_class(sample: dict) -> str:
    """Map HiTab supervision (`aggregation` + `answer_formulas`) to a label.

    Same definition as `scripts/run_hard_query_eval.py` so the two evals are
    directly comparable.
    """
    agg = tuple(sorted(set(sample.get("aggregation") or ["none"])))
    ops = _formula_ops(sample)
    if ops >= 2:
        return "multi_op_formula"
    if any(a in agg for a in ("div", "sum", "diff", "average", "range")):
        return "arithmetic_agg"
    if any(a in agg for a in ("pair-argmax", "pair-argmin", "topk-argmax", "topk-argmin", "kth-argmax")):
        return "pair_or_topk_arg"
    if any(a in agg for a in ("argmax", "argmin", "max", "min")):
        return "single_arg"
    if any(a in agg for a in ("greater_than", "less_than", "opposite", "counta")):
        return "comparison_or_count"
    if ops == 1:
        return "single_op_formula"
    return "simple_lookup"


# --- aggregation utility ---------------------------------------------------

def macro_avg(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
