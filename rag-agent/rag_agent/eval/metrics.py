"""Paper-aligned evaluation metrics for HiTab table QA.

Retrieval (HiTab paper / DTR, Herzig et al. NAACL 2021)
-------------------------------------------------------
  Recall@k     fraction of queries whose gold table is in the top-k.
  MRR          mean reciprocal rank of the gold table.
  nDCG@k       binary relevance (gold=1) on log2(i+1) discount.

Answer accuracy
---------------
  hitab_exact_match       HiTab's OWN official scorer, ported verbatim from
                           the dataset's evaluation harness
                           (qa/table/utils.py: hmt_score/hmt_equal/
                           hmt_process_answer in the HiTab repo). Numbers are
                           compared with a 1e-5 float tolerance (i.e. exact,
                           no % conversion, no abs()); strings are compared
                           after WikiTableQuestions-style normalisation. This
                           is the number that is directly comparable to any
                           other paper's reported HiTab accuracy/EM (e.g.
                           OHD 2602.01969's "60.07 EM") — use this, not
                           numeric_match, whenever a number will be quoted
                           against another paper.
  Numeric Match (NM)      OUR OWN more forgiving diagnostic metric — relative-
                           tolerance numeric equality (default ±2%) plus
                           ×100/÷100/abs() variants, meant to separate
                           "silently wrong" from "right value, sign/percent-
                           form mismatch" for internal error analysis. This is
                           NOT the dataset's official metric and is not
                           comparable to other papers' reported numbers —
                           report hitab_exact_match for that.
  Exact Match (EM)        plain case-insensitive string equality (no HiTab-
                           specific normalisation) — a cheap sanity check,
                           not a substitute for hitab_exact_match.

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
import unicodedata
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
    if not pred_s:                 # 빈 예측은 정답일 수 없음 ('' 가 모든 문자열 gold의 부분문자열이 되던 버그 수정)
        return False
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
        if gs in pred_s or pred_s == gs:   # gold가 예측에 등장(또는 정확일치). 역방향 pred⊂gold 제거(짧은 오답 오인 방지)
            return True
    return False


def exact_match(pred, gold) -> bool:
    pred_s = str(pred or "").strip().lower()
    for gs in _flatten_strs(gold):
        if pred_s == gs.strip().lower():
            return True
    return False


# --- HiTab's own official scorer (ported verbatim, for cross-paper comparison) ---
# Source: HiTab repo qa/table/utils.py (hmt_score / hmt_equal / hmt_process_answer)
# and qa/datadump/utils.py (naive_str_to_float / normalize, itself copied from the
# WikiTableQuestions official evaluator). Kept dependency-free here rather than
# importing the dataset repo's package.

_HMT_TOL = 1e-5


def _wtq_normalize(x: str) -> str:
    """WikiTableQuestions-style string normalisation (used by HiTab's own evaluator)."""
    if x is None:
        return None
    x = "".join(c for c in unicodedata.normalize("NFKD", x) if unicodedata.category(c) != "Mn")
    x = re.sub("[‘’´`]", "'", x)
    x = re.sub("[“”]", '"', x)
    x = re.sub("[‐‑‒–—−]", "-", x)
    while True:
        old_x = x
        x = re.sub(r"((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", x.strip())
        x = re.sub(r"(?<!^)( \([^)]*\))*$", "", x.strip())
        x = re.sub(r'^"([^"]*)"$', r"\1", x.strip())
        if x == old_x:
            break
    if x and x[-1] == ".":
        x = x[:-1]
    return re.sub(r"\s+", " ", x, flags=re.U).lower().strip()


def _hmt_str_to_float(s: str):
    sanitized = s
    try:
        if sanitized and sanitized[0] == "(":
            sanitized = sanitized[1:]
        if sanitized and (sanitized[-1] == "%" or sanitized[-1] == ")"):
            sanitized = sanitized[:-1]
        return float(sanitized.replace(",", ""))
    except (ValueError, IndexError):
        return _wtq_normalize(s)


def _hmt_process(v):
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        return _hmt_str_to_float(v.strip().lower())
    if isinstance(v, list):
        if len(v) == 1:
            return _hmt_process(v[0])
        return [_hmt_process(a) for a in v]
    return v


def _hmt_equal(p, g) -> bool:
    if type(p) is not type(g):
        return False
    if isinstance(p, str):
        return p == g
    if isinstance(p, float):
        return math.fabs(p - g) < _HMT_TOL
    if isinstance(p, list):
        return len(p) == len(g) and all(_hmt_equal(a, b) for a, b in zip(p, g))
    return p == g


def hitab_exact_match(pred, gold) -> bool:
    """HiTab's own official scorer: exact after minimal normalisation, no tolerance
    beyond floating-point noise. Use this (not ``numeric_match``) for any number
    that will be quoted alongside another paper's reported HiTab accuracy."""
    if pred is None:
        return False
    return _hmt_equal(_hmt_process(pred), _hmt_process(gold))


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
