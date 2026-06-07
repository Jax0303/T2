#!/usr/bin/env python3
"""Re-score a result JSON in place with the CURRENT metrics.

Earlier runs were scored with a metric that wrongly counted an empty prediction
as correct (``'' in gold``). This recomputes ``correct`` from the stored
``pred``/``gold`` with the fixed ``numeric_match``/``exact_match``, adds a strict
exact-match (HiTab-style denotation match), and rewrites overall/by_class so the
committed artifact matches what we report.

Usage: python scripts/rescore.py results/method_sc_pc20.json [...more]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rag_agent.eval.metrics import numeric_match, exact_match, _to_nums  # noqa: E402

HARD = ["multi_op_formula", "arithmetic_agg", "pair_or_topk_arg", "single_arg", "comparison_or_count"]


def strict_em(pred, gold) -> bool:
    """Exact denotation match: numbers equal (no tolerance), else normalized string eq."""
    pn, gn = _to_nums(pred), _to_nums(gold)
    if gn:
        return any(abs(p - g) <= 1e-6 for p in pn for g in gn) if pn else False
    ps = str(pred).strip().lower()
    if not ps:
        return False
    gs = [str(x).strip().lower() for x in (gold if isinstance(gold, list) else [gold])]
    return ps in gs


def rescore_file(path: str) -> dict:
    d = json.load(open(path))
    rows = [r for r in d.get("rows", []) if "pred" in r]
    for r in rows:
        r["correct"] = bool(numeric_match(r["pred"], r["gold"]) or exact_match(r["pred"], r["gold"]))
        r["em"] = bool(strict_em(r["pred"], r["gold"]))
    n = len(rows)
    nm = sum(r["correct"] for r in rows) / n if n else 0.0
    em = sum(r["em"] for r in rows) / n if n else 0.0
    d.setdefault("overall", {})["NM"] = round(nm, 4)
    d["overall"]["EM"] = round(em, 4)
    d["overall"]["n"] = n
    d["scoring"] = "fixed (empty-pred bug corrected); NM=±2% numeric_match, EM=strict denotation"
    bc = {}
    for c in HARD:
        cr = [r for r in rows if r.get("class") == c]
        if cr:
            bc[c] = {"n": len(cr),
                     "NM": round(sum(r["correct"] for r in cr) / len(cr), 4),
                     "EM": round(sum(r["em"] for r in cr) / len(cr), 4)}
    d["by_class"] = bc
    json.dump(d, open(path, "w"), ensure_ascii=False, indent=2)
    return {"file": Path(path).name, "n": n, "NM": round(nm, 4), "EM": round(em, 4)}


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(rescore_file(p))
