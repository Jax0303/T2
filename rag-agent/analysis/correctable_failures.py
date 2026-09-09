#!/usr/bin/env python3
"""Of the reader's wrong answers, how many are notation and how many are a wrong number?

A 'correction' that changes the numeric value is not the same thing as one that
only changes how it is written. The repo already learned this the expensive way:
+-1% tolerance scoring let 92 extra items pass and 76 of them were wrong answers
(TABLES.md, 2026-09-09). So the two kinds are counted separately and never summed.

  PYTHONPATH=. .venv/bin/python analysis/correctable_failures.py
"""
from __future__ import annotations
import json, re, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag_agent.eval.metrics import hitab_exact_match_text

D = ROOT / "results/retrieval_accuracy"
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def nums(x):
    """Every number in a prediction or a gold list, commas removed."""
    out = []
    for tok in (x if isinstance(x, list) else [x]):
        for m in NUM.findall(str(tok)):
            try:
                out.append(float(m.replace(",", "")))
            except ValueError:
                pass
    return sorted(out)


def norm(s):
    s = str(s).strip().lower().rstrip(".")
    s = re.sub(r"[$%,]", "", s)
    return " ".join(s.split())


def strings(x):
    return sorted(norm(t) for t in (x if isinstance(x, list) else [x]))


def rounds_to(a, b):
    """Is one a rounded/truncated writing of the other?"""
    for x, y in ((a, b), (b, a)):
        d = len(f"{x!r}".split(".")[1]) if "." in f"{x!r}" else 0
        if d <= 12 and round(y, d) == x:
            return True
    return False


def classify(pred, gold):
    p, g = nums(pred), nums(gold)
    ps, gs = strings(pred), strings(gold)
    if not p and not g:
        return "notation_string" if ps == gs else "different_text"
    # --- notation only: the numbers themselves are identical
    if p == g:
        return "notation_same_numbers"
    if len(p) > len(g) and set(g) <= set(p):
        return "extra_numbers_gold_present"
    # --- the number would have to change
    if len(p) == len(g) == 1:
        if rounds_to(p[0], g[0]):
            return "rounding"
        if g[0] and (abs(p[0] - g[0] * 100) < 1e-9 or abs(p[0] * 100 - g[0]) < 1e-9):
            return "rescale_100"
        if abs(p[0] + g[0]) < 1e-9 and p[0] != 0:
            return "sign_flip"
    if not p:
        return "no_number_predicted"
    return "different_value"


SAFE = {"notation_same_numbers", "notation_string"}
RISKY = {"extra_numbers_gold_present", "rounding", "rescale_100", "sign_flip"}


def report(name, path, ids, byq):
    rows = {json.loads(l)["query_id"]: json.loads(l) for l in open(path)}
    sub = [q for q in ids if q in rows]
    wrong = [q for q in sub if not hitab_exact_match_text(rows[q]["pred"], rows[q]["answer"])]
    c = Counter(classify(rows[q]["pred"], rows[q]["answer"]) for q in wrong)
    ex = {}
    for q in wrong:
        k = classify(rows[q]["pred"], rows[q]["answer"])
        ex.setdefault(k, []).append({"query_id": q, "pred": rows[q]["pred"],
                                     "gold": rows[q]["answer"],
                                     "retrieval_correct": byq[q]["correct"]})
    out = {"condition": name, "source": str(Path(path).relative_to(ROOT)),
           "n": len(sub), "correct": len(sub) - len(wrong),
           "em": round((len(sub) - len(wrong)) / len(sub), 4), "n_wrong": len(wrong),
           "safe_notation_only": sum(c[k] for k in SAFE),
           "value_changing": sum(c[k] for k in RISKY),
           "genuinely_wrong": len(wrong) - sum(c[k] for k in SAFE | RISKY),
           "breakdown": dict(c.most_common()),
           "examples": {k: v[:4] for k, v in ex.items()}}
    return out


def main():
    recs = [json.loads(l) for l in open(D / "t_s3c_hybrid_records.jsonl")]
    recs = [r for r in recs if "correct" in r]
    byq = {r["query_id"]: r for r in recs}
    ids = {r["query_id"] for r in recs
           if r["mode"] == "all" and (r.get("aggregation") or "none") == "none" and r["m"] == 1}
    res = [report("retrieved (top-20 cells)", D / "t_s3c_hybrid_answer_retrieved.jsonl", ids, byq),
           report("gold cells only (reader ceiling)", D / "t_s3c_hybrid_answer_gold_v2.jsonl", ids, byq)]
    (ROOT / "results/cell_id_v2/CORRECTABLE_FAILURES.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2))
    for r in res:
        print(f"\n=== {r['condition']}  n={r['n']}  EM={r['em']}  wrong={r['n_wrong']}")
        print(f"  표기만 다름 (숫자 동일)   {r['safe_notation_only']:>4}")
        print(f"  값을 바꿔야 통과         {r['value_changing']:>4}")
        print(f"  진짜 오답                {r['genuinely_wrong']:>4}")
        for k, v in r["breakdown"].items():
            print(f"      {k:32}{v:>4}")


if __name__ == "__main__":
    main()
