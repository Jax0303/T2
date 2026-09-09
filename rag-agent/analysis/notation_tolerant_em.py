#!/usr/bin/env python3
"""EM with gold's own unit/annotation suffix ignored, applied to every arm alike.

HiTab writes some gold answers as the cell reads: `326 days`, `6 q`, `6:14.60 or`,
`(192.1 overs)`. The reader answers `326`, `6`, `6:14.60`, `192.1` and the official
scorer rejects it. This variant accepts a prediction when the MULTISET OF NUMBERS
is identical to gold's -- nothing else moves. It does not accept a different
number, a rescaled one, or a list that merely contains the right value.

This is a SCORER CHANGE. Numbers under it are no longer directly comparable with
published HiTab EM, so both columns are always reported side by side.

  PYTHONPATH=. .venv/bin/python analysis/notation_tolerant_em.py
"""
from __future__ import annotations
import json, os, re, sys
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag_agent.eval.metrics import hitab_exact_match_text

D = ROOT / "results/retrieval_accuracy"
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def numbers(x):
    out = []
    for tok in (x if isinstance(x, list) else [x]):
        for m in NUM.findall(str(tok)):
            try:
                out.append(float(m.replace(",", "")))
            except ValueError:
                pass
    return sorted(out)


def norm(x):
    return sorted(" ".join(re.sub(r"[$%,()]", " ", str(t).strip().lower().rstrip(".")).split())
                  for t in (x if isinstance(x, list) else [x]))


def em_relaxed(pred, gold):
    """Official EM, or the same numbers written differently."""
    if hitab_exact_match_text(pred, gold):
        return True
    p, g = numbers(pred), numbers(gold)
    if p and p == g:
        return True                      # `326` vs `326 days`
    return bool(not p and not g and norm(pred) == norm(gold))


def main():
    recs = [json.loads(l) for l in open(D / "t_s3c_hybrid_records.jsonl")]
    recs = [r for r in recs if "correct" in r]
    pops = {
        "primary_991": {r["query_id"] for r in recs if r["mode"] == "all"
                        and (r.get("aggregation") or "none") == "none" and r["m"] == 1},
        "data_1245": {r["query_id"] for r in recs if r["mode"] == "all"},
        "header_336": {r["query_id"] for r in recs if r["mode"] == "any"},
        "all_1581": {r["query_id"] for r in recs},
    }
    files = sorted(glob(str(D / "*_answer_retrieved.jsonl"))) + [str(D / "t_s3c_hybrid_answer_gold_v2.jsonl")]
    table = []
    for f in files:
        rows = {json.loads(l)["query_id"]: json.loads(l) for l in open(f)}
        tag = os.path.basename(f).replace("_answer_retrieved.jsonl", "").replace(".jsonl", "")
        e = {"arm": tag}
        for pop, ids in pops.items():
            sub = [q for q in ids if q in rows]
            off = sum(bool(hitab_exact_match_text(rows[q]["pred"], rows[q]["answer"])) for q in sub)
            rel = sum(em_relaxed(rows[q]["pred"], rows[q]["answer"]) for q in sub)
            e[pop] = {"n": len(sub), "official": off, "official_em": round(off / len(sub), 4),
                      "relaxed": rel, "relaxed_em": round(rel / len(sub), 4), "gained": rel - off}
        table.append(e)
    out = {"rule": "official EM, or identical multiset of numbers (gold's unit/annotation suffix ignored)",
           "warning": "scorer change — not directly comparable with published HiTab EM",
           "arms": table}
    (ROOT / "results/cell_id_v2/NOTATION_TOLERANT_EM.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2))
    w = max(len(e["arm"]) for e in table)
    print(f"{'arm':{w}}  {'주지표 991 공식':>15} {'표기허용':>10} {'+':>4}   {'데이터셀 1245':>14} {'표기허용':>10} {'+':>4}")
    for e in sorted(table, key=lambda x: -x["primary_991"]["relaxed_em"]):
        a, b = e["primary_991"], e["data_1245"]
        print(f"{e['arm']:{w}}  {a['official']:>6}/{a['n']} {a['official_em']:>7} {b and a['relaxed_em']:>10} {a['gained']:>+4}   "
              f"{b['official']:>6}/{b['n']} {b['official_em']:>6} {b['relaxed_em']:>10} {b['gained']:>+4}")


if __name__ == "__main__":
    main()
