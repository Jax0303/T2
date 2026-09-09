#!/usr/bin/env python3
"""Evaluate the value-snap stage. No generation: it re-scores existing predictions.

  PYTHONPATH=. .venv/bin/python analysis/value_snap_eval.py dev
  PYTHONPATH=. .venv/bin/python analysis/value_snap_eval.py test --rule S3
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.stats import binomtest
from rag_agent.eval.metrics import hitab_exact_match_text
from rag_agent.serialization.cell_id import build_value_map, resolve
from rag_agent.serialization.value_snap import snap, SETS

OUT = ROOT / "results/value_snap_v1"
D = ROOT / "results/retrieval_accuracy"
PREREG = ROOT / "PREREG-2026-09-10-value-snap.md"


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(split):
    """(rows keyed by qid, each with pred/answer/context, populations)."""
    if split == "test":
        recs = {r["query_id"]: r for r in jl(D / "t_s3c_hybrid_records.jsonl") if "correct" in r}
        preds = {r["query_id"]: r for r in jl(D / "t_s3c_hybrid_answer_retrieved.jsonl")}
        rows = {q: {"pred": preds[q]["pred"], "answer": preds[q]["answer"],
                    "context": recs[q]["context"], "table_id": recs[q]["table_id"],
                    "mode": recs[q]["mode"], "aggregation": recs[q].get("aggregation"),
                    "m": recs[q]["m"]} for q in recs if q in preds}
        src = [D / "t_s3c_hybrid_records.jsonl", D / "t_s3c_hybrid_answer_retrieved.jsonl"]
    else:
        f = ROOT / "results/cell_id_v2/dev_base_n700.jsonl"
        rows = {r["query_id"]: {"pred": r["pred"], "answer": r["answer"],
                                "context": r["context"], "table_id": r["table_id"],
                                "mode": r["mode"], "aggregation": r.get("aggregation"),
                                "m": r["m"]} for r in jl(f)}
        src = [f]
    pops = {
        "primary": [q for q, r in rows.items() if r["mode"] == "all"
                    and (r.get("aggregation") or "none") == "none" and r["m"] == 1],
        "data": [q for q, r in rows.items() if r["mode"] == "all"],
        "header": [q for q, r in rows.items() if r["mode"] == "any"],
        "all": list(rows),
    }
    return rows, pops, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["dev", "test"])
    ap.add_argument("--rule", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, pops, src = load(a.split)
    pt = json.loads((ROOT / "results/tableconf/totto_page_titles.json").read_text())
    vm = build_value_map(str(ROOT / "data/hitab"), {r["table_id"] for r in rows.values()}, pt, "s3c")
    values = {q: [resolve(l, vm) for l in r["context"]] for q, r in rows.items()}
    base = {q: int(bool(hitab_exact_match_text(r["pred"], r["answer"]))) for q, r in rows.items()}

    rules = [a.rule] if a.rule else list(SETS)
    rng = np.random.default_rng(42)
    report = {"split": a.split, "prereg_sha256": sha(PREREG),
              "sources": [str(p.relative_to(ROOT)) for p in src],
              "source_sha256": [sha(p) for p in src],
              "n": len(rows), "n_primary": len(pops["primary"]), "rules": {}}
    for rule in rules:
        new, audits = {}, Counter()
        for q, r in rows.items():
            p, au = snap(r["pred"], values[q], rule)
            new[q] = int(bool(hitab_exact_match_text(p, r["answer"])))
            audits[au.get("kind") if au["snapped"] else "no_snap:" + au["reason"]] += 1
        e = {"snap_audit": dict(audits.most_common())}
        for pop, ids in pops.items():
            ks = sorted(ids)
            x = np.array([base[q] for q in ks], np.int8)
            y = np.array([new[q] for q in ks], np.int8)
            gain, loss = int(((x == 0) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum())
            d = (y - x).astype(float)
            idx = rng.integers(0, len(ks), (10000, len(ks)))
            e[pop] = {"n": len(ks), "before": int(x.sum()), "after": int(y.sum()),
                      "em_before": round(float(x.mean()), 4), "em_after": round(float(y.mean()), 4),
                      "delta": round(float(d.mean()), 4), "recovered": gain, "broken": loss,
                      "net": gain - loss,
                      "mcnemar_p": float(binomtest(min(gain, loss), gain + loss).pvalue) if gain + loss else 1.0,
                      "ci95": [round(c, 4) for c in np.quantile(d[idx].mean(axis=1), [.025, .975]).tolist()]}
        n_snap = sum(v for k, v in audits.items() if not k.startswith("no_snap"))
        e["n_snapped"] = n_snap
        e["snap_rate"] = round(n_snap / len(rows), 4)
        report["rules"][rule] = e
    (OUT / f"SNAP_{a.split}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"{a.split}: n={len(rows)} primary={len(pops['primary'])}")
    print(f"{'rule':5} {'주지표 before':>13} {'after':>8} {'Δ':>8} {'회수':>5} {'파괴':>5} {'스냅건수':>7} {'p':>9}")
    for rule, e in report["rules"].items():
        p = e["primary"]
        print(f"{rule:5} {str(p['before'])+'/'+str(p['n']):>13} {p['after']:>8} {p['delta']:>+8.4f} "
              f"{p['recovered']:>5} {p['broken']:>5} {e['n_snapped']:>7} {p['mcnemar_p']:>9.4g}")


if __name__ == "__main__":
    main()
