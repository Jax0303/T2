#!/usr/bin/env python3
"""Baseline-vs-cell-ID comparison and the post-hoc selection/value error split.

Gold coordinates enter ONLY in the post-hoc section, never in the prediction.

  PYTHONPATH=. .venv/bin/python analysis/cell_id_report.py --arm p1
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.stats import binomtest
from rag_agent.serialization.cell_id import build_value_map, resolve, RULES
from rag_agent.eval.metrics import hitab_exact_match_text

OUT = ROOT / "results/cell_id_v2"
D = ROOT / "results/retrieval_accuracy"
BASELINE = D / "t_s3c_hybrid_answer_retrieved.jsonl"


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def groups_of(recs, ids):
    def sel(r, mode):
        agg = (r.get("aggregation") or "none")
        return ((mode == "lookup_single" and r["query_id"] in ids)
                or (mode == "lookup_multi" and r["mode"] == "all" and agg == "none" and r["m"] > 1)
                or (mode == "arithmetic" and r["mode"] == "all" and agg != "none")
                or (mode == "header_answer" and r["mode"] == "any")
                or (mode == "all_data" and r["mode"] == "all")
                or mode == "all_scored")
    return {m: [r["query_id"] for r in recs if sel(r, m)] for m in
            ("lookup_single", "lookup_multi", "arithmetic", "header_answer", "all_data", "all_scored")}


def cell_index(recs, page_titles):
    """frozen sentence -> the (table, row, col) coordinates that render it."""
    from rag_agent.bench.hitab_grid import load_table
    from rag_agent.serialization.templates import render, STRUCTURAL_COMPACT
    from rag_agent.serialization.caption import with_page_title
    idx = {}
    for tid in sorted({r["table_id"] for r in recs}):
        tab = load_table(tid, str(ROOT / "data/hitab"))
        t = tab.table
        title = with_page_title(tab.title, page_titles.get(tid))
        for i in range(t.n_rows):
            for j in range(t.n_cols):
                if str(t.data[i][j]).strip():
                    s = render(STRUCTURAL_COMPACT, title, t.row_path(i), t.col_path(j), t.data[i][j])
                    idx.setdefault(s, set()).add((tid, i, j))
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="p1")
    ap.add_argument("--rule", default="bracket")
    a = ap.parse_args()

    recs = [r for r in jl(D / "t_s3c_hybrid_records.jsonl") if "correct" in r]
    byq = {r["query_id"]: r for r in recs}
    ids = {r["query_id"] for r in recs
           if r["mode"] == "all" and (r.get("aggregation") or "none") == "none" and r["m"] == 1}
    base = {r["query_id"]: r for r in jl(BASELINE)}
    new = {r["query_id"]: r for r in jl(OUT / f"test_{a.arm}.jsonl")}
    assert set(new) == set(byq), "incomplete run"
    pred_key = f"pred__{a.rule}"
    ok_new = {q: int(hitab_exact_match_text(r[pred_key], r["answer"])) for q, r in new.items()}

    ks = sorted(ids)
    x = np.array([base[k]["answer_correct"] for k in ks], dtype=np.int8)
    y = np.array([ok_new[k] for k in ks], dtype=np.int8)
    gain, loss = int(((x == 0) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum())
    p = float(binomtest(min(gain, loss), gain + loss).pvalue) if gain + loss else 1.0
    d = (y - x).astype(float)
    rng = np.random.default_rng(42)
    ci = np.quantile(d[rng.integers(0, len(ks), (10000, len(ks)))].mean(axis=1), [.025, .975]).tolist()

    br = Counter(r[f"audit__{a.rule}"].get("failure") for r in new.values()
                 if r[f"audit__{a.rule}"].get("failure"))
    secs = [r["seconds"] for r in new.values()]

    rep = {"arm": a.arm, "rule": a.rule,
           "baseline_source": str(BASELINE.relative_to(ROOT)),
           "source": f"results/cell_id_v2/test_{a.arm}.jsonl",
           "n_primary": len(ks), "base_correct": int(x.sum()), "new_correct": int(y.sum()),
           "base_em": round(float(x.mean()), 4), "new_em": round(float(y.mean()), 4),
           "delta": round(float(d.mean()), 4), "gain": gain, "loss": loss,
           "net": gain - loss, "mcnemar_exact_p": p,
           "delta_ci95": [round(c, 4) for c in ci],
           "target_0_8_reached": bool(y.sum() >= 793),
           "supported": bool(d.mean() >= .02 and p < .05),
           "format_breakage": dict(br),
           "format_breakage_rate": round(sum(br.values()) / len(new), 4),
           "breakage_stop_rule_violated": bool(sum(br.values()) / len(new) > .05),
           "seconds_total": round(sum(secs), 1),
           "seconds_per_query": round(sum(secs) / len(secs), 3),
           "rule_sensitivity": {}}

    for rule in RULES:
        o = {q: int(hitab_exact_match_text(r[f"pred__{rule}"], r["answer"])) for q, r in new.items()}
        b2 = Counter(r[f"audit__{rule}"].get("failure") for r in new.values()
                     if r[f"audit__{rule}"].get("failure"))
        rep["rule_sensitivity"][rule] = {
            "primary_correct": sum(o[k] for k in ks),
            "primary_em": round(sum(o[k] for k in ks) / len(ks), 4),
            "breakage_rate": round(sum(b2.values()) / len(new), 4)}

    g = groups_of(recs, ids)
    rep["groups"] = {m: {"n": len(v), "base_correct": sum(base[k]["answer_correct"] for k in v),
                         "new_correct": sum(ok_new[k] for k in v)} for m, v in g.items()}

    # ---- post-hoc only: where does the device still lose? gold enters here.
    pt = json.loads((ROOT / "results/tableconf/totto_page_titles.json").read_text())
    vm = build_value_map(str(ROOT / "data/hitab"), {r["table_id"] for r in recs}, pt, "s3c")
    idx = cell_index(recs, pt)
    post = Counter()
    for k in ks:
        r, rec = new[k], byq[k]
        if ok_new[k]:
            post["correct"] += 1
            continue
        if not rec["correct"]:
            post["retrieval_failed"] += 1
            continue
        picked = r[f"audit__{a.rule}"].get("ids") or []
        if not picked:
            post["no_id_emitted"] += 1
            continue
        gold = {tuple([str(c[0]), c[1], c[2]]) for c in rec["gold_cells"]}
        chosen = set()
        unknown = False
        for i in picked:
            if not 1 <= i <= len(rec["context"]):
                unknown = True
                continue
            c = idx.get(rec["context"][i - 1])
            if c is None:
                unknown = True
            else:
                chosen |= c
        if chosen & gold:
            post["value_return_error"] += 1          # right cell, value did not score
        elif unknown and not chosen:
            post["coordinates_unavailable"] += 1
        else:
            post["cell_selection_error"] += 1
    # how many of the value_return_error cells hold a value that could never score
    unreachable = sum(1 for k in ks if byq[k]["correct"] and not any(
        hitab_exact_match_text(resolve(l, vm), byq[k]["answer"]) for l in byq[k]["context"]))
    rep["posthoc_primary"] = {**post, "retrieval_correct_but_no_scoring_value": unreachable,
                              "note": "gold coordinates used here only, never in prediction"}
    (OUT / f"REPORT_{a.arm}_{a.rule}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
