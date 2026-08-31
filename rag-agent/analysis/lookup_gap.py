#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""hitab_lookup EM gap decomposition. Re-scores phase 4 reader records only;
runs no model. Output: results/lookup_gap/."""
import json
from collections import Counter
from pathlib import Path

from phase4_summary import em

ROWS = [json.loads(l) for l in open("results/phase4/reader_records.jsonl")]
LOOK = [r for r in ROWS if r["pool"] == "hitab_lookup"]
BY = {(r["policy"], r["query_id"]): r for r in LOOK}
QIDS = sorted({r["query_id"] for r in LOOK})
POLICIES = ["P1_fixed_512", "P4_path_cell", "gold_cell"]


def n_gold(r):
    return len(json.loads(r["gold_cell"]))


def bucket(r, ceil_ok):
    """A: gold cell absent from context. B: present, wrong here, right under
    gold_cell. C: wrong even under gold_cell."""
    if not r["gold_all_in_topk"]:
        return "A_retrieval_miss"
    return "B_distractor" if ceil_ok else "C_reader_limit"


out = {"n_queries": len(QIDS), "single_gold_cell": 0, "per_policy": {}}
single = {q for q in QIDS if n_gold(BY[("gold_cell", q)]) == 1}
out["single_gold_cell"] = len(single)

rows = []
# every hitab_lookup query has exactly one gold cell (asserted below), so
# there is no multi-cell subset to split out.
assert len(single) == len(QIDS)
for pol in POLICIES:
    for qs in (QIDS,):
        rec = [BY[(pol, q)] for q in qs]
        ok = {q: em(BY[(pol, q)]["pred_parsed"], BY[(pol, q)]["gold_answer"]) for q in qs}
        ceil = {q: em(BY[("gold_cell", q)]["pred_parsed"], BY[("gold_cell", q)]["gold_answer"]) for q in qs}
        b = Counter(bucket(BY[(pol, q)], ceil[q]) for q in qs if not ok[q])
        rows.append({
            "policy": pol, "n": len(qs),
            "em": round(sum(ok.values()) / len(qs), 4),
            "recall_all_gold_in_context": round(sum(r["gold_all_in_topk"] for r in rec) / len(rec), 4),
            "n_wrong": len(qs) - sum(ok.values()),
            **{k: b.get(k, 0) for k in ("A_retrieval_miss", "B_distractor", "C_reader_limit")},
        })
out["per_policy"] = rows

# queries the ceiling gets right but P4 misses, with the gold cell in context
p4_only = [q for q in QIDS
           if em(BY[("gold_cell", q)]["pred_parsed"], BY[("gold_cell", q)]["gold_answer"])
           and not em(BY[("P4_path_cell", q)]["pred_parsed"], BY[("P4_path_cell", q)]["gold_answer"])
           and BY[("P4_path_cell", q)]["gold_all_in_topk"]]
out["B_distractor_P4_queries"] = [{
    "query_id": q, "query": BY[("P4_path_cell", q)]["query"],
    "gold": BY[("P4_path_cell", q)]["gold_answer"],
    "pred_P4": BY[("P4_path_cell", q)]["pred_parsed"],
    "pred_gold_cell": BY[("gold_cell", q)]["pred_parsed"],
    "gold_rank": BY[("P4_path_cell", q)]["gold_rank"],
    "n_chunks_used": BY[("P4_path_cell", q)]["n_chunks_used"],
} for q in p4_only]

d = Path("results/lookup_gap")
d.mkdir(parents=True, exist_ok=True)
(d / "gap.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))

hdr = ["policy", "n", "em", "recall_all_gold_in_context", "n_wrong",
       "A_retrieval_miss", "B_distractor", "C_reader_limit"]
md = ["# hitab_lookup EM gap (re-scored from results/phase4/reader_records.jsonl)", "",
      f"queries {len(QIDS)}, single-gold-cell {len(single)}", "",
      "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
md += ["| " + " | ".join(str(r[h]) for h in hdr) + " |" for r in rows]
md += ["", f"B_distractor under P4_path_cell: {len(p4_only)} queries, listed in gap.json."]
(d / "gap.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
