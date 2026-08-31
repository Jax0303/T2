#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""One workbook holding every hitab_lookup failure, for reading by hand.

Sheets:
  reader_failures    every query the reader got wrong, per policy, with the
                     bucket it falls in and the chunk the wrong value came from
  retrieval_189      the Phase 4 queries whose gold cell is not rank 0
  above_gold_189     one row per cell ranked above the gold cell (top-200 list)
  retrieval_830      the same over all 830 lookup queries, corpus-wide ranking
  above_gold_830     the first 20 cells above gold, per failing query

The 830 sheets need results/lookup_gap/rank830/*_above.jsonl
(analysis/cell_rank_dump.py --dump-above 20); they are skipped if absent.
Runs no model. Output: results/lookup_gap/failures.xlsx."""
import json
import re
from pathlib import Path

import pandas as pd

from phase4_summary import em

D = Path("results/lookup_gap")
SENT = re.compile(r"^In the table '(?P<t>.*)', (?:among (?P<row>.*), )?"
                  r"the value of (?P<col>.*) is (?P<v>.+?)\.\s*$")


def jload(p):
    return json.load(open(p))


def jlines(p):
    return [json.loads(l) for l in open(p)]


# ---------------------------------------------------------------- reader
rows = [r for r in jlines("results/phase4/reader_records.jsonl")
        if r["pool"] == "hitab_lookup"]
by = {(r["policy"], r["query_id"]): r for r in rows}
qids = sorted({r["query_id"] for r in rows})
ok = {k: em(r["pred_parsed"], r["gold_answer"]) for k, r in by.items()}

dist = {c["query_id"]: c for c in jload(D / "distractor.json")["P4_path_cell"]}
dist1 = {c["query_id"]: c for c in jload(D / "distractor.json")["P1_fixed_512"]}
axis = {x["query_id"]: x for x in jload(D / "b_detail.json")["neighbour"]}

reader = []
for pol in ("P1_fixed_512", "P4_path_cell", "gold_cell"):
    for q in qids:
        r = by[(pol, q)]
        if ok[(pol, q)]:
            continue
        d = (dist if pol == "P4_path_cell" else dist1).get(q, {})
        a = axis.get(q, {}) if pol == "P4_path_cell" else {}
        bucket = ("C_reader_limit" if pol == "gold_cell" else
                  "A_retrieval_miss" if not r["gold_all_in_topk"] else
                  "B_distractor" if ok[("gold_cell", q)] else "C_reader_limit")
        m = (d.get("matched") or [{}])[0]
        reader.append({
            "policy": pol, "query_id": q, "bucket": bucket,
            "query": r["query"], "gold_answer": r["gold_answer"],
            "pred": r["pred_parsed"],
            "gold_in_context": r["gold_all_in_topk"],
            "gold_rank": r["gold_rank"], "n_chunks_used": r["n_chunks_used"],
            "value_origin": d.get("origin", ""),
            "wrong_header_axis": a.get("wrong_header", ""),
            "row_path_diff": a.get("row_diff", ""), "col_path_diff": a.get("col_diff", ""),
            "gold_chunk_text": d.get("gold_chunk_text", ""),
            "chunk_the_value_came_from": m.get("text", ""),
            "its_rank": m.get("rank", ""),
        })

# ------------------------------------------------------- retrieval, n=189
def load_dump(policy):
    return [json.loads(l) for part in ("294", "ext129")
            for l in open(f"results/phase4/retrieval_{part}/{policy}_hitab_lookup.jsonl")]


ret189, above189 = [], []
for r in load_dump("P4_path_cell"):
    gt, gi, gj = r["gold_cells"][0]
    rk = [c["rank"] for c in r["topk"]
          for cc in c["cells"] if (c["table_id"], cc[0], cc[1]) == (gt, gi, gj)]
    gr = min(rk) if rk else None
    if gr == 0:
        continue
    gold_chunk = next((c for c in r["topk"]
                       if c["table_id"] == gt and [gi, gj] in c["cells"]), None)
    hi = [c for c in r["topk"] if gr is None or c["rank"] < gr]
    ret189.append({
        "query_id": r["query_id"], "query": r["question"],
        "gold_rank": "not in top200" if gr is None else gr,
        "gold_in_context": bool(gold_chunk and gold_chunk["used"]),
        "n_above": len(hi),
        "n_above_in_context": sum(c["used"] for c in hi),
        "reader_em": ok[("P4_path_cell", r["query_id"])],
        "gold_chunk_text": gold_chunk["text"] if gold_chunk else "",
    })
    for c in hi:
        above189.append({
            "query_id": r["query_id"], "rank": c["rank"],
            "in_context": c["used"], "same_table": c["table_id"] == gt,
            "table_id": c["table_id"], "text": c["text"],
        })

sheets = {"reader_failures": reader, "retrieval_189": ret189,
          "above_gold_189": above189}

# ------------------------------------------------------- retrieval, n=830
ab = sorted(D.glob("rank830/*_above.jsonl"))
if ab:
    recs = jlines(ab[0])
    r830 = [{"query_id": r["query_id"], "query": r["question"],
             "gold_table": r["gold_table"], "gold_row": r["gold_row"],
             "gold_col": r["gold_col"],
             "gold_row_path": " > ".join(r["gold_row_path"]),
             "gold_col_path": " > ".join(r["gold_col_path"]),
             "gold_value": r["gold_value"], "gold_rank": r["gold_rank"],
             "n_above": r["n_above"], "above_capped": r["above_capped"],
             **{f"above_{k}": v for k, v in sorted(r["relations"].items())}}
            for r in recs]
    a830 = [{"query_id": r["query_id"], "rank": t["rank"],
             "relation": t["relation"], "same_table": t["table_id"] == r["gold_table"],
             "table_id": t["table_id"], "row": t["row"], "col": t["col"],
             "row_path": " > ".join(t["row_path"]),
             "col_path": " > ".join(t["col_path"]), "value": t["value"]}
            for r in recs for t in r["top_above"]]
    sheets["retrieval_830"] = r830
    sheets["above_gold_830"] = a830

out = D / "failures.xlsx"
with pd.ExcelWriter(out, engine="openpyxl") as w:
    for name, data in sheets.items():
        df = pd.DataFrame(data)
        df.to_excel(w, sheet_name=name, index=False)
        ws = w.sheets[name]
        ws.freeze_panes = "A2"
        for col in ws.columns:                       # readable widths, capped
            width = max(len(str(c.value or "")) for c in col[:200])
            ws.column_dimensions[col[0].column_letter].width = min(max(width, 10), 70)
        print(f"{name:18s} {len(df):5d} rows x {len(df.columns)} cols")
print(f"-> {out}")
