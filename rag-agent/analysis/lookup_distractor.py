#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Where did the wrong lookup answer come from? For each hitab_lookup query
that had the gold cell in context and still missed, match the predicted value
against the value asserted by every chunk in that same context.

Re-scores phase 4 records only; runs no model. Output: results/lookup_gap/."""
import json
import re
from collections import Counter
from pathlib import Path

from phase4_summary import em, norm_em

VAL = re.compile(r"\bis\s+(.+?)\s*\.\s*$")          # cell sentence: "... is X."
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# P4 chunks are one-cell sentences, so the value a chunk asserts is unambiguous
# and a match can be attributed to a row/column. A P1 chunk is a 512-token block
# of many cells: the only answerable question there is whether the predicted
# value occurs in the context at all, so P1 gets the coarse check and no
# row/column attribution.

ROWS = [json.loads(l) for l in open("results/phase4/reader_records.jsonl")]
LOOK = [r for r in ROWS if r["pool"] == "hitab_lookup"]
BY = {(r["policy"], r["query_id"]): r for r in LOOK}
QIDS = sorted({r["query_id"] for r in LOOK})


def chunk_values(c, cell_sentences):
    if cell_sentences:
        m = VAL.search(c["text"])
        return [m.group(1)] if m else []
    return NUM.findall(c["text"])


def report(policy):
    cell_sentences = policy != "P1_fixed_512"
    ceil_ok = {q: em(BY[("gold_cell", q)]["pred_parsed"], BY[("gold_cell", q)]["gold_answer"]) for q in QIDS}
    cases = []
    for q in QIDS:
        r = BY[(policy, q)]
        if em(r["pred_parsed"], r["gold_answer"]) or not r["gold_all_in_topk"] or not ceil_ok[q]:
            continue                                 # not a bucket-B miss
        topk = json.loads(r["retrieved_topk"])
        gtab, grow, gcol = json.loads(r["gold_cell"])[0]
        pn = norm_em(r["pred_parsed"])
        hits = [c for c in topk if any(norm_em(v) == pn for v in chunk_values(c, cell_sentences))]
        gold_chunks = [c for c in topk if c["table_id"] == gtab and [grow, gcol] in c["cells"]]
        gv = (chunk_values(gold_chunks[0], cell_sentences) or [None])[0] if gold_chunks else None
        cases.append({
            "query_id": q,
            "query": r["query"],
            "gold_answer": r["gold_answer"],
            "pred": r["pred_parsed"],
            "gold_rank": r["gold_rank"],
            "n_chunks_used": r["n_chunks_used"],
            "gold_chunk_text": gold_chunks[0]["text"] if gold_chunks else None,
            "gold_chunk_value": gv if cell_sentences else None,
            "gold_chunk_carries_answer": (bool(gv is not None and em(gv, r["gold_answer"]))
                                          if cell_sentences else None),
            "n_context_chunks_matching_pred": len(hits),
            "matched": [{
                "rank": c["rank"],
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "same_table": c["table_id"] == gtab,
                "same_row": c["table_id"] == gtab and any(rc == grow for rc, _ in c["cells"]),
                "same_col": c["table_id"] == gtab and any(cc == gcol for _, cc in c["cells"]),
            } for c in hits[:5]],
        })

    def cls(c):
        if not cell_sentences:
            return "in_context" if c["n_context_chunks_matching_pred"] else "not_in_context"
        if c["n_context_chunks_matching_pred"] == 0:
            return "not_in_context"                  # value the model made up / derived
        m = c["matched"][0]
        if not m["same_table"]:
            return "other_table"
        if m["same_row"]:
            return "same_row_wrong_col"
        if m["same_col"]:
            return "same_col_wrong_row"
        return "same_table_elsewhere"

    for c in cases:
        c["origin"] = cls(c)
    return cases, cell_sentences


out, md = {}, ["# hitab_lookup: origin of the wrong value (bucket B)", ""]
for pol in ("P4_path_cell", "P1_fixed_512"):
    cases, cell_sentences = report(pol)
    out[pol] = cases
    origin = Counter(c["origin"] for c in cases)
    gold_bad = (sum(not c["gold_chunk_carries_answer"] for c in cases) if cell_sentences
                else "N/A (512-token block, no single asserted value)")
    md += [f"## {pol} — n={len(cases)}", "",
           "| origin | n |", "|---|---:|"]
    md += [f"| {k} | {v} |" for k, v in origin.most_common()]
    md += ["",
           f"gold_rank == 0: {sum(c['gold_rank'] == 0 for c in cases)}",
           f"gold chunk in context whose sentence does NOT carry the gold answer: {gold_bad}",
           ("" if cell_sentences else
            "row/column attribution is N/A for P1: a matched block holds many cells."),
           ""]

d = Path("results/lookup_gap")
d.mkdir(parents=True, exist_ok=True)
(d / "distractor.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
(d / "distractor.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
