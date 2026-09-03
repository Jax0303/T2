#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The two open items in RESULTS.md 2.1, for P4_path_cell bucket B.

(1) When the model took a value from a neighbouring cell, was the header it got
    wrong a row header or a column header, and was the cell a sibling?
(2) When the value it answered is in no chunk, is it a rescaling of the gold?

Reads results/lookup_gap/distractor.json; runs no model.
Output: results/lookup_gap/b_detail.{json,md}."""
import json
import re
from collections import Counter
from pathlib import Path

from phase4_summary import norm_em

# "In the table 'T', among ROW, the value of COL is V." -- `among ROW,` is absent
# when the cell has no row path.
SENT = re.compile(r"^In the table '(?P<t>.*)', (?:among (?P<row>.*), )?"
                  r"the value of (?P<col>.*) is (?P<v>.+?)\.\s*$")
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
SCALES = {1000: "x1000", 0.001: "/1000", 100: "x100", 0.01: "/100",
          -1: "sign", 1: "same_value"}


def parse(text):
    m = SENT.match(text)
    if not m:
        return None
    return {"table": m["t"],
            "row": (m["row"] or "").split(" > ") if m["row"] else [],
            "col": m["col"].split(" > "), "value": m["v"]}


def path_diff(g, t):
    """How the two header paths differ: equal / sibling (same parent, last
    segment differs) / prefix / other."""
    if g == t:
        return "equal"
    if len(g) == len(t) and g[:-1] == t[:-1]:
        return "sibling"
    if g[:len(t)] == t or t[:len(g)] == g:
        return "prefix"
    return "other"


def where_in_context(c):
    """Does the answered number appear anywhere in the context at all -- in a
    header path or a table title, not just as a cell's value?"""
    pn = norm_em(c["pred"])
    hits = {"value": 0, "header_or_title": 0}
    for ch in TOPK[c["query_id"]]:
        p = parse(ch["text"])
        if p is None:
            continue
        if norm_em(p["value"]) == pn:
            hits["value"] += 1
        elif any(pn == norm_em(x) for x in
                 NUM.findall(p["table"] + " " + " ".join(p["row"] + p["col"]))):
            hits["header_or_title"] += 1
    return {"pred_as_cell_value_in_context": hits["value"],
            "pred_in_header_or_title": hits["header_or_title"]}


def num(s):
    m = NUM.findall(str(s))
    return float(m[0].replace(",", "")) if m else None


cases = json.load(open("results/lookup_gap/distractor.json"))["P4_path_cell"]
# the context itself, to ask where an "absent" value could have come from: the
# distractor pass only matched a chunk's asserted value, not the header path or
# the table title it also prints.
TOPK = {r["query_id"]: json.loads(r["retrieved_topk"])
        for r in map(json.loads, open("results/phase4/reader_records.jsonl"))
        if r["pool"] == "hitab_lookup" and r["policy"] == "P4_path_cell"}
neighbour, absent = [], []

for c in cases:
    if c["origin"] == "not_in_context":
        g, p = num(c["gold_answer"]), num(c["pred"])
        scale = None
        if g not in (None, 0) and p is not None:
            r = p / g
            scale = next((n for k, n in SCALES.items() if abs(r - k) < 1e-6 * max(1, abs(k))), None)
            if scale is None and abs(r - round(r)) < 1e-9 and abs(r) > 1:
                scale = f"x{int(round(r))}"
        absent.append({
            "query_id": c["query_id"], "gold": c["gold_answer"], "pred": c["pred"],
            "ratio_pred_over_gold": None if g in (None, 0) or p is None else round(p / g, 6),
            "scale_of_gold": scale,
            "pred_in_query_text": bool(p is not None and norm_em(c["pred"]) in
                                       {norm_em(x) for x in NUM.findall(c["query"])}),
            "gold_chunk_text": c["gold_chunk_text"],
            "query": c["query"],
            **where_in_context(c),
        })
        continue
    if not c["matched"]:
        continue
    gold, took = parse(c["gold_chunk_text"] or ""), parse(c["matched"][0]["text"])
    if gold is None or took is None:
        neighbour.append({"query_id": c["query_id"], "origin": c["origin"],
                          "wrong_header": "UNPARSED"})
        continue
    rd, cd = path_diff(gold["row"], took["row"]), path_diff(gold["col"], took["col"])
    # identical row AND col path: either the chunk the model read IS the gold
    # chunk (so the gold answer and the indexed cell value disagree), or a second
    # cell in the same table carries the same address.
    same_chunk = c["matched"][0]["text"] == c["gold_chunk_text"]
    wrong = (("gold_cell_itself" if same_chunk else "duplicate_address")
             if rd == cd == "equal" else
             "col_header" if rd == "equal" else
             "row_header" if cd == "equal" else "both")
    neighbour.append({
        "query_id": c["query_id"], "origin": c["origin"], "wrong_header": wrong,
        "row_diff": rd, "col_diff": cd,
        "same_table": took["table"] == gold["table"],
        "gold_row": gold["row"], "took_row": took["row"],
        "gold_col": gold["col"], "took_col": took["col"],
        "rank_taken": c["matched"][0]["rank"], "gold_rank": c["gold_rank"],
        "gold": c["gold_answer"], "pred": c["pred"],
    })

out = {"n_neighbour": len(neighbour), "n_absent": len(absent),
       "neighbour": neighbour, "absent": absent}
d = Path("results/lookup_gap")
(d / "b_detail.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))

md = ["# hitab_lookup bucket B, 세부 (P4_path_cell)", "",
      "출처 `results/lookup_gap/distractor.json`. 계측기 `analysis/lookup_b_detail.py`.",
      "모델 재실행 없음.", "",
      f"## 1. 이웃 셀에서 값을 가져온 {len(neighbour)}건 — 틀린 헤더의 축", "",
      "| 틀린 헤더 | n |", "|---|---:|"]
md += [f"| {k} | {v} |" for k, v in Counter(x["wrong_header"] for x in neighbour).most_common()]
md += ["", "축별 경로 차이 (`equal` = 그 축은 정답과 동일):", "",
       "| 축 | equal | sibling | prefix | other |", "|---|---:|---:|---:|---:|"]
for ax in ("row_diff", "col_diff"):
    cnt = Counter(x.get(ax) for x in neighbour)
    md.append(f"| {ax[:3]} | " + " | ".join(str(cnt.get(k, 0)) for k in
                                            ("equal", "sibling", "prefix", "other")) + " |")
md += ["", "origin × 틀린 헤더:", "", "| origin | 틀린 헤더 | n |", "|---|---|---:|"]
md += [f"| {o} | {w} | {n} |" for (o, w), n in
       Counter((x["origin"], x["wrong_header"]) for x in neighbour).most_common()]

md += ["", f"## 2. 컨텍스트에 없는 값 {len(absent)}건", "",
       "| gold | pred | pred/gold | gold의 배율 | 질문 본문의 수 | 컨텍스트 헤더·제목의 수 |",
       "|---|---|---:|---|---|---:|"]
md += [f"| {a['gold']} | {a['pred']} | {a['ratio_pred_over_gold']} | "
       f"{a['scale_of_gold'] or '아니오'} | {'예' if a['pred_in_query_text'] else '아니오'} | "
       f"{a['pred_in_header_or_title']} |"
       for a in absent]
md += ["", "배율로 설명되는 건: "
       f"{sum(bool(a['scale_of_gold']) for a in absent)}/{len(absent)}. "
       "질문 본문의 수를 그대로 답한 건: "
       f"{sum(a['pred_in_query_text'] for a in absent)}/{len(absent)}. "
       "컨텍스트의 헤더·제목에 그 수가 있던 건: "
       f"{sum(bool(a['pred_in_header_or_title']) for a in absent)}/{len(absent)}. "
       "(셀 값으로는 정의상 0건 — 그래서 이 버킷이다.)"]
(d / "b_detail.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
