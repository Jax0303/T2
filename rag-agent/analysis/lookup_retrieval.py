#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Retrieval accuracy on the 189 hitab_lookup queries, and what outranks gold.

Rank-based, not recall: R@k and MRR over the stored top-200 candidate list,
then the header relation of every cell ranked above the gold cell.

The corpus-wide board over all 830 lookup queries is
results/rank/STAGE1_BOARD.md; its dump kept counts, not cell identities, which
is why the relation table below is computed here instead. Runs no model.
Output: results/lookup_gap/retrieval.{json,md}."""
import json
import re
from collections import Counter
from pathlib import Path

SENT = re.compile(r"^In the table '(?P<t>.*)', (?:among (?P<row>.*), )?"
                  r"the value of (?P<col>.*) is (?P<v>.+?)\.\s*$")
SRC = "results/phase4/retrieval_{}/{}_hitab_lookup.jsonl"
KS = (1, 5, 10, 50, 200)


def parse(text):
    m = SENT.match(text)
    if not m:
        return None
    return {"row": (m["row"] or "").split(" > ") if m["row"] else [],
            "col": m["col"].split(" > ")}


def relation(gold, other, same_table):
    """Where a cell ranked above the gold cell sits relative to it."""
    if not same_table:
        return "other_table"
    if gold is None or other is None:
        return "unparsed"
    if gold["row"] == other["row"] and gold["col"] == other["col"]:
        return "same_address"
    if gold["row"] == other["row"]:
        return "same_row"
    if gold["col"] == other["col"]:
        return "same_col"
    if gold["row"][:-1] == other["row"][:-1] or gold["col"][:-1] == other["col"][:-1]:
        return "sibling_elsewhere"
    return "same_table_far"


def load(policy):
    return [json.loads(l) for part in ("294", "ext129")
            for l in open(SRC.format(part, policy))]


out, md = {}, [
    "# 조회 쿼리 검색 정확도 (Recall 아님)", "",
    "출처 `results/phase4/retrieval_{294,ext129}/*_hitab_lookup.jsonl` (후보 top-200).",
    "계측기 `analysis/lookup_retrieval.py`. 모델 실행 없음.",
    "830건 코퍼스 전체 랭킹 보드는 `results/rank/STAGE1_BOARD.md`.", ""]

for policy in ("P4_path_cell", "P1_fixed_512"):
    recs = load(policy)
    cell_ranks, per_query, above = [], [], Counter()
    for r in recs:
        gold = {tuple(g) for g in r["gold_cells"]}
        assert len(gold) == 1                       # lookup: one gold cell each
        gt, gi, gj = next(iter(gold))
        rk = [c["rank"] for c in r["topk"]
              for cc in c["cells"] if (c["table_id"], cc[0], cc[1]) in gold]
        gr = min(rk) if rk else None
        cell_ranks.append(gr)
        if policy != "P4_path_cell" or gr in (None, 0):
            continue
        gsent = next((parse(c["text"]) for c in r["topk"]
                      if c["table_id"] == gt and [gi, gj] in c["cells"]), None)
        rel = [(relation(gsent, parse(c["text"]), c["table_id"] == gt), c["used"])
               for c in r["topk"] if c["rank"] < gr]
        above.update(k for k, _ in rel)
        per_query.append({"query_id": r["query_id"], "gold_rank": gr,
                          "n_above": len(rel),
                          "n_above_used": sum(u for _, u in rel),
                          "gold_used": any(c["used"] for c in r["topk"]
                                           if c["table_id"] == gt and [gi, gj] in c["cells"]),
                          "relations": dict(Counter(k for k, _ in rel))})

    n = len(recs)
    hit = [x for x in cell_ranks if x is not None]
    stats = {"n": n,
             **{f"R@{k}": round(sum(x < k for x in hit) / n, 4) for k in KS},
             "MRR": round(sum(1 / (x + 1) for x in hit) / n, 4),
             "gold_in_top200": round(len(hit) / n, 4),
             "median_rank_when_found": sorted(hit)[len(hit) // 2],
             "chunks_entering_context_mean": round(
                 sum(r["n_chunks_used"] for r in recs) / n, 2)}
    if per_query:
        stats["n_queries_gold_not_rank0"] = len(per_query)
        stats["n_cells_above_gold"] = sum(p["n_above"] for p in per_query)
        stats["n_cells_above_gold_that_entered_context"] = sum(
            p["n_above_used"] for p in per_query)
    out[policy] = {"stats": stats, "above_gold": dict(above), "per_query": per_query}

    md += [f"## {policy}", "", "| 지표 | 값 |", "|---|---:|"]
    md += [f"| {k} | {v} |" for k, v in stats.items()]
    if above:
        tot = sum(above.values())
        md += ["", "gold 셀보다 위에 있던 셀의 관계 (전 쿼리 합산):", "",
               "| 관계 | 셀 수 | 비율 |", "|---|---:|---:|"]
        md += [f"| {k} | {v} | {v / tot:.4f} |" for k, v in above.most_common()]
    md.append("")

md += ["`R@k`의 분모는 쿼리 189건, 분자는 gold 셀이 후보 상위 k 안에 든 쿼리 수다.",
       "greedy fill 컨텍스트 안 포함률(= 지금까지 보고한 Recall)은 P4 0.9312 / P1 0.7989로",
       "따로이며, 후보 순위와 예산 절단은 다른 축이다.",
       "P1은 청크가 512토큰 블록이라 gold 셀 위 청크의 헤더 관계를 정의할 수 없다."]

d = Path("results/lookup_gap")
(d / "retrieval.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
(d / "retrieval.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
