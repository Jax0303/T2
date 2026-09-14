#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""top-m 정확 일치 실패 원인 분류 — dev 질의만 (results/opt/split.json). test 질의는 분류하지 않는다.

순위: results/evaluation_v2/s3c_v2_records.jsonl 의 context_units 순서(검색 순위, 기록 깊이 20).
분류는 데이터만 쓰는 기계 규칙이고 사람 판정이 아니다.

  PYTHONPATH=. .venv/bin/python analysis/topm_failure_diag.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rag_agent.bench import hitab_grid as hg                       # noqa: E402
from rag_agent.eval.artifacts import file_digest, read_records     # noqa: E402
from rag_agent.eval.metrics import hitab_exact_match_text          # noqa: E402
from scripts.retrieval_accuracy import TYPES, query_type           # noqa: E402

REC = ROOT / "results/evaluation_v2/s3c_v2_records.jsonl"
SPLIT = ROOT / "results/opt/split.json"
OUT = ROOT / "results/opt/diag/topm_failures_dev.json"
STOP = {"the", "of", "in", "and", "a", "an", "to", "for", "by", "on", "at", "or", "with", "from", "is", "was",
        "what", "how", "many", "much", "which", "were", "are", "did", "do", "that", "this"}
TABS: dict = {}


def toks(s):
    # ponytail: 복수형 s 제거만 하는 어간 처리, 불규칙 변화는 못 잡는다
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w
            for w in re.findall(r"[a-z0-9]+", str(s).lower())} - STOP


def cell(c):
    tid, i, j = c
    if tid not in TABS:
        TABS[tid] = hg.load_table(tid, str(ROOT / "data/hitab"))
    tab = TABS[tid]
    t = tab.table
    return {"tid": tid, "i": i, "j": j, "row": list(t.row_path(i)), "col": list(t.col_path(j)),
            "value": t.data[i][j], "title": tab.title}


def prefix(a, b):
    return len(a) <= len(b) and b[:len(a)] == a


def single_failure(r):
    g, top = cell(r["gold_cells"][0]), cell(r["context_units"][0]["cells"][0])
    rank = r["gold_rank"]
    if top["tid"] != g["tid"]:
        where = "other_table_same_title" if top["title"] == g["title"] else "other_table"
        hier = "other_table"
    else:
        where = {(True, False): "same_row", (False, True): "same_col"}.get(
            (top["i"] == g["i"], top["j"] == g["j"]), "diff_row_and_col")
        below = prefix(g["row"], top["row"]) and prefix(g["col"], top["col"])
        above = prefix(top["row"], g["row"]) and prefix(top["col"], g["col"])
        hier = "top1_below_gold" if below else "top1_above_gold" if above else "not_nested"
    gl = set(g["row"] + g["col"] + [g["title"]])
    tl = set(top["row"] + top["col"] + [top["title"]])
    dg = toks(" ".join(gl - tl)) - toks(" ".join(tl - gl))
    dt = toks(" ".join(tl - gl)) - toks(" ".join(gl - tl))
    q = toks(r["question"])
    lex = {(True, False): "question_names_gold_side", (False, True): "question_names_top1_side",
           (True, True): "question_names_both"}.get((bool(q & dg), bool(q & dt)), "question_names_neither")
    return {"query_id": r["query_id"], "question": r["question"], "answer": r["answer"],
            "gold_rank": "2" if rank == 2 else "3-5" if rank and rank <= 5 else "6-20" if rank and rank <= 20
                         else "21-500" if rank else ">500",
            "where": where, "hierarchy": hier, "lexical": lex,
            "gold_value_matches_answer": hitab_exact_match_text(str(g["value"]), r["answer"]),
            "top1_value_matches_answer": hitab_exact_match_text(str(top["value"]), r["answer"]),
            "gold_labels": g["row"] + g["col"], "top1_labels": top["row"] + top["col"],
            "gold_distinct_words": sorted(dg), "top1_distinct_words": sorted(dt)}


def multi_failure(r, t):
    gold = {tuple(c) for c in r["gold_cells"]}
    ranked = [tuple(u["cells"][0]) for u in r["context_units"]]
    intr = [c for c in ranked[:r["m"]] if c not in gold]
    return {"query_id": r["query_id"], "type": t, "m": r["m"], "aggregation": r["aggregation"],
            "gold_in_top_m": len(gold & set(ranked[:r["m"]])),
            "intruders_same_table": sum(c[0] == r["table_id"] for c in intr),
            "intruders_other_table": sum(c[0] != r["table_id"] for c in intr),
            "all_gold_within_recorded_depth": gold <= set(ranked)}


def main():
    split = json.loads(SPLIT.read_text())
    dev = {q["query_id"] for q in split["dev_queries"]}
    assert json.loads(REC.with_name("s3c_v2.json").read_text())["records_sha256"] == file_digest(REC)
    recs = read_records(REC)
    acc = {t: [0, 0] for t in TYPES}
    singles, multis, label_mismatch = [], [], 0
    for qid in sorted(dev):
        r = recs[qid]
        if "correct" not in r:
            continue
        t = query_type({"mode": r["mode"], "aggregation": r["aggregation"], "gold": r["gold_cells"]})
        if t not in TYPES:
            continue
        units = r["context_units"]
        assert all(len(u["cells"]) == 1 for u in units)
        ok = {tuple(u["cells"][0]) for u in units[:r["m"]]} == {tuple(c) for c in r["gold_cells"]}
        acc[t][0] += 1
        acc[t][1] += ok
        if t == "single_cell":
            label_mismatch += not hitab_exact_match_text(str(cell(r["gold_cells"][0])["value"]), r["answer"])
            if not ok:
                singles.append(single_failure(r))
        elif not ok:
            multis.append(multi_failure(r, t))
    bt = split["by_type"]["dev"]
    assert (acc["single_cell"][0], acc["multi_cell"][0], acc["arithmetic"][0]) == (bt["single"], bt["multi"], bt["arith"])

    keys = ("gold_rank", "where", "hierarchy", "lexical", "gold_value_matches_answer", "top1_value_matches_answer")
    result = {
        "population": "dev queries of results/opt/split.json only",
        "source": {"records": str(REC.relative_to(ROOT)), "records_sha256": file_digest(REC),
                   "split_sha256": file_digest(SPLIT)},
        "top_m_exact_dev": {t: {"n": n, "success": s, "accuracy": round(s / n, 4)} for t, (n, s) in acc.items()},
        "single_all_dev_gold_value_not_matching_answer": label_mismatch,
        "single_failures": {"n": len(singles), **{k: dict(Counter(str(x[k]) for x in singles)) for k in keys},
                            "where_x_lexical": dict(Counter(f"{x['where']} | {x['lexical']}" for x in singles)),
                            "rows": singles},
        "multi_arith_failures": {"n": len(multis),
                                 "type_m_gold_in_top_m": dict(Counter(f"{x['type']} m={x['m']} hit={x['gold_in_top_m']}"
                                                                      for x in multis)),
                                 "with_same_table_intruder": sum(x["intruders_same_table"] > 0 for x in multis),
                                 "with_other_table_intruder": sum(x["intruders_other_table"] > 0 for x in multis),
                                 "all_gold_within_recorded_depth": sum(x["all_gold_within_recorded_depth"] for x in multis),
                                 "rows": multis},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    printable = {k: v for k, v in result.items() if k not in ("single_failures", "multi_arith_failures")}
    printable["single_failures"] = {k: v for k, v in result["single_failures"].items() if k != "rows"}
    printable["multi_arith_failures"] = {k: v for k, v in result["multi_arith_failures"].items() if k != "rows"}
    print(json.dumps(printable, ensure_ascii=False, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
