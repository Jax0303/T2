"""필터 후 근거 지표(B)와 답변(C) 집계 (PREREG-2026-09-28-fair-mh.md).

실행:  .venv/bin/python results/fair_mh_20260928/filter_metrics.py   (rag-agent/ 에서)
입력:  filter/<arm>_filtered_records.jsonl, answer/<arm>_filtered_answer.jsonl, 필터 없음 답변(표 5-8 원천), answer/gold_answer.jsonl
출력:  filter_metrics.json, filter_metrics.md
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).parent
NOFILTER = {"s3c": "results/mh_arms/cap300_20260924/cell_uniq.jsonl",
            "chunk": "results/reader_format_20260927/test/mh_chunk_final.jsonl",
            "trag_hetero": "results/mh_arms/cap300_20260924/trag_hetero.jsonl"}


def load(p):
    return {r["query_id"]: r for r in map(json.loads, open(p, encoding="utf-8"))} if Path(p).exists() else None


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def r4(x):
    return round(x, 4)


out, md = {}, ["# 필터 후 근거 지표(B)와 답변(C), MultiHiertt 882\n"]
answers = {}
for arm in ("s3c", "chunk", "trag_hetero"):
    fr = load(HERE / "filter" / f"{arm}_filtered_records.jsonl")
    if fr is None:
        out[arm] = "미실행"; continue
    rows = list(fr.values())
    f = [r["doc"] for r in rows]
    res = {"n": len(rows), "n_filter_empty": sum(x["filter"]["filter_empty"] for x in f),
           "n_queries_with_invalid_ids": sum(x["filter"]["n_invalid_ids"] > 0 for x in f),
           "lines_mean": r4(sum(x["filter"]["n_lines"] for x in f) / len(f)),
           "kept_lines_mean": r4(sum(x["filter"]["n_kept"] for x in f) / len(f)),
           "pre_filter_all_evidence": r4(sum(x["pre_filter"]["correct"] for x in f) / len(f)),
           "post_filter_all_evidence": r4(sum(x["correct"] for x in f) / len(f)),
           "correct_is_proxy": f[0]["correct_is_proxy"]}
    if arm == "s3c":   # 셀 좌표가 있어 정확
        rec, prec, f1, rem_gold, rem_noise, sel = [], [], [], 0, 0, []
        for r in rows:
            d, gold = r["doc"], set(r["gold_ids"])
            pre = {c for u in d["ranked_units"][:len(d["pre_filter"]) and 20] for c in u}
            pre = {c for u in d["ranked_units"] for c in u}
            kept = set(d["kept_cells"]); inter = len(gold & kept)
            R = inter / len(gold); P = inter / len(kept) if kept else 0.0
            rec.append(R); prec.append(P); f1.append(2 * P * R / (P + R) if P + R else 0.0); sel.append(len(kept))
            rem_gold += len((gold & pre) - kept); rem_noise += len((pre - gold) - kept)
        res |= {"recall_mean": r4(sum(rec) / len(rec)), "precision_mean(분모=선택 셀 수, 빈 선택=0)": r4(sum(prec) / len(prec)),
                "f1_mean": r4(sum(f1) / len(f1)), "selected_cells_mean": r4(sum(sel) / len(sel)),
                "removed_gold_cells_total": rem_gold, "removed_nongold_cells_total": rem_noise,
                "pre_filter_gold_cells_in_context_total": sum(len(set(r["gold_ids"]) & {c for u in r["doc"]["ranked_units"] for c in u}) for r in rows)}
    else:
        res |= {"note": "줄 = 표의 행(셀 여러 개). 셀 단위 Recall/Precision 없음; All-Evidence 는 정답 값 문자열 잔존 대리 지표"}
    ans = load(HERE / "answer" / f"{arm}_filtered_answer.jsonl")
    nof = load(NOFILTER[arm])
    if ans is not None:
        answers[arm] = ans
        res["answer_filtered"] = {"correct": sum(x["answer_correct"] for x in ans.values()), "n": len(ans),
                                  "input_tokens_mean": r4(sum(x["n_tok"] for x in ans.values()) / len(ans)),
                                  "by_layer": {L: sum(x["answer_correct"] for x in ans.values() if x["layer"] == L) for L in sorted({x["layer"] for x in ans.values()})}}
        if nof is not None:
            ids = sorted(set(ans) & set(nof))
            b = sum(nof[q]["answer_correct"] and not ans[q]["answer_correct"] for q in ids)
            c = sum(ans[q]["answer_correct"] and not nof[q]["answer_correct"] for q in ids)
            res["answer_nofilter"] = {"correct": sum(nof[q]["answer_correct"] for q in ids), "n": len(ids), "source": NOFILTER[arm]}
            res["nofilter_vs_filtered"] = {"nofilter_only": b, "filtered_only": c, "p": mcnemar(b, c)}
    out[arm] = res
if "s3c" in answers and "chunk" in answers:
    a, b_ = answers["s3c"], answers["chunk"]
    ids = sorted(set(a) & set(b_))
    x = sum(a[q]["answer_correct"] and not b_[q]["answer_correct"] for q in ids)
    y = sum(b_[q]["answer_correct"] and not a[q]["answer_correct"] for q in ids)
    out["s3c_filtered_vs_chunk_filtered"] = {"s3c_only": x, "chunk_only": y, "p": mcnemar(x, y)}
    # 주 비교 2개 Holm
    ps = {"s3c_filtered_vs_chunk_filtered": out["s3c_filtered_vs_chunk_filtered"]["p"],
          "s3c_nofilter_vs_filtered": out["s3c"]["nofilter_vs_filtered"]["p"]}
    order = sorted(ps, key=ps.get); run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (2 - i) * ps[k])); out.setdefault("holm", {})[k] = round(run, 6)
g = load(HERE / "answer" / "gold_answer.jsonl")
if g is not None:
    out["gold_condition_answer"] = {"correct": sum(x["answer_correct"] for x in g.values()), "n": len(g),
                                    "by_layer": {L: sum(x["answer_correct"] for x in g.values() if x["layer"] == L) for L in sorted({x["layer"] for x in g.values()})}}
# 문서 군집 부트스트랩(10,000회, seed 0): 같은 문서(표 내용 해시)의 문항을 한 군집으로 복원 추출, 정답 수 차이의 95% 구간
cl = json.load(open(HERE / "doc_clusters.json"))["cluster_of_query_882"]


def cluster_boot(a, b, n=10000, seed=0):
    import random
    rng = random.Random(seed)
    groups = {}
    for q in a:
        groups.setdefault(cl[q], []).append(a[q]["answer_correct"] - b[q]["answer_correct"])
    G = list(groups.values()); diffs = []
    for _ in range(n):
        diffs.append(sum(sum(groups_i) for groups_i in (G[rng.randrange(len(G))] for _ in G)))
    diffs.sort()
    return {"n_clusters": len(G), "diff": sum(map(sum, G)), "ci95": [diffs[int(.025 * n)], diffs[int(.975 * n) - 1]]}


if "s3c" in answers and "chunk" in answers:
    out["s3c_filtered_vs_chunk_filtered"]["cluster_bootstrap"] = cluster_boot(answers["s3c"], answers["chunk"])
if "s3c" in answers and load(NOFILTER["s3c"]) is not None:
    nof = load(NOFILTER["s3c"]); a_ = {q: answers["s3c"][q] for q in answers["s3c"] if q in nof}
    out["s3c"]["nofilter_vs_filtered"]["cluster_bootstrap(filtered − nofilter)"] = cluster_boot(a_, {q: nof[q] for q in a_})
(HERE / "filter_metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
