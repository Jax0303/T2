"""disc28.jsonl 집계 + 본 방법 200건 전체의 경로 충돌 기준선. 점수·설정은 바꾸지 않는다."""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path("/home/user/T2-1/rag-agent")
sys.path.insert(0, str(ROOT))
from rag_agent.eval.multihiertt_em import mh_exact_match, str_to_num  # noqa

S = Path(__file__).parent
R = [json.loads(l) for l in open(S / "disc28.jsonl", encoding="utf-8")]
out = {}


def ct(key):
    c = defaultdict(Counter)
    for r in R:
        c[key(r)][r["winner"]] += 1
    return {k: f"{v['ours']}:{v['chunk']}" for k, v in sorted(c.items(), key=lambda x: str(x[0]))}


out["check"] = {k: sum(r[k]["reconstruct_ok"] for r in R) for k in ("ours", "chunk")}
out["check_undetermined_texts"] = {r["query_id"][:8]: r["ours"]["undetermined_texts"] for r in R if r["ours"]["undetermined_texts"]}
out["check_failed"] = [r["query_id"][:8] for r in R if not (r["ours"]["reconstruct_ok"] and r["chunk"]["reconstruct_ok"])]
print("ours:chunk wins")
out["retrieval_combo"] = ct(lambda r: {(1, 1): "both", (1, 0): "ours_only", (0, 1): "chunk_only", (0, 0): "neither"}[
    (r["ours"]["retrieved_all_gold"], r["chunk"]["retrieved_all_gold"])])
out["layer"] = ct(lambda r: r["layer"])
out["ops"] = ct(lambda r: ",".join(r["ops"]) or "(lookup)")
out["n_gold_tables"] = ct(lambda r: r["n_gold_tables"])
# 답이 gold 셀 값 그 자체인가(아니면 헤더·계산값) — 데이터셋 answer 와 gold 셀 값을 공식 채점기로 대조
out["answer_is_gold_value"] = ct(lambda r: any(mh_exact_match(g["value"], r["answer"]) for g in r["gold_cells"]))


def stats(rs, k):
    x = [r[k] for r in rs]
    o = {"n": len(x), "n_tok_mean": round(mean(v["n_tok"] for v in x), 1), "n_tok_median": median(v["n_tok"] for v in x),
         "cells_mean": round(mean(v["cells"] for v in x), 1), "cells_median": median(v["cells"] for v in x),
         "units_mean": round(mean(v["units"] for v in x), 1),
         "cell_repeats_total": sum(v["cell_repeats"] for v in x),
         "dup_lines_cases": sum(v["dup_lines"] > 0 for v in x), "dup_lines_total": sum(v["dup_lines"] for v in x),
         "marker_missing": sum(not v["marker_found"] for v in x),
         "tables_in_ctx_mean": round(mean(v["tables_in_ctx"] for v in x), 2),
         "gold_row_share_mean": round(mean(v["gold_row_cells_share"] for v in x), 3),
         "gold_col_share_mean": round(mean(v["gold_col_cells_share"] for v in x), 3),
         "retrieved_all_gold": sum(v["retrieved_all_gold"] for v in x)}
    if k == "ours":
        o["path_collision_cases"] = sum(v["path_collision_lines"] > 0 for v in x)
        o["gold_path_collision_cases"] = sum(v["gold_path_collision"] > 0 for v in x)
    return o


for name, rs in (("all28", R), ("chunk_won19", [r for r in R if r["winner"] == "chunk"]),
                 ("ours_won9", [r for r in R if r["winner"] == "ours"])):
    out[name] = {k: stats(rs, k) for k in ("ours", "chunk")}

c19 = [r for r in R if r["winner"] == "chunk"]
out["chunk_won19_ours_all_gold"] = {
    "record_retrieval_correct": sum(r["ours"]["retrieved_all_gold"] for r in c19),
    "reconstructed_gold_in_ctx_eq_m": sum(r["ours"]["gold_in_ctx"] == r["m"] for r in c19),
    "missing": [(r["query_id"][:8], f"{r['ours']['gold_in_ctx']}/{r['m']}") for r in c19 if not r["ours"]["retrieved_all_gold"]],
    "ours_output_mentions_all_gold_values": sum(r["ours"]["gold_values_in_output"] == r["ours"]["gold_values_numeric"] for r in c19)}

# 기준선: 본 방법 200건 전체에서 경로 충돌(같은 경로가 문맥에 2번 이상)과 EM
H = ROOT / "results/mh_interim200"
ans = {x["query_id"]: x for x in map(json.loads, open(H / "mh_cell_hv2_answer_doc_qwen3_8b_cot_200.jsonl", encoding="utf-8"))}
chk = {x["query_id"]: x for x in map(json.loads, open(H / "mh_chunk_hv2_answer_doc_qwen3_8b_cot_200.jsonl", encoding="utf-8"))}
tab = defaultdict(lambda: [0, 0])
for line in open(ROOT / "results/mh_arms/mh_cell_hv2_records.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["query_id"] not in ans:
        continue
    paths = Counter(t.rsplit(": ", 1)[0] for t in r["doc"]["context"])
    col = any(n > 1 for n in paths.values())
    tab[col][0] += 1
    tab[col][1] += ans[r["query_id"]]["answer_correct"]
    tab[(col, "chunk")] = tab[(col, "chunk")]
    tab[f"{col}_chunk_em"] = [tab[f"{col}_chunk_em"][0] + 1, tab[f"{col}_chunk_em"][1] + chk[r["query_id"]]["answer_correct"]]
out["all200_ours_path_collision"] = {
    f"collision={c}": {"n": tab[c][0], "ours_em": round(tab[c][1] / tab[c][0], 4),
                       "chunk_em_same_ids": round(tab[f"{c}_chunk_em"][1] / tab[f"{c}_chunk_em"][0], 4)}
    for c in (True, False) if tab[c][0]}

# 9807e7e7: 데이터셋 정답 10 을 gold 셀 값으로 재계산
r = next(x for x in R if x["query_id"].startswith("9807e7e7"))
v = [abs(str_to_num(g["value"])) * (-1 if g["value"].strip().startswith("-") else 1) for g in r["gold_cells"]]
avg = sum(v) / len(v)
out["check_9807e7e7"] = {"values": v, "sum": sum(v), "mean": avg, "count_gt_mean": sum(x > avg for x in v),
                         "count_lt_mean": sum(x < avg for x in v), "dataset_answer": r["answer"]}
(S / "agg28.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
