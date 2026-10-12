"""틀린 답 중 LLM 답 = 입력의 정답 아닌 셀 값인 문항 수 (정답 셀 다 들어간 틀림 396)."""
import json, re, sys
from collections import Counter
sys.path[:0] = [".", "scripts"]
from answer_accuracy_mh import gold_cells_by_query, gold_context
rows = {r["query_id"]: r for r in map(json.loads, open("results/mh_arms/cap300_20260924/cell_uniq.jsonl"))}
recs = {r["query_id"]: r for r in map(json.loads, open("results/rerun_20260926/mh/mh_train_s3c_records.jsonl")) if r["query_id"] in rows}
gold, tables = gold_cells_by_query("train", "v3.3u", "none")
def norm(s):
    s = re.sub(r"[\s$,%]", "", str(s)).lower().strip(".")
    try: return float(s)
    except ValueError: return s
tally = Counter()
for q, r in rows.items():
    if r["answer_correct"] or not r["retrieval_correct"]: continue
    ctx = recs[q]["doc"]["context"]; gs = set(gold_context(gold[q], tables))
    val = lambda s: norm(s.rsplit(": ", 1)[-1])
    p = norm(r["pred"])
    other = [i + 1 for i, s in enumerate(ctx) if s not in gs and val(s) == p]
    ingold = any(val(s) == p for s in gs)
    k = "다른 셀 값" if other and not ingold else "정답 셀 값" if ingold else "입력에 없는 값"
    tally[(r["layer"], k)] += 1; tally[("ALL", k)] += 1
for k in sorted(tally): print(k, tally[k])
