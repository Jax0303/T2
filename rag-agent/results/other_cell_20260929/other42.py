"""정답 셀이 입력에 다 있는데 LLM 답 = 정답 아닌 셀 값(정답 셀 값과는 다름)인 문항을 뽑아 저장."""
import json, re, sys
sys.path[:0] = [".", "scripts"]
from answer_accuracy_mh import gold_cells_by_query, gold_context
rows = {r["query_id"]: r for r in map(json.loads, open("results/mh_arms/cap300_20260924/cell_uniq.jsonl"))}
recs = {r["query_id"]: r for r in map(json.loads, open("results/rerun_20260926/mh/mh_train_s3c_records.jsonl")) if r["query_id"] in rows}
gold, tables = gold_cells_by_query("train", "v3.3u", "none")
def norm(s):
    s = re.sub(r"[\s$,%]", "", str(s)).lower().strip(".")
    try: return float(s)
    except ValueError: return s
val = lambda s: norm(s.rsplit(": ", 1)[-1])
out = []
for q in sorted(rows):
    r = rows[q]
    if r["answer_correct"] or not r["retrieval_correct"]: continue
    ctx = recs[q]["doc"]["context"]; gs = gold_context(gold[q], tables)
    p = norm(r["pred"])
    other = [(i + 1, s) for i, s in enumerate(ctx) if s not in gs and val(s) == p]
    if other and not any(val(s) == p for s in gs):
        out.append({"query_id": q, "layer": r["layer"], "question": r["question"], "answer": str(r["answer"]),
                    "pred": r["pred"], "gold": [(ctx.index(s) + 1, s) for s in gs], "chosen": other,
                    "raw": r["raw"]})
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print(len(out))
for n, e in enumerate(out, 1):
    print(f"\n#{n} {e['layer']} | Q: {e['question']}\n   ANS {e['answer']} | PRED {e['pred']}")
    for k, s in e["gold"]: print(f"   G{k:>3} {s}")
    for k, s in e["chosen"]: print(f"   X{k:>3} {s}")
