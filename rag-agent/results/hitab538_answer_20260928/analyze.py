"""HiTab 538표 한 색인 조건 답변 300건 집계 (PREREG-2026-09-28-hitab538-answer.md, 실행 전 커밋).

실행:  PYTHONPATH=. .venv/bin/python results/hitab538_answer_20260928/analyze.py   (rag-agent/ 에서)
입력:  이 폴더의 s3c_rows.jsonl·chunk_rows.jsonl·table_top1_rows.jsonl,
       표 안 s3c 답변 results/s3c_answer_hitab300_20260926/rows.jsonl, 표 전체 답변 results/fulltable_20260924/hitab_rows.jsonl
출력:  이 폴더의 analyze.json
"""
import json
from pathlib import Path

from scripts.fair_filter_eval import mcnemar

HERE = Path(__file__).parent


def rows(path, arm):
    rs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return {r["query_id"]: r for r in rs if r["arm"] == arm}


arms = {"s3c": rows(HERE / "s3c_rows.jsonl", "s3c"), "chunk": rows(HERE / "chunk_rows.jsonl", "chunk"),
        "table_top1": rows(HERE / "table_top1_rows.jsonl", "table_top1")}
intable = rows("results/s3c_answer_hitab300_20260926/rows.jsonl", "s3c")
fulltable = rows("results/fulltable_20260924/hitab_rows.jsonl", "fulltable")
pop = json.load(open("results/ksweep_population_300.json"))["query_ids"]
assert all(set(d) == set(pop) for d in [*arms.values(), intable, fulltable]) and len(pop) == 300

ans = lambda d: {q: d[q]["correct_base"] for q in pop}
out = {"n": len(pop)}
for name, d in [*arms.items(), ("s3c_intable", intable)]:
    out[name] = {"answer_correct": sum(ans(d).values()),
                 "answer": round(sum(ans(d).values()) / len(pop), 4),
                 "retrieval_hits": sum(d[q]["retrieval_correct"] for q in pop),
                 "reader_input_tokens_mean": round(sum(d[q]["reader_input_tokens"] for q in pop) / len(pop), 1)}

# 확증 비교 2개: 정확 McNemar(양측), Holm. a = s3c
tests = {f"s3c_vs_{b}": mcnemar(ans(arms["s3c"]), ans(arms[b])) for b in ("chunk", "table_top1")}
order = sorted(tests, key=lambda k: tests[k]["p_value"])
running = 0.0
for i, k in enumerate(order):
    running = max(running, min(1.0, (len(order) - i) * tests[k]["p_value"]))
    tests[k]["p_holm"] = round(running, 6)
out["confirmatory"] = tests

# 기술 통계 (검정 없음): 표 안 s3c 237 과의 차이와 불일치 수
out["s3c_538_minus_intable"] = out["s3c"]["answer_correct"] - out["s3c_intable"]["answer_correct"]
pair = mcnemar(ans(arms["s3c"]), ans(intable))
out["s3c_538_vs_intable_counts"] = {"only_538": pair["a_only"], "only_intable": pair["b_only"]}

# 점검: 1위 표가 정답 표인 문항은 입력이 표 전체 조건과 같으므로 출력도 같아야 한다
hit = [q for q in pop if arms["table_top1"][q]["retrieval_correct"]]
out["check_table_top1_hit_same_as_fulltable"] = {
    "n_hit": len(hit),
    "n_input_tokens_differ": sum(arms["table_top1"][q]["reader_input_tokens"] != fulltable[q]["reader_input_tokens"] for q in hit),
    "n_pred_differ": sum(arms["table_top1"][q]["pred_base"] != fulltable[q]["pred_base"] for q in hit)}

(HERE / "analyze.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
