"""PREREG-2026-09-28-mh-name-append.md 집계.

실행:  .venv/bin/python results/mh_name_append_20260928/analyze.py   (rag-agent/ 에서)
입력:  이 폴더의 {s3c,chunk,leaf}_{base,append}_records.jsonl, groups.json,
       재현 점검용 기존 기록 (rerun_20260926 s3c·chunk, tablerag_official_20260928 leaf)
출력:  이 폴더의 analyze.json
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).parent
ARMS = {"s3c": "results/rerun_20260926/mh/mh_train_s3c_records.jsonl",
        "chunk": "results/rerun_20260926/mh/mh_train_chunk_records.jsonl",
        "leaf": "results/tablerag_official_20260928/mh/mh_train_tablerag_leaf_official_records.jsonl"}
GROUPS = ("값 비교 조회", "그 밖의 조회", "산술")
groups = json.loads((HERE / "groups.json").read_text())


def load(path):
    return {r["query_id"]: r for r in map(json.loads, open(path, encoding="utf-8")) if "excluded" not in r}


def mcnemar(b, c):   # 정확 이항, 양측
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def ok(r, k):
    return r["doc"]["correct"] if k == 20 else r["correct_at"]["doc"][str(k)]


out = {"n": {g: sum(v == g for v in groups.values()) for g in GROUPS}, "arms": {}}
primary = {}
for arm, old_path in ARMS.items():
    base, app = load(HERE / f"{arm}_base_records.jsonl"), load(HERE / f"{arm}_append_records.jsonl")
    old = load(old_path)
    assert set(base) == set(app) == set(groups)
    res = {"base_vs_existing_records_differ": sum(base[q]["doc"]["correct"] != old[q]["doc"]["correct"] for q in groups)}
    for k in (20, 1):
        for g in GROUPS:
            qs = [q for q in groups if groups[q] == g]
            b = sum(ok(base[q], k) and not ok(app[q], k) for q in qs)
            c = sum(ok(app[q], k) and not ok(base[q], k) for q in qs)
            res[f"budget{k} {g}"] = {"base": sum(ok(base[q], k) for q in qs), "append": sum(ok(app[q], k) for q in qs),
                                     "n": len(qs), "only_base": b, "only_append": c, "p": round(mcnemar(b, c), 6)}
    out["arms"][arm] = res
    primary[arm] = res["budget20 값 비교 조회"]

# 확증 검정: 값 비교 조회, 예산 20, 원래 대 붙임, 방법 3개 Holm
order = sorted(primary, key=lambda a: primary[a]["p"])
run = 0.0
for i, a in enumerate(order):
    run = max(run, min(1.0, (len(order) - i) * primary[a]["p"]))
    primary[a]["p_holm"] = round(run, 6)

(HERE / "analyze.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
