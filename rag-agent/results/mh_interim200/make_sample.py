# SPDX-License-Identifier: MIT
"""중간 보고용 200문항 탐색 비교 — 질의 id 표본. 답변 결과를 읽지 않는다.

기존 답변 레그의 1,047개 질의 id 에서 층별 고정 개수를 고정 시드로 뽑는다. 읽는 필드는
query_id, layer, context_sha256 셋뿐이다(정답 여부·예측은 읽지 않는다). 층 이름 순서대로
정렬된 id 에 random.Random(SEED).sample 을 부르므로 같은 입력이면 항상 같은 목록이 나온다.
같은 id 목록과 기준 문맥 해시 파일을 함께 쓴다.

  .venv/bin/python results/mh_interim200/make_sample.py
"""
import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE.parents[0] / "mh_arms"
SEED = 20260913
QUOTA = {"arith_m1": 14, "arith_m2+": 76, "lookup_m1": 40, "lookup_m2+": 70}
SRC = {"ours": D / "mh_cell_hv2_answer_doc.jsonl"}


def fields(path):
    return {r["query_id"]: {"layer": r["layer"], "context_sha256": r["context_sha256"]}
            for r in map(json.loads, path.open(encoding="utf-8"))}


arms = {k: fields(p) for k, p in SRC.items()}
if len(arms["ours"]) != 1047:
    raise SystemExit("기준 파일의 질의 id 가 1,047개가 아니다")

rng = random.Random(SEED)
by_layer, population = {}, {}
for layer in sorted(QUOTA):
    pool = sorted(i for i, v in arms["ours"].items() if v["layer"] == layer)
    population[layer] = len(pool)
    by_layer[layer] = sorted(rng.sample(pool, QUOTA[layer]))
ids = sorted(i for v in by_layer.values() for i in v)

spec = {"purpose": "중간 보고용 탐색 비교 (사전등록 아님)", "seed": SEED, "quota": QUOTA,
        "population_per_layer": population,
        "rule": "층 이름 순서대로, 층 안 id 를 정렬한 뒤 random.Random(seed).sample(pool, quota)",
        "fields_read": ["query_id", "layer", "context_sha256"],
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SRC.values()},
        "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "ids_by_layer": by_layer, "ids": ids}
with (HERE / "ids_200.json").open("x", encoding="utf-8") as f:
    json.dump(spec, f, ensure_ascii=False, indent=1)
for k in arms:
    with (HERE / f"ref_contexts_{k}_200.jsonl").open("x", encoding="utf-8", newline="\n") as f:
        for i in ids:
            f.write(json.dumps({"query_id": i, "context_sha256": arms[k][i]["context_sha256"]}) + "\n")
print(json.dumps({"n": len(ids), "per_layer": {k: len(v) for k, v in by_layer.items()},
                  "population": population, "ids_sha256": spec["ids_sha256"]}, ensure_ascii=False))
