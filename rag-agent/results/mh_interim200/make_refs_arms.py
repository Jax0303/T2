# SPDX-License-Identifier: MIT
"""interim200 비교군 네 개의 기준 문맥 해시. 기존 답변 레그에서 query_id 와 context_sha256 만 쓴다.

ids_200.json(커밋된 표본)은 바꾸지 않는다. 표본을 뽑은 기준 파일이 그대로인지, 비교군 답변 파일의
질의 id 집합이 그 1,047개와 같은지 확인한 뒤 200개 id 의 문맥 해시를 ref_contexts_<arm>_200.jsonl 로 쓴다.

  .venv/bin/python results/mh_interim200/make_refs_arms.py
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE.parents[0] / "mh_arms"
ARMS = {"chunk": "mh_chunk_hv2", "huawei": "mh_huawei_hv2", "rowcol": "mh_rowcol_hv2",
        "tablerag": "mh_tablerag_allobj_path_hv2"}

spec = json.loads((HERE / "ids_200.json").read_text(encoding="utf-8"))
src = D / "mh_cell_hv2_answer_doc.jsonl"
if hashlib.sha256(src.read_bytes()).hexdigest() != spec["source_sha256"][src.name]:
    raise SystemExit("표본을 뽑은 기준 파일이 바뀌었다")
population = {json.loads(l)["query_id"] for l in src.open(encoding="utf-8")}
for arm, tag in ARMS.items():
    path = D / f"{tag}_answer_doc.jsonl"
    ctx = {r["query_id"]: r["context_sha256"] for r in map(json.loads, path.open(encoding="utf-8"))}
    if set(ctx) != population:
        raise SystemExit(f"{arm}: 질의 id 집합이 표본 모집단 1,047개와 다르다")
    with (HERE / f"ref_contexts_{arm}_200.jsonl").open("x", encoding="utf-8", newline="\n") as f:
        for i in spec["ids"]:
            f.write(json.dumps({"query_id": i, "context_sha256": ctx[i]}) + "\n")
    print(arm, path.name, hashlib.sha256(path.read_bytes()).hexdigest())
