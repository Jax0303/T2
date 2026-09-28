"""2026-09-28 원고 4.1·6.5 머리글 규칙 수정 이력 문장의 수치를 결과 파일에서 다시 센다(모델 실행 없음).
출처 서술: DATA-USE-2026-09-14.md §1·§3 표, PREREG-2026-09-13-header-units-note.md, PREREG-2026-09-14-header-v3.md(정정 1~4).
- v2: v1 답변 실행(train, Qwen2.5-7B neutral/64)의 문항 = results/mh_arms/mh_cell_answer_doc.jsonl 의 질의 수.
- v3: interim200(train) = results/mh_interim200/ids_200.json. 본 방법(cell_hv2) 대 1,000자 청크(chunk_hv2) 답변 불일치와 청크만 맞힌 수
  = results/mh_interim200/mh_{cell,chunk}_hv2_answer_doc_qwen3_8b_cot_200.jsonl 의 answer_correct.
- v3.1~v3.3: train 표 수 = results/rerun_20260926/mh/mh_train_s3c.json 의 n_tables(모집단 문서의 표 전부, 최종 규칙 색인).
실행: .venv/bin/python results/header_history_20260928/values.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
jl = lambda p: [json.loads(x) for x in open(ROOT / p) if x.strip()]
v1 = jl("results/mh_arms/mh_cell_answer_doc.jsonl")
ids200 = json.load(open(ROOT / "results/mh_interim200/ids_200.json"))
ids200 = ids200 if isinstance(ids200, list) else ids200.get("query_ids") or ids200.get("ids")
cell = {r["query_id"]: int(r["answer_correct"]) for r in jl("results/mh_interim200/mh_cell_hv2_answer_doc_qwen3_8b_cot_200.jsonl")}
chunk = {r["query_id"]: int(r["answer_correct"]) for r in jl("results/mh_interim200/mh_chunk_hv2_answer_doc_qwen3_8b_cot_200.jsonl")}
assert set(cell) == set(chunk) == set(ids200)
out = {"v1_answer_queries": len({r["query_id"] for r in v1}),
       "interim200_queries": len(ids200),
       "interim200_cell_vs_chunk_discordant": sum(cell[q] != chunk[q] for q in cell),
       "interim200_chunk_only_correct": sum(chunk[q] and not cell[q] for q in cell),
       "interim200_cell_only_correct": sum(cell[q] and not chunk[q] for q in cell),
       "train_tables": json.load(open(ROOT / "results/rerun_20260926/mh/mh_train_s3c.json"))["n_tables"]}
(Path(__file__).parent / "values.json").write_text(json.dumps(out, indent=1))
print(out)
