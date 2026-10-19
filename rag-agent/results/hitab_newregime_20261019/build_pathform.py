"""HiTab 확인 표본 691 의 경로형 목록(Cp) 입력 — PREREG-2026-10-19-hitab-pathform.md. 기존 C 는 문장형 "In the table 'T', among R, the value of C is V."
Cp = 첫 줄 "Table: 제목" + 셀마다 "행 경로 > 열 경로: 값"(MultiHiertt 경로형과 같은 꼴), 같은 순위·같은 T, 부동소수 표기 보정 동일.
  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/hitab_newregime_20261019/build_pathform.py
"""
import glob, json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from rag_agent.bench.hitab_grid import load_table  # noqa: E402
from transformers import AutoTokenizer              # noqa: E402
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); n_tok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
FNOISE = re.compile(r"-?\d+\.\d{7,}"); clean = lambda s: FNOISE.sub(lambda m: format(float(m.group(0)), ".10g"), s)
OUT = HERE / "confirm"; ids = json.load(open(HERE / "confirm_ids.json")); want = set(ids)
A = {r["query_id"]: r for r in map(json.loads, open(OUT / "A_records.jsonl", encoding="utf-8"))}
rec_path = sorted(glob.glob(str(HERE / "retrieval" / "*b100*records.jsonl")))[0]
R = {r["query_id"]: r for r in map(json.loads, open(rec_path, encoding="utf-8")) if r["query_id"] in want}
rows = []
for q in ids:
    r = R[q]; tab = load_table(r["table_id"], "data/hitab"); t = tab.table; T = A[q]["budget_T"]; gold = {(int(c[1]), int(c[2])) for c in r["gold_cells"]}
    units = [(int(u["cells"][0][1]), int(u["cells"][0][2])) for u in r["context_units"] if u.get("cells")]
    line = lambda i, j: clean(f"{' > '.join(map(str, t.row_path(i)))} > {' > '.join(map(str, t.col_path(j)))}: {t.cell(i, j)}")
    head = f"Table: {tab.title}"; sel = []
    for c in units:
        if n_tok("\n".join([head] + [line(*x) for x in sel + [c]])) > T: break
        sel.append(c)
    ctx = "\n".join([head] + [line(*x) for x in sel])
    rows.append({**{k: A[q][k] for k in ("query_id", "table_id", "kind", "layer", "m", "question", "answer", "gold_ids", "budget_T")}, "context": ctx, "correct": int(gold <= set(sel)), "cells_in_context": len(sel), "n_tok_ctx": n_tok(ctx)})
with open(OUT / "Cp_records.jsonl", "x", encoding="utf-8") as f:
    for x in rows: f.write(json.dumps(x, ensure_ascii=False) + "\n")
print(f"Cp: {len(rows)}건, 정답 셀 포함 {sum(x['correct'] for x in rows)}, 전달 셀 평균 {sum(x['cells_in_context'] for x in rows)/len(rows):.1f}, 토큰 평균 {sum(x['n_tok_ctx'] for x in rows)/len(rows):.0f}")
print("예시:\n" + rows[0]["context"][:400])
