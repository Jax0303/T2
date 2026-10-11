"""[라] D_ours_md — [다]가 고른 셀을 그대로, markdown 부분 표로만 바꾼 입력 (PREREG-2026-10-12-cellform.md 추가 조건).
[다] 선택을 build_inputs.py 와 같은 규칙으로 다시 만들고 저장된 [다] 문맥과 글자 단위로 같은지 확인한 뒤,
같은 셀을 retrieval_accuracy.subtable_context(header_mode="path", 행 머리글 = 행 경로 전체, 열 머리글 = 열 경로 전체)로 그린다.
  PYTHONPATH=. .venv/bin/python results/cellform_20261012/build_md.py
"""
import json, sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from mh_arms import build_tables, cell_id, load_population          # noqa: E402
from retrieval_accuracy import build_corpus, subtable_context        # noqa: E402
from transformers import AutoTokenizer                               # noqa: E402

D = ROOT / "results/cellform_20261012"
ids = [json.loads(l)["query_id"] for l in open(D / "ids_order.jsonl")]
A = {r["query_id"]: r for r in map(json.loads, open(D / "A_chunk_md_records.jsonl"))}
C = {r["query_id"]: r for r in map(json.loads, open(D / "C_ours_budget_records.jsonl"))}
R100 = {r["query_id"]: r for r in map(json.loads, open(ROOT / "results/mh_only_20260928/retrieval_r100/mh_train_s3c_records.jsonl"))
        if r["query_id"] in A and "doc" in r}

_q, docs, _ = load_population("train")
tables, hdr = build_tables({u: docs[u] for u in ids}, "v3.3u", "none")
texts, covers, _, unit_tids, _ = build_corpus("", sorted(tables), "s3c", "cell", {}, load=lambda t, _d: tables.get(t))
sent, coord = defaultdict(dict), defaultdict(dict)
for txt, cov, tid in zip(texts, covers, unit_tids):
    c = next(iter(cov)); s = cell_id(c, hdr).split("::", 1)[1]
    sent[tid.split("::")[0]][s] = txt; coord[tid.split("::")[0]][s] = c
tabs = {t: SimpleNamespace(table=v.table, title="") for t, v in tables.items()}
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
n_tok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])

rows = []
for q in ids:
    S, T = sent[q], n_tok(A[q]["context"])
    sel = []
    for u in R100[q]["doc"]["ranked_units"]:
        cand = sel + [u[0]]
        if n_tok("\n".join(S[c] for c in cand)) > T:
            break
        sel = cand
    assert "\n".join(S[c] for c in sel) == C[q]["context"], q          # [다]와 같은 셀
    units = subtable_context(sorted(coord[q][c] for c in sel), tabs, "", {}, header_mode="path")
    ctx = "\n\n".join("\n".join(ln for ln in u["text"].split("\n") if ln.strip() != "#") for u in units)
    rows.append({**{k: C[q][k] for k in ("query_id", "kind", "layer", "m", "question", "gold_ids", "correct", "cells_in_context")},
                 "context": ctx, "n_tok_ctx": n_tok(ctx)})
with open(D / "D_ours_md_records.jsonl", "x") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
n = len(rows)
print(f"D_ours_md: 정답 셀 포함 {sum(r['correct'] for r in rows)}/{n}, 셀 평균 {sum(r['cells_in_context'] for r in rows)/n:.1f}, "
      f"토큰 평균 {sum(r['n_tok_ctx'] for r in rows)/n:.0f}, 최대 {max(r['n_tok_ctx'] for r in rows)}")
print(rows[0]["context"][:600])
