"""셀 문장 이득 분해 실험 입력 [가]·[나]·[다] 만들기 — PREREG-2026-10-12-cellform.md.

[가] LangChain 청크 상위 4개 원문(markdown).  T = [가] 문자열의 토큰 수(Qwen3-8B 토크나이저).
[나] [가] 청크에 값이 통째로 든 셀 전부를 셀 문장(본 방법 색인 문장)으로, 청크 순서·청크 안 (표, 행, 열) 순, 중복 제거, 줄바꿈 연결.
[다] 본 방법 검색 순위(상위 100 기록)대로 셀 문장을 넣되, 줄바꿈으로 이은 문자열이 T 토큰을 넘기기 직전에 멈춤.
출력: 조건마다 langchain_chunk.py --stage answer 가 읽는 레코드 jsonl.
  PYTHONPATH=. .venv/bin/python results/cellform_20261012/build_inputs.py
"""
import json, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from mh_arms import build_tables, cell_id, load_population          # noqa: E402
from retrieval_accuracy import build_corpus                          # noqa: E402
from transformers import AutoTokenizer                               # noqa: E402

OUT = ROOT / "results/cellform_20261012"
IDS = ROOT / "results/mh_arms/cap300_20260924/cell_uniq.jsonl"
LC = ROOT / "results/mh_only_20260928/baselines/langchain_chunk/langchain_chunk_records.jsonl"
R100 = ROOT / "results/mh_only_20260928/retrieval_r100/mh_train_s3c_records.jsonl"

ids = [json.loads(l)["query_id"] for l in open(IDS) if l.strip()]
want = set(ids)
lc = {r["query_id"]: r for r in map(json.loads, open(LC)) if r["query_id"] in want}
r100 = {r["query_id"]: r for r in map(json.loads, open(R100)) if r["query_id"] in want and "doc" in r}
assert len(lc) == len(r100) == len(ids) == 882

# 색인 문장: answer_accuracy_mh.rowexp_contexts(noise) 와 같은 코드 경로 (v3.3u, 라벨 없음, s3c, cell)
_q, docs, _ = load_population("train")
tables, hdr = build_tables({u: docs[u] for u in ids}, "v3.3u", "none")
texts, covers, _, unit_tids, _ = build_corpus("", sorted(tables), "s3c", "cell", {}, load=lambda t, _d: tables.get(t))
sent = defaultdict(dict)
for txt, cov, tid in zip(texts, covers, unit_tids):
    sent[tid.split("::")[0]][cell_id(next(iter(cov)), hdr).split("::", 1)[1]] = txt

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
n_tok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
key = lambda c: tuple(int(x) for x in c.split("-"))

rows = {"A_chunk_md": [], "B_chunk_sent": [], "C_ours_budget": []}
stats = defaultdict(list)
for q in ids:
    L, R, S = lc[q], r100[q], sent[q]
    gold = set(L["gold_ids"])
    assert gold == set(R["gold_ids"]) if "gold_ids" in R else True
    # 본 방법 상위 20줄이 다시 만든 문장과 같은지 (문장 재현 확인)
    top20 = [S[u[0]] for u in R["doc"]["ranked_units"][:20]]
    assert top20 == R["doc"]["context"], q
    base = {"query_id": q, "kind": L["kind"], "layer": L["layer"], "m": L["m"], "question": L["question"], "gold_ids": L["gold_ids"]}
    # [가]
    a_ctx = L["context"]
    T = n_tok(a_ctx)
    a_cells = set().union(*(set(u["cells"]) for u in L["ranked_units"][:4]))
    rows["A_chunk_md"].append({**base, "context": a_ctx, "correct": int(gold <= a_cells), "cells_in_context": len(a_cells), "n_tok_ctx": T})
    # [나]
    seen, b_cells = set(), []
    for u in L["ranked_units"][:4]:
        for c in sorted(u["cells"], key=key):
            if c not in seen:
                seen.add(c); b_cells.append(c)
    missing = [c for c in b_cells if c not in S]
    assert not missing, (q, missing[:3])
    b_ctx = "\n".join(S[c] for c in b_cells)
    assert set(b_cells) == a_cells
    rows["B_chunk_sent"].append({**base, "context": b_ctx, "correct": int(gold <= set(b_cells)), "cells_in_context": len(b_cells), "n_tok_ctx": n_tok(b_ctx)})
    # [다]
    sel = []
    for u in R["doc"]["ranked_units"]:
        cand = sel + [u[0]]
        if n_tok("\n".join(S[c] for c in cand)) > T:
            break
        sel = cand
    c_ctx = "\n".join(S[c] for c in sel)
    rows["C_ours_budget"].append({**base, "context": c_ctx, "correct": int(gold <= set(sel)), "cells_in_context": len(sel), "n_tok_ctx": n_tok(c_ctx)})
    stats["exhausted_r100"].append(int(len(sel) == len(R["doc"]["ranked_units"])))

for name, rs in rows.items():
    with open(OUT / f"{name}_records.jsonl", "x") as f:
        for r in rs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(rs)
    print(f"{name}: 정답 셀 포함 {sum(r['correct'] for r in rs)}/{n}, 셀 평균 {sum(r['cells_in_context'] for r in rs)/n:.1f}, "
          f"토큰 평균 {sum(r['n_tok_ctx'] for r in rs)/n:.0f}")
print("[다] 상위 100 을 다 쓰고도 예산이 남은 문항:", sum(stats["exhausted_r100"]))
