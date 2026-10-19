"""MultiHiertt 새 문항 확인 표본(882·dev 밖, 산술 셀2+ 층, 시드 42 로 600개) 입력 — PREREG-2026-10-19-mh-confirm-arith.md.
A = LangChain 4청크 원문(기존 baselines/langchain_chunk 기록 그대로), C = 셀 문장 순위대로 T 까지, D = 채운 부분 표 T 까지(format_render.py, 882 와 같은 규칙).
  PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python results/mh_confirm_20261019/build_fresh.py
"""
import json, random, sys
from pathlib import Path
from transformers import AutoTokenizer
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "results/format_dev_20261017"))
from format_render import load_split, make_renderers, select_render  # noqa: E402
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); n_tok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
ids882 = {json.loads(l)["query_id"] for l in open(ROOT / "results/cellform_20261012/A_chunk_md_records.jsonl", encoding="utf-8")}
dev = set(json.load(open(ROOT / "results/prompt_sweep_20261015/dev_ids.json")))
LC = {r["query_id"]: r for r in map(json.loads, open(ROOT / "results/mh_only_20260928/baselines/langchain_chunk/langchain_chunk_records.jsonl", encoding="utf-8"))}
r100 = {}
for l in open(ROOT / "results/mh_only_20260928/retrieval_r100/mh_train_s3c_records.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["query_id"] in ids882 or r["query_id"] in dev or "doc" not in r or r["layer"] != "arith_m2+" or r["query_id"] not in LC: continue
    r100[r["query_id"]] = r
pool = sorted(r100); print(f"모집단(882·dev 밖 산술 셀2+, LC 기록 있음) {len(pool)}")
ids = sorted(random.Random(42).sample(pool, 600)); json.dump(ids, open(HERE / "ids600.json", "w"))
tables, hdr, sent, coord = load_split("train", ids)
out = {"A": [], "C": [], "D": []}
for q in ids:
    L, R = LC[q], r100[q]; gold = set(L["gold_ids"]); assert gold == set(R["gold_ids"]), q
    rank = [u[0] for u in R["doc"]["ranked_units"]]; a_ctx = L["context"]; T = n_tok(a_ctx)
    a_cells = set().union(*(set(u["cells"]) for u in L["ranked_units"][:4]))
    base = {"query_id": q, "kind": L["kind"], "layer": L["layer"], "m": L["m"], "question": L["question"], "gold_ids": L["gold_ids"], "budget_T": T}
    out["A"].append({**base, "context": a_ctx, "correct": int(gold <= a_cells), "cells_in_context": len(a_cells), "n_tok_ctx": T, "n_chunks_in_doc": L["n_chunks_in_doc"]})
    S = sent[q]; sel = []
    for u in rank:
        if n_tok("\n".join(S[c] for c in sel + [u])) > T: break
        sel.append(u)
    c_ctx = "\n".join(S[c] for c in sel)
    out["C"].append({**base, "context": c_ctx, "correct": int(gold <= set(sel)), "cells_in_context": len(sel), "n_tok_ctx": n_tok(c_ctx)})
    _, subtable = make_renderers(tables, hdr, S, coord[q])
    text, deliv, nsel = select_render(rank, lambda s_: subtable(s_, True), T, n_tok)
    out["D"].append({**base, "context": text, "correct": int(gold <= deliv), "cells_in_context": len(deliv), "n_tok_ctx": n_tok(text), "selected_units": nsel})
for arm, rs in out.items():
    with open(HERE / f"{arm}_records.jsonl", "x", encoding="utf-8") as f:
        for x in rs: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"{arm}: {len(rs)}건, 정답 셀 포함 {sum(x['correct'] for x in rs)}, 전달 셀 평균 {sum(x['cells_in_context'] for x in rs)/len(rs):.1f}, 토큰 평균 {sum(x['n_tok_ctx'] for x in rs)/len(rs):.0f}, 최대 {max(x['n_tok_ctx'] for x in rs)}" + (f", 문서당 청크 평균 {sum(x['n_chunks_in_doc'] for x in rs)/len(rs):.2f}, 4개 이하 {sum(x['n_chunks_in_doc']<=4 for x in rs)}" if arm == "A" else ""))
import collections; print("m 분포:", dict(sorted(collections.Counter(x["m"] for x in out["A"]).items())))
