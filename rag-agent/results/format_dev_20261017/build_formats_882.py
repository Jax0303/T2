"""882 확인용 D(채운 부분 표) 입력 — PREREG-2026-10-17-delivery-format-882.md. 규칙은 dev 와 같다(T = LangChain 청크 4개 토큰, 순위대로 더해 T 직전까지).
  --verify-dev: dev 395 의 D 레코드를 이 모듈로 다시 만들어 커밋본(ac9aea7)과 글자 단위로 대조한다(렌더링 동일성 확인).
  PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python results/format_dev_20261017/build_formats_882.py [--verify-dev]
"""
import json, sys
from pathlib import Path
from transformers import AutoTokenizer
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from format_render import load_split, make_renderers, select_render  # noqa: E402
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
n_tok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])

def build(ids, A, C, r100_path, split, out_path=None):
    r100 = {r["query_id"]: r for r in map(json.loads, open(r100_path, encoding="utf-8")) if r["query_id"] in set(ids) and "doc" in r}
    assert set(r100) == set(ids), (len(r100), len(ids))
    tables, hdr, sent, coord = load_split(split, ids)
    rows, n_same = [], 0
    for q in ids:
        rank = [u[0] for u in r100[q]["doc"]["ranked_units"]]; T = A[q]["n_tok_ctx"]; gold = set(A[q]["gold_ids"])
        grouped, subtable = make_renderers(tables, hdr, sent[q], coord[q])
        sel = []
        for u in rank:
            if n_tok("\n".join(sent[q][c] for c in sel + [u])) > T: break
            sel.append(u)
        n_same += ("\n".join(sent[q][c] for c in sel) == C[q]["context"])
        text, deliv, nsel = select_render(rank, lambda s_: subtable(s_, True), T, n_tok)
        rec = {k: A[q][k] for k in ("query_id", "kind", "layer", "m", "question", "gold_ids")}
        rec.update({"context": text, "correct": int(gold <= deliv), "cells_in_context": len(deliv), "n_tok_ctx": n_tok(text), "selected_units": nsel, "budget_T": T})
        rows.append(rec)
    print(f"[{split}] {len(rows)}건, C 재현 {n_same}/{len(rows)}, D 정답 셀 포함 {sum(r['correct'] for r in rows)}, 전달 셀 평균 {sum(r['cells_in_context'] for r in rows)/len(rows):.1f}, 토큰 평균 {sum(r['n_tok_ctx'] for r in rows)/len(rows):.0f}, T 초과 {sum(r['n_tok_ctx']>r['budget_T'] for r in rows)}")
    assert n_same == len(rows)
    if out_path:
        with open(out_path, "x", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows

if __name__ == "__main__":
    SW = ROOT / "results/prompt_sweep_20261015"
    if "--verify-dev" in sys.argv:
        A = {r["query_id"]: r for r in map(json.loads, open(SW / "dev_A_records.jsonl", encoding="utf-8"))}
        C = {r["query_id"]: r for r in map(json.loads, open(SW / "dev_C_records.jsonl", encoding="utf-8"))}
        ids = json.load(open(SW / "dev_ids.json")); old = {r["query_id"]: r for r in map(json.loads, open(HERE / "D_records.jsonl", encoding="utf-8"))}
        R = {"validation": ROOT / "results/mh_only_20260928/dev_k/dev_s3c_k100_records.jsonl", "train": ROOT / "results/mh_only_20260928/retrieval_r100/mh_train_s3c_records.jsonl"}
        same = 0
        for split in ("validation", "train"):
            qs = [q for q in ids if A[q]["split"] == split]
            for r in build(qs, A, C, R[split], split):
                same += (r["context"] == old[r["query_id"]]["context"] and r["cells_in_context"] == old[r["query_id"]]["cells_in_context"])
        print(f"[verify-dev] 커밋본 D 레코드와 같음 {same}/{len(ids)}"); assert same == len(ids)
    else:
        A = {r["query_id"]: r for r in map(json.loads, open(ROOT / "results/cellform_20261012/A_chunk_md_records.jsonl", encoding="utf-8"))}
        C = {r["query_id"]: r for r in map(json.loads, open(ROOT / "results/cellform_20261012/C_ours_budget_records.jsonl", encoding="utf-8"))}
        ids = sorted(A); assert len(ids) == 882
        build(ids, A, C, ROOT / "results/mh_only_20260928/retrieval_r100/mh_train_s3c_records.jsonl", "train", HERE / "D882_records.jsonl")
