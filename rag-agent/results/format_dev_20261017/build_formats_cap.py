"""표 수 제한 변형 dev 입력 — PREREG-2026-10-18-table-cap.md. 채운 부분 표(D)와 같은 렌더링(format_render.py)·같은 T, 선택만 다르다.
  D1  순위상 첫 표 하나만: 그 표의 셀을 순위대로 T 까지
  D2  순위상 첫 두 표만: 두 표의 셀을 순위대로 T 까지(세 번째 이후 표 셀은 건너뜀)
  DF2 첫 표는 비어 있지 않은 셀 전부(통째로; T 를 넘기면 D1 규칙으로 후퇴), 남는 길이에 두 번째 표 셀을 순위대로
"표의 순위" = 검색 순위에서 그 표의 셀이 처음 나오는 순서. 882 용은 --split882.
  PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python results/format_dev_20261017/build_formats_cap.py [--split882]
"""
import json, sys
from pathlib import Path
from transformers import AutoTokenizer
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from format_render import load_split, make_renderers  # noqa: E402
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
n_tok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])

def build(ids, A, r100_path, split, tag):
    r100 = {r["query_id"]: r for r in map(json.loads, open(r100_path, encoding="utf-8")) if r["query_id"] in set(ids) and "doc" in r}
    assert set(r100) == set(ids)
    tables, hdr, sent, coord = load_split(split, ids)
    out = {"D1": [], "D2": [], "DF2": []}
    for q in ids:
        rank = [u[0] for u in r100[q]["doc"]["ranked_units"]]; T = A[q]["n_tok_ctx"]; gold = set(A[q]["gold_ids"]); K_ = coord[q]
        _g, subtable = make_renderers(tables, hdr, sent[q], K_)
        order = []
        for u in rank:
            tid = K_[u][0]
            if tid not in order: order.append(tid)
        def fill(allowed, seed=()):
            sel, best = list(seed), None
            text, deliv = subtable(sel, True) if sel else ("", set())
            if sel and n_tok(text) <= T: best = (text, deliv, len(sel))
            for u in rank:
                if K_[u][0] not in allowed or u in sel: continue
                text, deliv = subtable(sel + [u], True)
                if n_tok(text) > T: break
                sel.append(u); best = (text, deliv, len(sel))
            return best if best else ("", set(), 0)
        res = {"D1": fill(set(order[:1])), "D2": fill(set(order[:2]))}
        top = order[0]; t = tables[top].table
        full = [u for u in sent[q] if K_[u][0] == top]           # 첫 표의 비어 있지 않은 셀 전부(색인 단위 = 비어 있지 않은 셀)
        tf, _ = subtable(full, True)
        res["DF2"] = fill(set(order[:2]), seed=full) if n_tok(tf) <= T else res["D1"]
        for arm, (text, deliv, nsel) in res.items():
            rec = {k: A[q][k] for k in ("query_id", "kind", "layer", "m", "question", "gold_ids")}
            if "split" in A[q]: rec["split"] = A[q]["split"]; rec["answer"] = A[q]["answer"]
            rec.update({"context": text, "correct": int(gold <= deliv), "cells_in_context": len(deliv), "n_tok_ctx": n_tok(text), "selected_units": nsel, "budget_T": T, "n_tables": text.count("\nTable ") + (1 if text else 0)})
            out[arm].append(rec)
    for arm, rs in out.items():
        print(f"[{split}{tag}] {arm}: {len(rs)}건, 정답 셀 포함 {sum(r['correct'] for r in rs)}, 전달 셀 평균 {sum(r['cells_in_context'] for r in rs)/len(rs):.1f}, 표 수 평균 {sum(r['n_tables'] for r in rs)/len(rs):.2f}, 토큰 평균 {sum(r['n_tok_ctx'] for r in rs)/len(rs):.0f}, T 초과 {sum(r['n_tok_ctx']>r['budget_T'] for r in rs)}")
    return out

if __name__ == "__main__":
    SW = ROOT / "results/prompt_sweep_20261015"
    if "--split882" in sys.argv:
        A = {r["query_id"]: r for r in map(json.loads, open(ROOT / "results/cellform_20261012/A_chunk_md_records.jsonl", encoding="utf-8"))}
        out = build(sorted(A), A, ROOT / "results/mh_only_20260928/retrieval_r100/mh_train_s3c_records.jsonl", "train", "882")
        for arm, rs in out.items():
            with open(HERE / f"{arm}882_records.jsonl", "x", encoding="utf-8") as f:
                for r in rs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        A = {r["query_id"]: r for r in map(json.loads, open(SW / "dev_A_records.jsonl", encoding="utf-8"))}
        ids = json.load(open(SW / "dev_ids.json"))
        R = {"validation": ROOT / "results/mh_only_20260928/dev_k/dev_s3c_k100_records.jsonl", "train": ROOT / "results/mh_only_20260928/retrieval_r100/mh_train_s3c_records.jsonl"}
        allrows = {"D1": {}, "D2": {}, "DF2": {}}
        for split in ("validation", "train"):
            qs = [q for q in ids if A[q]["split"] == split]
            for arm, rs in build(qs, A, R[split], split, "").items():
                for r in rs: allrows[arm][r["query_id"]] = r
        for arm, d in allrows.items():
            rs = [d[q] for q in ids]
            with open(HERE / f"{arm}_records.jsonl", "x", encoding="utf-8") as f:
                for r in rs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[dev] {arm}: 정답 셀 포함 {sum(r['correct'] for r in rs)}/395, 전달 셀 {sum(r['cells_in_context'] for r in rs)/395:.1f}, 표 수 {sum(r['n_tables'] for r in rs)/395:.2f}, 토큰 {sum(r['n_tok_ctx'] for r in rs)/395:.0f}")
