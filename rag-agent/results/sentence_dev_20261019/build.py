"""셀 문장의 문장 형태 비교 입력(dev 395) — PREREG-2026-10-19-sentence-form.md.
Cp = 기존 경로형 "행 경로 > 열 경로: 값"(dev_C_records, T 까지). 같은 셀을 두 문장형으로:
  S1 = "Among <행 경로 ' > '>, the value of <열 경로 ' > '> is <값>."   (문장 틀 + 꺽쇠 경로)
  S2 = "Among <행 경로 ', '>, the value of <열 경로 ', '> is <값>."     (문장 틀, 꺽쇠 없음)
조건 둘: same = Cp 와 같은 셀(토큰 한도 무시, 길어짐) / T = 순위대로 T 까지(셀 수 줄어듦).
  PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python results/sentence_dev_20261019/build.py
"""
import json, sys
from pathlib import Path
from transformers import AutoTokenizer
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "results/format_dev_20261017"))
from format_render import load_split, strip_tbl  # noqa: E402
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); n_tok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
SW = ROOT / "results/prompt_sweep_20261015"
recs = lambda p: {r["query_id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}
A = recs(SW / "dev_A_records.jsonl"); C = recs(SW / "dev_C_records.jsonl"); ids = json.load(open(SW / "dev_ids.json"))
R = {"validation": ROOT / "results/mh_only_20260928/dev_k/dev_s3c_k100_records.jsonl", "train": ROOT / "results/mh_only_20260928/retrieval_r100/mh_train_s3c_records.jsonl"}
jp = lambda path, sep: sep.join(str(p) for p in path)
out = {k: [] for k in ("Cp", "S1same", "S2same", "S1T", "S2T")}; n_same = 0
for split in ("validation", "train"):
    qs = [q for q in ids if A[q]["split"] == split]
    r100 = {r["query_id"]: r for r in map(json.loads, open(R[split], encoding="utf-8")) if r["query_id"] in set(qs) and "doc" in r}
    tables, hdr, sent, coord = load_split(split, qs)
    for q in qs:
        rank = [u[0] for u in r100[q]["doc"]["ranked_units"]]; T = A[q]["n_tok_ctx"]; gold = set(A[q]["gold_ids"]); S = sent[q]; K = coord[q]
        def forms(u):
            tid, i, j = K[u]; t = tables[tid].table; rp = strip_tbl(t.row_path(i)); cp = strip_tbl(t.col_path(j)); v = str(t.data[i][j])
            return (f"Among {jp(rp, ' > ')}, the value of {jp(cp, ' > ')} is {v}.", f"Among {jp(rp, ', ')}, the value of {jp(cp, ', ')} is {v}.")
        sel = []
        for u in rank:
            if n_tok("\n".join(S[c] for c in sel + [u])) > T: break
            sel.append(u)
        cp_ctx = "\n".join(S[c] for c in sel); n_same += (cp_ctx == C[q]["context"])
        base = {k: A[q][k] for k in ("query_id", "kind", "layer", "m", "question", "gold_ids")}; base["budget_T"] = T
        def rec(ctx, cells):
            return {**base, "context": ctx, "correct": int(gold <= set(cells)), "cells_in_context": len(cells), "n_tok_ctx": n_tok(ctx)}
        out["Cp"].append(rec(cp_ctx, sel))
        for k, idx in (("S1same", 0), ("S2same", 1)):
            out[k].append(rec("\n".join(forms(u)[idx] for u in sel), sel))
        for k, idx in (("S1T", 0), ("S2T", 1)):
            s2 = []
            for u in rank:
                if n_tok("\n".join(forms(c)[idx] for c in s2 + [u])) > T: break
                s2.append(u)
            out[k].append(rec("\n".join(forms(u)[idx] for u in s2), s2))
print(f"Cp 재현(기존 dev_C 와 같음) {n_same}/{len(ids)}")
for k, rs in out.items():
    with open(HERE / f"{k}_records.jsonl", "x", encoding="utf-8") as f:
        for x in rs: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"{k}: {len(rs)}건, 정답 셀 포함 {sum(x['correct'] for x in rs)}, 전달 셀 평균 {sum(x['cells_in_context'] for x in rs)/len(rs):.1f}, 토큰 평균 {sum(x['n_tok_ctx'] for x in rs)/len(rs):.0f}, 최대 {max(x['n_tok_ctx'] for x in rs)}")
print("\n예시 Cp:\n" + out["Cp"][0]["context"][:300] + "\n예시 S1:\n" + out["S1same"][0]["context"][:300] + "\n예시 S2:\n" + out["S2same"][0]["context"][:300])
