"""전달 단계 상한 진단 입력(새 문항 600) — PREREG-2026-10-21-ceiling-600.md. 렌더링은 D(잘라낸 표, format_render.py)와 같다.
O1 = 정답 셀만 넣은 잘라낸 표(T 제한 없음) → LLM 상한. O3 = D 와 같은 순위·같은 T 이되 정답 표 밖 셀은 뺌 → 표 고르기 상한.
  PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python results/ceiling600_20261021/build_oracle600.py
"""
import json, sys
from pathlib import Path
from transformers import AutoTokenizer
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "results/format_dev_20261017"))
from format_render import load_split, make_renderers, select_render  # noqa: E402
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); n_tok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
MC = ROOT / "results/mh_confirm_20261019"
ids = json.load(open(MC / "ids600.json")); assert len(ids) == 600
D = {r["query_id"]: r for r in map(json.loads, open(MC / "D_records.jsonl", encoding="utf-8"))}; assert set(D) == set(ids)
r100 = {r["query_id"]: r for r in map(json.loads, open(ROOT / "results/mh_only_20260928/retrieval_r100/mh_train_s3c_records.jsonl", encoding="utf-8")) if r["query_id"] in set(ids) and "doc" in r}
assert set(r100) == set(ids)
tables, hdr, sent, coord = load_split("train", ids)
out = {"O1": [], "O3": []}; n_same_d = 0
for q in ids:
    K = coord[q]; gold = list(D[q]["gold_ids"]); gin = [g for g in gold if g in K]; gtids = {K[g][0] for g in gin}
    rank = [u[0] for u in r100[q]["doc"]["ranked_units"]]; T = D[q]["budget_T"]
    _, subtable = make_renderers(tables, hdr, sent[q], K)
    base = {k: D[q][k] for k in ("query_id", "kind", "layer", "m", "question", "gold_ids")}; base["budget_T"] = T
    def rec(text, deliv, extra):
        return {**base, "context": text, "correct": int(set(gold) <= deliv), "cells_in_context": len(deliv), "n_tok_ctx": n_tok(text), **extra}
    t1, d1 = subtable(gin, True); out["O1"].append(rec(t1, d1, {"missing_gold": len(gold) - len(gin)}))
    t3, d3, n3 = select_render([u for u in rank if K[u][0] in gtids], lambda s_: subtable(s_, True), T, n_tok); out["O3"].append(rec(t3, d3, {"selected_units": n3, "n_gold_tables": len(gtids)}))
    tD, _, _ = select_render(rank, lambda s_: subtable(s_, True), T, n_tok); n_same_d += (tD == D[q]["context"])
print(f"D 재현 {n_same_d}/600")
for arm, rs in out.items():
    with open(HERE / f"{arm}_records.jsonl", "x", encoding="utf-8") as f:
        for x in rs: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    toks = [x["n_tok_ctx"] for x in rs]
    print(f"{arm}: {len(rs)}건, 정답 셀 포함 {sum(x['correct'] for x in rs)}, 전달 셀 평균 {sum(x['cells_in_context'] for x in rs)/len(rs):.1f}, 토큰 평균 {sum(toks)/len(rs):.0f}, 최대 {max(toks)}")
print("missing_gold>0:", sum(x["missing_gold"] > 0 for x in out["O1"]), "| 정답 표 2개+:", sum(x["n_gold_tables"] >= 2 for x in out["O3"]))
