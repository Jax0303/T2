"""검색 기록(ranked_units) → 본 방법 전달 레코드 — PREREG-2026-10-15-prf.md. cellform 의 C 규칙 그대로:
점수 순 셀 문장을 줄바꿈으로 잇되, 토큰 수가 T(같은 문항의 LangChain 청크 4개 원문 토큰 수)를 넘기기 직전에 멈춤.
  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/prf_20261015/build_inputs.py <split> <records.jsonl> <ids.json> <lc_records.jsonl> <out_records.jsonl> [--check <기존 레코드>]
"""
import json, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from mh_arms import build_tables, cell_id, load_population          # noqa: E402
from retrieval_accuracy import build_corpus                          # noqa: E402
from transformers import AutoTokenizer                               # noqa: E402

split, rec_p, ids_p, lc_p, out_p = sys.argv[1:6]
check = sys.argv[7] if len(sys.argv) > 7 and sys.argv[6] == "--check" else None
ids = json.load(open(ids_p))
R = {r["query_id"]: r for r in map(json.loads, open(rec_p, encoding="utf-8")) if "doc" in r}
LC = {r["query_id"]: r for r in map(json.loads, open(lc_p, encoding="utf-8")) if r["query_id"] in set(ids)}
assert set(ids) <= set(R) and set(ids) <= set(LC), (len(set(ids) - set(R)), len(set(ids) - set(LC)))
_q, docs, _ = load_population(split)
tables, hdr = build_tables({u: docs[u] for u in ids}, "v3.3u", "none")
texts, covers, _, unit_tids, _ = build_corpus("", sorted(tables), "s3c", "cell", {}, load=lambda t, _d: tables.get(t))
sent = defaultdict(dict)
for txt, cov, tid in zip(texts, covers, unit_tids):
    sent[tid.split("::")[0]][cell_id(next(iter(cov)), hdr).split("::", 1)[1]] = txt
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
n_tok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
rows = []
for q in ids:
    L, Rq, S = LC[q], R[q], sent[q]
    T = n_tok(L["context"]); gold = set(L["gold_ids"])
    assert gold == set(Rq["gold_ids"]), q
    lines, cells = [], set()
    for u in Rq["doc"]["ranked_units"]:
        cand = lines + [S[u[0]]]
        if n_tok("\n".join(cand)) > T:
            break
        lines = cand; cells |= set(u)
    rows.append({"query_id": q, "kind": L["kind"], "layer": L["layer"], "m": L["m"], "question": L["question"], "gold_ids": L["gold_ids"],
                 "context": "\n".join(lines), "correct": int(gold <= cells), "cells_in_context": len(cells), "n_tok_ctx": n_tok("\n".join(lines)),
                 "prf": Rq.get("prf", {}).get("n", 0)})
Path(out_p).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
print(f"{out_p}: {len(rows)}건, 정답 포함 {sum(r['correct'] for r in rows)}, 셀 평균 {sum(r['cells_in_context'] for r in rows)/len(rows):.1f}, 토큰 평균 {sum(r['n_tok_ctx'] for r in rows)/len(rows):.0f}")
if check:
    C = {r["query_id"]: r for r in map(json.loads, open(check, encoding="utf-8"))}
    same = sum(C[r["query_id"]]["context"] == r["context"] for r in rows)
    print(f"기존 레코드와 본문 같음 {same}/{len(rows)}")
