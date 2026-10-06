"""PREREG-2026-10-06-claim-chain.md A·B·C 집계 — 200문항(ids200) 공식 EM·형식 허용, 본 방법(주 설정) 대비 McNemar(정확) p.
A 는 검색(정답 셀 전부 포함@20·R@20)도 적는다. C 시험(dev 50)은 같은 문항의 기존 cot 답변(dev k=20)과 짝지어 적는다."""
import json, sys
from math import comb
from pathlib import Path
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from mh_arms import official_answers                                        # noqa: E402
from rag_agent.eval.multihiertt_em import format_match, mh_exact_match      # noqa: E402

R = Path("results"); S = R / "post_defense_20261006/claim"; B = R / "mh_only_20260928/baselines"
IDS = [json.loads(l)["query_id"] for l in open(B / "ids200.jsonl", encoding="utf-8") if l.strip()]
gold = {"train": official_answers("train")}
def score(p, split="train", ids=None):
    if not p.exists(): return None
    g = gold.setdefault(split, official_answers(split)); rows = {}
    for l in open(p, encoding="utf-8"):
        if not l.strip(): continue
        r = json.loads(l)
        if ids is not None and r["query_id"] not in ids: continue
        prog = r["kind"] == "arith"
        rows[r["query_id"]] = (int(mh_exact_match(r["pred"], g[r["query_id"]], prog)), int(format_match(r["pred"], g[r["query_id"]], prog)), r["kind"])
    return rows
def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
def line(name, rows, ref):
    if rows is None or (ref is not None and len(rows) != len(ref)):
        return f"{name} | 진행 중 {0 if rows is None else len(rows)}"
    em, le = sum(v[0] for v in rows.values()), sum(v[1] for v in rows.values())
    lk = sum(v[1] for v in rows.values() if v[2] == "lookup"); ar = sum(v[1] for v in rows.values() if v[2] == "arith")
    if ref is None or rows is ref: return f"{name} | {len(rows)} | {em} | {le} ({100 * le / len(rows):.1f}%) | {lk} / {ar} | — | —"
    b = sum(ref[q][1] and not rows[q][1] for q in rows); c = sum(rows[q][1] and not ref[q][1] for q in rows)
    return f"{name} | {len(rows)} | {em} | {le} ({100 * le / len(rows):.1f}%) | {lk} / {ar} | {b}:{c} | {mcnemar(b, c):.4f}"

ids = set(IDS)
main = score(R / "mh_arms/cap300_20260924/cell_uniq.jsonl", ids=ids)
print("## A·B 답변 (200, 형식 허용 / 공식 EM; b:c = 본 방법만 맞힘:조건만 맞힘)\n조건 | 문항 | 공식 EM | 형식 허용 | 조회 / 산술(허용) | b:c | p")
print(line("본 방법(전체 경로, 셀 문장 20셀)", main, None))
print(line("A 마지막 머리글만 색인", score(S / "A_leaf_answer.jsonl", ids=ids), main))
print(line("값만 색인(기존 ablation, 882 중 200)", score(R / "header_ablation_20260929/mh_value_answer.jsonl", ids=ids), main))
print(line("B1 같은 20셀 마크다운 부분 표", score(S / "B1_subtable_answer.jsonl", ids=ids), main))
print(line("B2 문서 전체 셀 문장", score(S / "B2_fullcells_answer.jsonl", ids=ids), main))
print(line("표 전체 입력(기존, 마크다운 전체)", score(R / "mh_arms/cap300_20260924/fulltable.jsonl", ids=ids), main))

print("\n## A 검색 (2,885·200)")
for name, meta in [("전체 경로(본 방법)", R / "mh_only_20260928/retrieval_r100/mh_train_s3c.json"), ("마지막 머리글만", S / "A_leaf/mh_train_s3c_leaf.json"),
                   ("값만(기존 ablation)", R / "header_ablation_20260929/mh_train_s3c_value.json")]:
    if not meta.exists(): print(name, "| 없음"); continue
    m = json.loads(meta.read_text(encoding="utf-8")); d = m["by_layer"]["ALL"]["doc"]
    rec = meta.with_name(meta.stem + "_records.jsonl")
    n200 = ok200 = 0
    if rec.exists():
        for l in open(rec, encoding="utf-8"):
            if '"doc"' not in l: continue
            r = json.loads(l)
            if r["query_id"] in ids: n200 += 1; ok200 += r["doc"]["correct"]
    print(f"{name} | 정답 셀 전부 포함@20: {d['accuracy_all']:.4f} ({m['n_scored']}) | 200문항: {ok200}/{n200} | @k: {d.get('accuracy_all_at_k')}")

print("\n## C 시험 (dev 산술 50, 계산식 실행)")
p = S / "C_pilot_dev.jsonl"
if p.exists():
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    ref = {json.loads(l)["query_id"]: json.loads(l) for l in open(R / "mh_only_20260928/dev_k/dev_s3c_k20_answer.jsonl", encoding="utf-8")}
    gv = official_answers("validation")
    base = {r["query_id"]: int(format_match(ref[r["query_id"]]["pred"], gv[r["query_id"]], True)) for r in rows if r["query_id"] in ref}
    print(f"n={len(rows)} Program 줄 {sum(r['program'] is not None for r in rows)} 실행됨 {sum(r['used_program'] for r in rows)} 표시 없음 {sum(not r['marker_found'] for r in rows)}")
    print(f"허용: 실행값 {sum(r['answer_correct_lenient'] for r in rows)} / 같은 출력의 LLM 값 {sum(r['llm_correct_lenient'] for r in rows)} / 기존 cot(k=20) {sum(base.values())} (같은 {len(base)}문항)")
    print(f"공식: 실행값 {sum(r['answer_correct'] for r in rows)} / LLM 값 {sum(r['llm_correct'] for r in rows)}")
else:
    print("없음")
