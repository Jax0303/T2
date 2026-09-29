"""형식 허용 채점으로 모든 방법 재채점 (2026-09-30, 지도교수 지시: 채점 형식 차이는 정답 처리). 새 생성 없음.

허용 규칙 = 공식 EM 이 맞으면 정답, 아니면
  산술(정답에 계산식 있음): DocMath-Eval compare_two_numbers (x100/x1000 척도, 상대오차 0.15%, 부호 무시) — 기존 docmath_match
  조회(span): 날짜의 '월 일,'·'$'·'level' 을 지운 뒤, 정답에 숫자가 있으면 숫자 집합이 같을 때, 없으면 단어 묶음이 같을 때 정답
조회 규칙은 본 방법의 오답(results/wrong_cause_20260930)을 본 뒤 정했다 — 모든 방법에 같은 함수로 적용한다.
반올림 1% 같은 수치 허용은 넣지 않는다(형식 차이가 아님).
"""
import json, re, sys
from math import comb
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from mh_arms import official_answers
from rag_agent.eval.multihiertt_em import docmath_match, mh_exact_match

MON = r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?"
def _clean(s): return re.sub(MON, "", str(s).lower()).replace("$", "").replace("level", "")
def _nums(s): return sorted(set(float(x) for x in re.findall(r"-?\d+\.?\d*", _clean(s).replace(",", "").replace("%", ""))))
def _words(s): return sorted(re.sub(r"[^a-z0-9. ]", " ", _clean(s)).split())

def lenient(pred, gold, program):
    if pred is None: return False
    if mh_exact_match(pred, gold, program): return True
    if program: return docmath_match(pred, gold, True)
    g = _nums(gold)
    return (g == _nums(pred)) if g else (_words(gold) == _words(pred) != [])

R = ROOT / "results"
FILES = {  # 이름: (파일, 문항 범위)
    "본 방법": R / "mh_arms/cap300_20260924/cell_uniq.jsonl",
    "머리글 경로 제거": R / "header_ablation_20260929/mh_value_answer.jsonl",
    "표 전체 입력(참고)": R / "mh_arms/cap300_20260924/fulltable.jsonl",
    "공식 근거 셀만(부록 D)": R / "gold_only_20260929/mh_gold_cot384.jsonl",
    "LangChain 청크": R / "mh_only_20260928/baselines/langchain_chunk/langchain_chunk_answer.jsonl",
    "RAPTOR(2,000토큰)": R / "mh_only_20260928/baselines/raptor/raptor_answer_2000.jsonl",
    "RAPTOR(575토큰)": R / "mh_only_20260928/baselines/raptor/raptor_answer_575.jsonl",
    "TableRAG(Chen)": R / "mh_only_20260928/baselines/tablerag_chen/tablerag_chen_answer.jsonl",
    "TableRAG(Yu)": R / "mh_only_20260928/baselines/tablerag_yu/tablerag_yu_answer.jsonl",
    "셀 문장+튜토리얼 지시문": R / "cell_tutorial_prompt_20260930/answer.jsonl",
}
gold = official_answers("train")
ids200 = [json.loads(l)["query_id"] for l in open(R / "mh_only_20260928/baselines/ids200.jsonl") if l.strip()]

def exact_p(a, b):
    n = a + b
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n) if n else 1.0

res = {}
for name, f in FILES.items():
    rows = {r["query_id"]: r for r in map(json.loads, open(f))}
    off, len_ = {}, {}
    for q, r in rows.items():
        prog = r["kind"] == "arith"
        off[q] = mh_exact_match(r["pred"], gold[q], prog)
        assert off[q] == bool(r["answer_correct"]), (name, q)       # 공식 채점 재현 확인
        len_[q] = lenient(r["pred"], gold[q], prog)
    res[name] = {"off": off, "len": len_}

ours = res["본 방법"]["len"]
out = {}
for name, d in res.items():
    for tag, ids in (("882", list(d["len"])), ("200", ids200)):
        ids = [q for q in ids if q in d["len"]]
        if tag == "882" and len(ids) != 882: continue
        a = sum(ours[q] and not d["len"][q] for q in ids); b = sum(d["len"][q] and not ours[q] for q in ids)
        by = {}
        for q in ids:
            k = res["본 방법"] and json.loads(json.dumps(None))
        out.setdefault(name, {})[tag] = {"n": len(ids), "official": sum(d["off"][q] for q in ids),
                                          "lenient": sum(d["len"][q] for q in ids),
                                          "vs_ours": None if name == "본 방법" else {"ours_only": a, "other_only": b, "p": exact_p(a, b)}}
# Holm: 200문항 기존 방법 5개
fam = ["LangChain 청크", "RAPTOR(2,000토큰)", "RAPTOR(575토큰)", "TableRAG(Chen)", "TableRAG(Yu)"]
ps = sorted((out[n]["200"]["vs_ours"]["p"], n) for n in fam)
run = 0
for i, (p, n) in enumerate(ps):
    run = max(run, min(1.0, (len(ps) - i) * p)); out[n]["200"]["vs_ours"]["holm"] = run
# 그룹별 (본 방법, 200·882)
grp = {}
for name in ("본 방법", "표 전체 입력(참고)", "공식 근거 셀만(부록 D)", "머리글 경로 제거"):
    rows = {r["query_id"]: r for r in map(json.loads, open(FILES[name]))}
    for q, r in rows.items():
        g = grp.setdefault(name, {}).setdefault(r["layer"], [0, 0]); g[0] += res[name]["len"][q]; g[1] += 1
out["_groups_882_lenient"] = grp
for name in ["본 방법"] + fam + ["표 전체 입력(참고)", "셀 문장+튜토리얼 지시문"]:
    rows = {r["query_id"]: r for r in map(json.loads, open(FILES[name]))}
    g = {}
    for q in ids200:
        x = g.setdefault(rows[q]["layer"], [0, 0]); x[0] += res[name]["len"][q]; x[1] += 1
    out.setdefault("_groups_200_lenient", {})[name] = g
(Path(__file__).parent / "rescore.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
for n, v in out.items():
    if not n.startswith("_"): print(n, json.dumps(v, ensure_ascii=False))
print(json.dumps(out["_groups_882_lenient"], ensure_ascii=False)); print(json.dumps(out["_groups_200_lenient"], ensure_ascii=False))
