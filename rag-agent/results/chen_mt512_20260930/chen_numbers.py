"""TableRAG(Chen) 호출당 생성 상한 512 실행(PREREG-2026-09-29-chen-max-tokens.md)의 원고 수치. 새 생성 없음.
128(원본 설정) 실행과 같은 계산을 나란히 낸다: 검색 R/P/MRR·셀 정밀도(200), 형식 허용·공식 답변, 그룹별, McNemar·Holm(기존 방법 4개),
코드 실행 오류·최종 답 없음·잘림형 SyntaxError.
실행: .venv/bin/python results/chen_mt512_20260930/chen_numbers.py  (저장소 rag-agent 에서) → chen_numbers.json
"""
import json, re, statistics, sys
from math import comb
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "results/mh_only_20260928/final"))
from mh_arms import official_answers
from rag_agent.eval.multihiertt_em import format_match, mh_exact_match
from rpm import per_query, units_of

R = ROOT / "results"; B = R / "mh_only_20260928/baselines"
ids = [json.loads(l)["query_id"] for l in open(B / "ids200.jsonl") if l.strip()]
gold = official_answers("train")
L = lambda p: {r["query_id"]: r for r in map(json.loads, open(p))}
P = lambda a, b: min(1.0, 2 * sum(comb(a + b, i) for i in range(min(a, b) + 1)) / 2 ** (a + b)) if a + b else 1.0
KEYS = ["R@1", "R@5", "R@10", "R@20", "P@1", "P@5", "P@10", "P@20", "MRR"]

ours = L(R / "mh_arms/cap300_20260924/cell_uniq.jsonl")
ok = lambda r: format_match(r["pred"], gold[r["query_id"]], r["kind"] == "arith")
ours_ok = {q: ok(ours[q]) for q in ids}
others = {"LangChain 청크": B / "langchain_chunk/langchain_chunk_answer.jsonl",
          "RAPTOR": B / "raptor/raptor_answer_2000.jsonl",
          "TableRAG(Yu)": B / "tablerag_yu/tablerag_yu_answer.jsonl"}
out = {}
for tag, d in (("chen128", B / "tablerag_chen"), ("chen512", B / "tablerag_chen_mt512")):
    a, rec = L(d / "tablerag_chen_answer.jsonl"), L(d / "tablerag_chen_records.jsonl")
    assert all(q in a and q in rec for q in ids), (tag, sum(q in a for q in ids))
    o = {}
    # 검색
    per, cpu, cp = {}, [], {1: [], 5: [], 10: [], 20: []}
    for q in ids:
        u = units_of(rec[q]); g = set(rec[q]["gold_ids"])
        per[q] = per_query(g, u); cpu += [len(x) for x in u]
        for k in cp:
            S = set().union(*u[:k]) if u[:k] else set()
            cp[k].append(len(g & S) / len(S) if S else 0.0)
    o["retrieval"] = {k: round(statistics.mean(per[q][k] for q in ids), 4) for k in KEYS}
    o["cells_per_unit"] = round(statistics.mean(cpu), 1)
    o["cellP"] = {k: round(100 * statistics.mean(v), 1) for k, v in cp.items()}
    # 답변
    lo = {q: ok(a[q]) for q in ids}
    o["official"] = sum(mh_exact_match(a[q]["pred"], gold[q], a[q]["kind"] == "arith") for q in ids)
    o["lenient"] = sum(lo.values())
    grp = {}
    for q in ids:
        x = grp.setdefault(a[q]["layer"], [0, 0]); x[0] += lo[q]; x[1] += 1
    o["groups"] = grp
    b1 = sum(ours_ok[q] and not lo[q] for q in ids); b2 = sum(lo[q] and not ours_ok[q] for q in ids)
    o["vs_ours"] = [b1, b2, P(b1, b2)]
    # 실행 기록
    o["code_error_q"] = sum(a[q]["n_code_errors"] > 0 for q in ids)
    o["no_final_answer_q"] = sum(not a[q]["terminal"] for q in ids)
    obs = syn = tr = trq = 0
    for q in ids:
        ob = re.findall(r"Observation:(.*?)(?=\n(?:Thought|Action|Answer)|\Z)", a[q]["raw"], re.S)
        s = [x for x in ob if "SyntaxError" in x]; t = [x for x in s if "was never closed" in x or "unterminated" in x]
        obs += len(ob); syn += len(s); tr += len(t); trq += bool(t)
    o["observations"], o["syntax_errors"], o["truncation_errors"], o["truncation_q"] = obs, syn, tr, trq
    o["mean_llm_calls"] = round(statistics.mean(a[q]["n_llm_calls"] for q in ids), 2)
    # Holm: 기존 방법 4개(LangChain, RAPTOR 2,000, Chen, Yu)
    ps = {"TableRAG(Chen)": o["vs_ours"][2]}
    for n, f in others.items():
        r = L(f); x = {q: ok(r[q]) for q in ids}
        c1 = sum(ours_ok[q] and not x[q] for q in ids); c2 = sum(x[q] and not ours_ok[q] for q in ids)
        ps[n] = P(c1, c2)
    run, holm = 0, {}
    for i, (n, p) in enumerate(sorted(ps.items(), key=lambda t: t[1])):
        run = max(run, min(1.0, (len(ps) - i) * p)); holm[n] = [p, run]
    o["holm4"] = holm
    out[tag] = o
(Path(__file__).parent / "chen_numbers.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
