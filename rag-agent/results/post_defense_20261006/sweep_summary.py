"""PREREG-2026-10-06-post-defense.md C·D: 조건(주 설정 / 둘째 LLM / 둘째 인코더)별 200문항 답변 — 공식 EM·형식 허용 정답 수,
같은 조건 안에서 본 방법 대비 McNemar(정확) p. 파일이 없거나 200행이 안 되면 '진행 중'으로 적는다. 사용: python sweep_summary.py [조건 ...]"""
import json, sys
from math import comb
from pathlib import Path
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from mh_arms import official_answers                                        # noqa: E402
from rag_agent.eval.multihiertt_em import format_match, mh_exact_match      # noqa: E402

R = Path("results"); S = R / "post_defense_20261006"; B = R / "mh_only_20260928/baselines"
IDS = [json.loads(l)["query_id"] for l in open(B / "ids200.jsonl", encoding="utf-8") if l.strip()]
COND = {
    "주 설정(Qwen3-8B 4bit, bge-base)": {
        "본 방법": R / "mh_arms/cap300_20260924/cell_uniq.jsonl", "LangChain 청크": B / "langchain_chunk/langchain_chunk_answer.jsonl",
        "RAPTOR 2000": B / "raptor/raptor_answer_2000.jsonl", "RAPTOR 575": B / "raptor/raptor_answer_575.jsonl",
        "TableRAG(Chen)": B / "tablerag_chen_mt512/tablerag_chen_answer.jsonl", "TableRAG(Yu)": B / "tablerag_yu/tablerag_yu_answer.jsonl"},
    "둘째 LLM(Qwen3-1.7B bf16, bge-base)": {
        "본 방법": S / "llm_qwen17/ours_answer.jsonl", "LangChain 청크": S / "llm_qwen17/langchain_chunk/langchain_chunk_answer.jsonl",
        "RAPTOR 2000": S / "llm_qwen17/raptor/raptor_answer_2000.jsonl", "RAPTOR 575": S / "llm_qwen17/raptor/raptor_answer_575.jsonl",
        "TableRAG(Chen)": S / "llm_qwen17/tablerag_chen/tablerag_chen_answer.jsonl", "TableRAG(Yu)": S / "llm_qwen17/tablerag_yu/tablerag_yu_answer.jsonl"},
    "둘째 인코더(bge-m3, Qwen3-8B 4bit)": {
        "본 방법": S / "enc_bgem3/ours_answer.jsonl", "LangChain 청크": S / "enc_bgem3/langchain_chunk/langchain_chunk_answer.jsonl",
        "RAPTOR 2000": S / "enc_bgem3/raptor/raptor_answer_2000.jsonl", "RAPTOR 575": S / "enc_bgem3/raptor/raptor_answer_575.jsonl",
        "TableRAG(Chen)": S / "enc_bgem3/tablerag_chen/tablerag_chen_answer.jsonl", "TableRAG(Yu)": S / "enc_bgem3/tablerag_yu/tablerag_yu_answer.jsonl"},
}
gold = official_answers("train")
def load(p):
    if not p.exists(): return None
    rows = {}
    for l in open(p, encoding="utf-8"):
        if not l.strip(): continue
        r = json.loads(l)
        if r["query_id"] not in set(IDS): continue
        prog = r["kind"] == "arith"; g = gold[r["query_id"]]
        rows[r["query_id"]] = (int(mh_exact_match(r["pred"], g, prog)), int(format_match(r["pred"], g, prog)))
    return rows
def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
want = sys.argv[1:] or list(COND)
for cond in want:
    print(f"\n## {cond}\n방법 | 문항 | 공식 EM | 형식 허용 | 허용 b:c(본 방법:비교군) | p")
    ours = load(COND[cond]["본 방법"])
    for name, p in COND[cond].items():
        rows = load(p)
        if rows is None or len(rows) < 200: print(f"{name} | 진행 중 {0 if rows is None else len(rows)}/200"); continue
        em, le = sum(v[0] for v in rows.values()), sum(v[1] for v in rows.values())
        if name == "본 방법" or ours is None or len(ours) < 200: print(f"{name} | {len(rows)} | {em} | {le} ({le / 2:.1f}%) | — | —"); continue
        b = sum(ours[q][1] and not rows[q][1] for q in IDS); c = sum(rows[q][1] and not ours[q][1] for q in IDS)
        print(f"{name} | {len(rows)} | {em} | {le} ({le / 2:.1f}%) | {b}:{c} | {mcnemar(b, c):.4f}")
