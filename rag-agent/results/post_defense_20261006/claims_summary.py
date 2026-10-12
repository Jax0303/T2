"""PREREG-2026-10-07-claims.md 1~4 집계 — 공식 EM·형식 허용(주), 같은 문항 McNemar 정확 p.
  1 지시문×단위 2×2 (882·200)  2 작은 LLM 문서 전체 대 20셀  3 1.7B bf16 대 4비트  4 잡음 사다리·정답 위치.
  PYTHONPATH=. .venv/bin/python results/post_defense_20261006/claims_summary.py"""
import json, sys
from math import comb
from pathlib import Path
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from mh_arms import official_answers                                        # noqa: E402
from rag_agent.eval.multihiertt_em import format_match, mh_exact_match      # noqa: E402

R = Path("results"); S = R / "post_defense_20261006"; B = R / "mh_only_20260928/baselines"
IDS200 = [json.loads(l)["query_id"] for l in open(B / "ids200.jsonl", encoding="utf-8") if l.strip()]
G = official_answers("train")

def load(p, ids=None):
    if not Path(p).exists(): return None
    d = {}
    for l in open(p, encoding="utf-8"):
        if not l.strip(): continue
        r = json.loads(l); q = r["query_id"]
        if ids is not None and q not in ids: continue
        prog = r["kind"] == "arith"
        d[q] = (int(mh_exact_match(r["pred"], G[q], prog)), int(format_match(r["pred"], G[q], prog)), r["kind"], r.get("n_tok"))
    return d

def mcnemar(b, c):
    n = b + c; return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)

def cell(name, d, n_expect=None):
    if d is None: return f"{name}: 없음"
    tag = f" (진행 중 {len(d)}/{n_expect})" if n_expect and len(d) != n_expect else ""
    em, le = sum(v[0] for v in d.values()), sum(v[1] for v in d.values())
    lk = sum(v[1] for v in d.values() if v[2] == "lookup"); ar = sum(v[1] for v in d.values() if v[2] == "arith")
    tok = [v[3] for v in d.values() if v[3]]
    return f"{name}: 허용 {le}/{len(d)} ({100 * le / len(d):.1f}%) 공식 {em} 조회/산술 {lk}/{ar} 토큰 {sum(tok) / len(tok):.0f}{tag}"

def pair(a, b, na, nb, key=1):
    if a is None or b is None: return f"  {na} 대 {nb}: 없음"
    common = sorted(set(a) & set(b))
    bb = sum(a[q][key] and not b[q][key] for q in common); cc = sum(b[q][key] and not a[q][key] for q in common)
    return (f"  {na} 대 {nb} ({len(common)}문항): {sum(a[q][key] for q in common)} 대 {sum(b[q][key] for q in common)}, "
            f"b:c {bb}:{cc}, p={mcnemar(bb, cc):.4f}")

print("## 1. 지시문 × 단위 2×2 (형식 허용; 괄호 공식 EM)")
ours882 = load(R / "mh_arms/cap300_20260924/cell_uniq.jsonl")
lc882 = load(B / "langchain_chunk/langchain_chunk_answer.jsonl")
cells_lc = load(S / "prompt2x2/cells_lcprompt/answer.jsonl")
chunks_cot = load(S / "prompt2x2/langchain_cot/langchain_chunk_answer.jsonl")
for scope, ids in [("882", None), ("200", set(IDS200))]:
    sub = (lambda d: None if d is None else {q: v for q, v in d.items() if ids is None or q in ids})
    A, Bv, C, D = map(sub, (ours882, cells_lc, chunks_cot, lc882))
    n = 882 if ids is None else 200
    print(f"[{scope}문항]")
    print(" ", cell("셀 문장+본 방법 지시문", A, n)); print(" ", cell("셀 문장+LangChain 지시문", Bv, n))
    print(" ", cell("청크+본 방법 지시문", C, n)); print(" ", cell("청크+LangChain 지시문", D, n))
    print(" 단위 효과(주 비교):"); print(pair(A, C, "셀 문장", "청크", ) + "  [본 방법 지시문]"); print(pair(Bv, D, "셀 문장", "청크") + "  [LangChain 지시문]")
    print(" 지시문 효과(보고만):"); print(pair(A, Bv, "본 방법 지시문", "LangChain 지시문") + "  [셀 문장]"); print(pair(C, D, "본 방법 지시문", "LangChain 지시문") + "  [청크]")

print("\n## 2. 작은 LLM: 문서 전체 셀 문장 대 20셀 (200)")
for name, full, twenty in [("Qwen3-1.7B bf16", S / "fullcells_small/qwen17_fullcells.jsonl", S / "llm_qwen17/ours_answer.jsonl"),
                           ("Llama-3.2-3B bf16", S / "fullcells_small/llama3b_fullcells.jsonl", S / "llm_llama3b/ours_answer.jsonl"),
                           ("Qwen3-8B 4비트(기준)", S / "claim/B2_fullcells_answer.jsonl", R / "mh_arms/cap300_20260924/cell_uniq.jsonl")]:
    F, T = load(full), load(twenty, set(IDS200))
    print(f" {name}:"); print("  ", cell("문서 전체", F, 200)); print("  ", cell("20셀", T, 200)); print(pair(T, F, "20셀", "문서 전체"))

print("\n## 3. Qwen3-1.7B bf16 대 4비트 (200)")
for name, bf, q4 in [("본 방법", S / "llm_qwen17/ours_answer.jsonl", S / "llm_qwen17_4bit/ours_answer.jsonl"),
                     ("LangChain", S / "llm_qwen17/langchain_chunk/langchain_chunk_answer.jsonl", S / "llm_qwen17_4bit/langchain_chunk/langchain_chunk_answer.jsonl")]:
    Bf, Q = load(bf), load(q4)
    print(f" {name}:"); print("  ", cell("bf16", Bf, 200)); print("  ", cell("4비트", Q, 200)); print(pair(Bf, Q, "bf16", "4비트"))

print("\n## 4. 잡음 사다리·정답 위치 (200, 8B; 기준 정답 셀만(부록 D) 133)")
gold = load(R / "gold_only_20260929/mh_gold_cot384.jsonl", set(IDS200))
base = load(S / "noise/noise_0_top.jsonl")
print(" ", cell("부록 D 정답 셀만(caption_sentence)", gold, 200))
for n, pos in [(0, "top"), (5, "top"), (10, "top"), (20, "top"), (50, "top"), (20, "mid"), (20, "bottom")]:
    d = load(S / f"noise/noise_{n}_{pos}.jsonl")
    print(" ", cell(f"정답+비정답 {n} ({pos})", d, 200), "" if d is None or base is None or (n, pos) == (0, "top") else pair(base, d, "N=0", f"N={n} {pos}").strip())
