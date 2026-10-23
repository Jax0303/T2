"""순서 시험 채점 — PREREG-2026-10-15-prompt-order.md. 같은 200문항: Question-first(기존 882 출력) 대 Context-first(새 출력), 조건별 McNemar.
예/아니오 정답에 yes/no 낱말 없는 출력: 기존 출력은 10-13 판정표, 새 출력은 judgments_cf.csv(없으면 미판정으로 출력해 둠).
  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/order_pilot_20261015/score_cf.py
"""
import csv, json, re, sys
from pathlib import Path
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "results/cellform_20261012"), str(ROOT / "results/score_fix_20261014"), str(ROOT / "results/gpt4o_20261013")]
from mh_arms import official_answers                       # noqa: E402
from score_inclusion import gold_kind                      # noqa: E402
from score_v2 import included_v2                           # noqa: E402
import by_type                                             # noqa: E402

HERE = Path(__file__).resolve().parent
G = official_answers("train")
ids = [json.loads(l)["query_id"] for l in open(HERE / "ids200.jsonl") if l.strip()]
J = {(r["query_id"], r["cond"]): int(r["judge"]) for r in csv.DictReader(open(ROOT / "results/yesno_judge_20261013/judgments.csv", encoding="utf-8"))}
if (HERE / "judgments_cf.csv").exists():
    J.update({(r["query_id"], r["cond"]): int(r["judge"]) for r in csv.DictReader(open(HERE / "judgments_cf.csv", encoding="utf-8"))})
COND = {"qwen_ours": ("results/cellform_20261012/C_ours_budget", "qwen_ours_cf", "results/order_pilot_20261015/C_ours_cf"),
        "qwen_langchain": ("results/cellform_20261012/A_chunk_md", "qwen_langchain_cf", "results/order_pilot_20261015/A_chunk_cf")}
load = lambda d: {r["query_id"]: r for r in map(json.loads, open(ROOT / d / "langchain_chunk_answer.jsonl", encoding="utf-8"))}
pending = []


def score(r, cond):
    g = G[r["query_id"]]
    if gold_kind(g) == "yesno" and not re.search(r"\b(yes|no)\b", r["raw"].lower()):
        if (r["query_id"], cond) not in J:
            pending.append((cond, r["query_id"], g, r["raw"].replace("\n", " ")))
            return 0
        return J[(r["query_id"], cond)]
    return int(included_v2(r["raw"], g, r["kind"] == "arith"))


def mc(a, b):
    bb = sum(x and not y for x, y in zip(a, b)); cc = sum(y and not x for x, y in zip(a, b))
    return bb, cc, (float(binomtest(min(bb, cc), bb + cc, 0.5).pvalue) if bb + cc else 1.0)


R = {}
for c, (d_old, c_new, d_new) in COND.items():
    old, new = load(d_old), load(d_new)
    miss = [q for q in ids if q not in new]
    if miss:
        print(f"{c}: 새 출력 없음 {len(miss)}건 — 실행 중?"); 
    R[c] = {"old": [score(old[q], c) for q in ids if q in new], "new": [score(new[q], c_new) for q in ids if q in new], "n": len(ids) - len(miss)}
    T = [by_type.qtype(old[q]) for q in ids if q in new]
    R[c]["types"] = T
print("| 조건 | 문항 | Question-first | Context-first | 전→후 바뀜(0→1 : 1→0) | p |")
print("|---|---|---|---|---|---|")
for c, r in R.items():
    bb, cc, p = mc(r["old"], r["new"])
    print(f"| {c} | {r['n']} | {sum(r['old'])} ({100*sum(r['old'])/r['n']:.1f}) | {sum(r['new'])} ({100*sum(r['new'])/r['n']:.1f}) | {cc} : {bb} | {p:.3f} |")
if all(R[c]["n"] == 200 for c in R):
    o, l = R["qwen_ours"], R["qwen_langchain"]
    for key, name in (("old", "Question-first"), ("new", "Context-first")):
        bb, cc, p = mc(o[key], l[key])
        print(f"본 방법 대 LangChain ({name}): {sum(o[key])} 대 {sum(l[key])} ({bb}:{cc} p={p:.3f})")
    print("\n유형별(본 방법 전→후 / LangChain 전→후):")
    for t in by_type.TYPES:
        idx = [i for i, x in enumerate(o["types"]) if x == t]
        if idx:
            print(f"  {t} ({len(idx)}): {sum(o['old'][i] for i in idx)}→{sum(o['new'][i] for i in idx)} / {sum(l['old'][i] for i in idx)}→{sum(l['new'][i] for i in idx)}")
if pending:
    print(f"\n미판정 예/아니오 출력 {len(pending)}건(0 처리됨) → judgments_cf.csv 에 적을 것:")
    for cond, q, g, raw in pending:
        print(f"  {cond}\t{q}\t정답 {g}\t{raw[:160]}")
