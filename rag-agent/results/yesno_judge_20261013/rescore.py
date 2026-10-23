"""예/아니오 사람 판정 반영 재채점 — PREREG-2026-10-13-yesno-judge.md.
규칙: 예/아니오 정답 문항에서 출력에 yes/no 낱말이 있으면 기존 규칙(첫 낱말 = 정답), 없으면 judgments.csv 의 judge. 나머지 문항은 그대로.
  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/yesno_judge_20261013/rescore.py
"""
import csv, json, re, sys
from pathlib import Path
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "results/cellform_20261012"), str(ROOT / "results/gpt4o_20261013")]
from mh_arms import official_answers                       # noqa: E402
from score_inclusion import gold_kind, included            # noqa: E402
import by_type                                             # noqa: E402  (qtype: PPT 표 유형 규칙)

HERE = Path(__file__).resolve().parent
G = official_answers("train")
J = {(r["query_id"], r["cond"]): int(r["judge"]) for r in csv.DictReader(open(HERE / "judgments.csv", encoding="utf-8"))}
COND = {"gpt_closed": "results/gpt4o_20261013/ans_closed", "gpt_fulldoc": "results/gpt4o_20261013/ans_fulldoc",
        "gpt_ours": "results/gpt4o_20261013/ans_ours", "gpt_langchain": "results/gpt4o_20261013/ans_langchain",
        "qwen_ours": "results/cellform_20261012/C_ours_budget", "qwen_langchain": "results/cellform_20261012/A_chunk_md"}


def score(r, cond):
    g = G[r["query_id"]]
    old = int(included(r["raw"], g, r["kind"] == "arith"))
    if gold_kind(g) != "yesno" or re.search(r"\b(yes|no)\b", r["raw"].lower()):
        return old, old
    return old, J[(r["query_id"], cond)]


def mcnemar(a, b, ids, key):
    bb = sum(a[q][key] and not b[q][key] for q in ids)
    cc = sum(b[q][key] and not a[q][key] for q in ids)
    return {"b": bb, "c": cc, "p": float(binomtest(min(bb, cc), bb + cc, 0.5).pvalue) if bb + cc else 1.0}


R = {}
for c, d in COND.items():
    rows = {}
    for r in map(json.loads, open(ROOT / d / "langchain_chunk_answer.jsonl", encoding="utf-8")):
        r["old"], r["new"] = score(r, c)
        rows[r["query_id"]] = r
    R[c] = rows
ids = sorted(R["gpt_ours"])
yn = [q for q in ids if gold_kind(G[q]) == "yesno"]
T = {q: by_type.qtype(R["gpt_ours"][q]) for q in ids}
out = {"n": len(ids), "yesno_n": len(yn), "conditions": {}, "by_type": {}, "mcnemar": {}}
print("| 조건 | 예/아니오 54 (전→후) | 전체 882 (전→후) | % 전→후 |")
print("|---|---|---|---|")
for c in COND:
    o54, n54 = sum(R[c][q]["old"] for q in yn), sum(R[c][q]["new"] for q in yn)
    o, n = sum(R[c][q]["old"] for q in ids), sum(R[c][q]["new"] for q in ids)
    out["conditions"][c] = {"yesno_old": o54, "yesno_new": n54, "old": o, "new": n, "pct_old": round(100 * o / 882, 1), "pct_new": round(100 * n / 882, 1)}
    print(f"| {c} | {o54}→{n54} | {o}→{n} | {100*o/882:.1f}→{100*n/882:.1f} |")
pairs = {"gpt_fulldoc_vs_ours": ("gpt_fulldoc", "gpt_ours"), "gpt_ours_vs_langchain": ("gpt_ours", "gpt_langchain"),
         "qwen_ours_vs_langchain": ("qwen_ours", "qwen_langchain"), "ours_gpt_vs_qwen": ("gpt_ours", "qwen_ours"), "langchain_gpt_vs_qwen": ("gpt_langchain", "qwen_langchain")}
for k, (a, b) in pairs.items():
    out["mcnemar"][k] = {"old": mcnemar(R[a], R[b], ids, "old"), "new": mcnemar(R[a], R[b], ids, "new")}
    print(k, "전", out["mcnemar"][k]["old"], "후", out["mcnemar"][k]["new"])
groups = {t: [q for q in ids if T[q] == t] for t in by_type.TYPES}
groups["전체"] = ids
print("\n| 쿼리 유형 | 문항 수 | 표 전부 | LangChain | 본 방법 | 본−LC | Qwen LC | Qwen 본 | Qwen 차이 |")
print("|---|---|---|---|---|---|---|---|---|")
for t, g in groups.items():
    pct = {c: round(100 * sum(R[c][q]["new"] for q in g) / len(g), 1) for c in COND}
    m1, m2 = mcnemar(R["gpt_ours"], R["gpt_langchain"], g, "new"), mcnemar(R["qwen_ours"], R["qwen_langchain"], g, "new")
    out["by_type"][t] = {"n": len(g), "pct_new": pct, "gpt_ours_vs_langchain": m1, "qwen_ours_vs_langchain": m2}
    print(f"| {t} | {len(g)} | {pct['gpt_fulldoc']} | {pct['gpt_langchain']} | {pct['gpt_ours']} | {pct['gpt_ours']-pct['gpt_langchain']:+.1f}({m1['b']}:{m1['c']} p={m1['p']:.3f}) | "
          f"{pct['qwen_langchain']} | {pct['qwen_ours']} | {pct['qwen_ours']-pct['qwen_langchain']:+.1f}({m2['b']}:{m2['c']} p={m2['p']:.3f}) |")
(HERE / "rescore.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
