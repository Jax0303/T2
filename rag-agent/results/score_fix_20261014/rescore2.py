"""재채점 v2 — PREREG-2026-10-14-score-fix.md. 전 = 10-13 예/아니오 판정 반영 점수(rescore.py 의 new), 후 = included_v2 + 같은 예/아니오 판정.
  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/score_fix_20261014/rescore2.py
"""
import csv, json, re, sys
from pathlib import Path
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "results/cellform_20261012"), str(ROOT / "results/gpt4o_20261013"), str(ROOT / "results/score_fix_20261014")]
from mh_arms import official_answers                       # noqa: E402
from score_inclusion import gold_kind, included            # noqa: E402
from score_v2 import included_v2, included_sens                           # noqa: E402
import by_type                                             # noqa: E402

HERE = Path(__file__).resolve().parent
G = official_answers("train")
J = {(r["query_id"], r["cond"]): int(r["judge"]) for r in csv.DictReader(open(ROOT / "results/yesno_judge_20261013/judgments.csv", encoding="utf-8"))}
COND = {"gpt_closed": "results/gpt4o_20261013/ans_closed", "gpt_fulldoc": "results/gpt4o_20261013/ans_fulldoc",
        "gpt_ours": "results/gpt4o_20261013/ans_ours", "gpt_langchain": "results/gpt4o_20261013/ans_langchain",
        "qwen_ours": "results/cellform_20261012/C_ours_budget", "qwen_langchain": "results/cellform_20261012/A_chunk_md"}
NAME = {"gpt_closed": "GPT-4o 질문만", "gpt_fulldoc": "GPT-4o 표 전부", "gpt_ours": "GPT-4o 본 방법", "gpt_langchain": "GPT-4o LangChain",
        "qwen_ours": "Qwen3-8B 본 방법", "qwen_langchain": "Qwen3-8B LangChain"}


def score(r, cond, fn):
    g = G[r["query_id"]]
    s = int(fn(r["raw"], g, r["kind"] == "arith"))
    if gold_kind(g) == "yesno" and not re.search(r"\b(yes|no)\b", r["raw"].lower()):
        return J[(r["query_id"], cond)]
    return s


def mcnemar(a, b, ids, key):
    bb = sum(a[q][key] and not b[q][key] for q in ids)
    cc = sum(b[q][key] and not a[q][key] for q in ids)
    return {"b": bb, "c": cc, "p": float(binomtest(min(bb, cc), bb + cc, 0.5).pvalue) if bb + cc else 1.0}


R = {}
for c, d in COND.items():
    rows = {}
    for r in map(json.loads, open(ROOT / d / "langchain_chunk_answer.jsonl", encoding="utf-8")):
        r["old"], r["new"], r["sens"] = score(r, c, included), score(r, c, included_v2), score(r, c, included_sens)
        rows[r["query_id"]] = r
    R[c] = rows
ids = sorted(R["gpt_ours"])
# 자기 점검
sc = sum(included_v2(f"Based on the table, the answer is {G[q]}.", G[q], R["gpt_ours"][q]["kind"] == "arith") for q in ids)
dk = sum(included_v2("I don't know.", G[q], R["gpt_ours"][q]["kind"] == "arith") for q in ids)
print("self_check gold-in-sentence", sc, "/", len(ids), " dont_know", dk)
T = {q: by_type.qtype(R["gpt_ours"][q]) for q in ids}
out = {"n": len(ids), "self_check": {"gold_in_sentence": sc, "dont_know": dk}, "conditions": {}, "by_type": {}, "mcnemar": {}}
print("\n| 조건 | 882 전→후 | % 전→후 | 바뀐 문항(0→1 / 1→0) | 민감도(반올림 허용) |")
print("|---|---|---|---|---|")
for c in COND:
    o, n = sum(R[c][q]["old"] for q in ids), sum(R[c][q]["new"] for q in ids)
    up = sum(R[c][q]["new"] and not R[c][q]["old"] for q in ids); dn = sum(R[c][q]["old"] and not R[c][q]["new"] for q in ids)
    se = sum(R[c][q]["sens"] for q in ids)
    out["conditions"][c] = {"old": o, "new": n, "pct_old": round(100 * o / 882, 1), "pct_new": round(100 * n / 882, 1), "up": up, "down": dn, "sens": se, "pct_sens": round(100 * se / 882, 1)}
    print(f"| {NAME[c]} | {o}→{n} | {100*o/882:.1f}→{100*n/882:.1f} | {up} / {dn} | {se} ({100*se/882:.1f}) |")
pairs = {"gpt_fulldoc_vs_ours": ("gpt_fulldoc", "gpt_ours"), "gpt_ours_vs_langchain": ("gpt_ours", "gpt_langchain"),
         "qwen_ours_vs_langchain": ("qwen_ours", "qwen_langchain"), "ours_gpt_vs_qwen": ("gpt_ours", "qwen_ours"), "langchain_gpt_vs_qwen": ("gpt_langchain", "qwen_langchain")}
print()
for k, (a, b) in pairs.items():
    out["mcnemar"][k] = {"old": mcnemar(R[a], R[b], ids, "old"), "new": mcnemar(R[a], R[b], ids, "new"), "sens": mcnemar(R[a], R[b], ids, "sens")}
    print(k, "전", out["mcnemar"][k]["old"], "후", out["mcnemar"][k]["new"], "민감도", out["mcnemar"][k]["sens"])
groups = {t: [q for q in ids if T[q] == t] for t in by_type.TYPES}
groups["전체"] = ids
print("\n| 쿼리 유형 | 문항 수 | 표 전부 | LangChain | 본 방법 | 본−LC | Qwen LC | Qwen 본 | Qwen 차이 |")
print("|---|---|---|---|---|---|---|---|---|")
for t, g in groups.items():
    pct = {c: round(100 * sum(R[c][q]["new"] for q in g) / len(g), 1) for c in COND}
    pold = {c: round(100 * sum(R[c][q]["old"] for q in g) / len(g), 1) for c in COND}
    m1, m2 = mcnemar(R["gpt_ours"], R["gpt_langchain"], g, "new"), mcnemar(R["qwen_ours"], R["qwen_langchain"], g, "new")
    out["by_type"][t] = {"n": len(g), "pct_old": pold, "pct_new": pct, "gpt_ours_vs_langchain": m1, "qwen_ours_vs_langchain": m2}
    print(f"| {t} | {len(g)} | {pold['gpt_fulldoc']}→{pct['gpt_fulldoc']} | {pold['gpt_langchain']}→{pct['gpt_langchain']} | {pold['gpt_ours']}→{pct['gpt_ours']} | {pct['gpt_ours']-pct['gpt_langchain']:+.1f}({m1['b']}:{m1['c']} p={m1['p']:.3f}) | "
          f"{pold['qwen_langchain']}→{pct['qwen_langchain']} | {pold['qwen_ours']}→{pct['qwen_ours']} | {pct['qwen_ours']-pct['qwen_langchain']:+.1f}({m2['b']}:{m2['c']} p={m2['p']:.3f}) |")
# 바뀐 문항 목록
with open(HERE / "changed.tsv", "w", encoding="utf-8") as f:
    f.write("cond\tquery_id\told\tnew\tgold\traw\n")
    for c in COND:
        for q in ids:
            r = R[c][q]
            if r["old"] != r["new"]:
                f.write(f"{c}\t{q}\t{r['old']}\t{r['new']}\t{G[q]}\t{(r['raw'] or '')[:200].replace(chr(10),' ')}\n")
(HERE / "rescore2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
