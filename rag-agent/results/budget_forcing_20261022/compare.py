"""잘린 답변 마무리(budget forcing) 전후 — 집합·입력·실행별 맞힘, D 대 A 전후 McNemar. PREREG-2026-10-22-budget-forcing.md"""
import json, sys
from pathlib import Path
from scipy.stats import binomtest
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "results/format_dev_20261017"), str(ROOT), str(ROOT / "scripts")]
import score_formats2 as sf
from mh_arms import official_answers
G = official_answers("train"); GD = {r["query_id"]: r["answer"] for r in map(json.loads, open(ROOT / "results/prompt_sweep_20261015/dev_A_records.jsonl", encoding="utf-8"))}
jobs = json.load(open(HERE / "jobs.json")); name = {"D": "잘라낸 표", "C": "셀 문장 목록", "Cp": "셀 문장 목록", "A": "LangChain 청크"}
def load(p, gold, key):
    return {r["query_id"]: (sf.score(r, gold[r["query_id"]])[key], r.get("forced", 0)) for r in map(json.loads, open(p, encoding="utf-8"))}
by = {}
for j in jobs:
    gold = GD if j["set"] == "dev" else G; key = "inc" if j["variant"].endswith("C0E0") else "fin_inc"
    b = load(j["out"], gold, key); f = load(Path(j["out"]).with_suffix(".forced.jsonl"), gold, key)
    by.setdefault((j["set"], j["variant"]), {}).setdefault(j["run"], {})[j["arm"]] = (b, f)
for (s, v), runs in sorted(by.items()):
    print(f"\n== {s} {v} ==\n| 입력 | 실행 | 전 | 후 | 늘어남 | 덧붙인 수 |\n|---|---|---|---|---|---|")
    tot = {}
    for run in sorted(runs):
        for a, (b, f) in runs[run].items():
            nb = sum(x for x, _ in b.values()); nf = sum(x for x, _ in f.values()); nforce = sum(k for _, k in f.values())
            print(f"| {name[a]} | {run} | {nb} | {nf} | {nf-nb:+d} | {nforce} |"); tot.setdefault(a, []).append(nf - nb)
    print("  3회 평균 늘어남: " + ", ".join(f"{name[a]} {sum(v_)/len(v_):+.1f}" for a, v_ in tot.items()))
    for run in sorted(runs):
        R = runs[run]
        if "D" in R and "A" in R:
            for lab, k in (("전", 0), ("후", 1)):
                d = R["D"][k]; a_ = R["A"][k]; qs = sorted(set(d) & set(a_))
                bb = sum(d[q][0] and not a_[q][0] for q in qs); cc = sum(a_[q][0] and not d[q][0] for q in qs); p = binomtest(min(bb, cc), bb + cc, 0.5).pvalue if bb + cc else 1
                print(f"  {run} 잘라낸 표 대 LangChain {lab}: {sum(d[q][0] for q in qs)} 대 {sum(a_[q][0] for q in qs)} b:c {bb}:{cc} p={p:.3f}")
