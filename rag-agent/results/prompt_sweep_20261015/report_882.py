"""882 확인 집계 — PREREG-2026-10-15-prompt-sweep.md (판정 규칙은 사전등록대로, 결과 전에 작성).
조건 = <arm>:<variant> (파일 <out-dir>/<arm>/<variant>.jsonl). 채점 = 정답 포함 v2 + 예/아니오 낱말 없는 출력은 판정표(<out-dir>/judgments.csv: query_id,cond,judge)
— 판정표에 없으면 낱말 규칙(0)으로 두고 "판정 필요" 건수를 센다. 쌍 비교 = 같은 문항 정확 McNemar, --holm 묶음은 Holm.
  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/prompt_sweep_20261015/report_882.py <out-dir> \
      --pair main=C:R0X1F1B1D1C2E1,A:R0X1F1B1D1C2E1 --pair orig=C:R0X0F0B0D0C0E0,A:R0X0F0B0D0C0E0 --holm main,orig ...
"""
import argparse, csv, json, re, sys
from collections import defaultdict
from pathlib import Path
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "results/cellform_20261012"), str(ROOT / "results/score_fix_20261014"),
                str(ROOT / "results/gpt4o_20261013")]
from answer_accuracy_mh import extract                                  # noqa: E402
from mh_arms import official_answers                                    # noqa: E402
from rag_agent.eval.multihiertt_em import mh_exact_match                # noqa: E402
from score_inclusion import gold_kind                                   # noqa: E402
from score_v2 import included_v2                                        # noqa: E402
import by_type                                                          # noqa: E402  (유형 규칙 = 10-13 PPT 근사)

G = official_answers("train")
Q = {r["query_id"]: r for r in map(json.loads, open(ROOT / "results/cellform_20261012/A_chunk_md_records.jsonl", encoding="utf-8"))}
REF = {"Qwen3-8B 4bit 본 방법(원문)": ROOT / "results/cellform_20261012/C_ours_budget/langchain_chunk_answer.jsonl",
       "Qwen3-8B 4bit LangChain(원문)": ROOT / "results/cellform_20261012/A_chunk_md/langchain_chunk_answer.jsonl",
       "GPT-4o 본 방법(원문)": ROOT / "results/gpt4o_20261013/ans_ours/langchain_chunk_answer.jsonl",
       "GPT-4o LangChain(원문)": ROOT / "results/gpt4o_20261013/ans_langchain/langchain_chunk_answer.jsonl"}
REF_COND = {"Qwen3-8B 4bit 본 방법(원문)": "qwen_ours", "Qwen3-8B 4bit LangChain(원문)": "qwen_langchain",
            "GPT-4o 본 방법(원문)": "gpt_ours", "GPT-4o LangChain(원문)": "gpt_langchain"}


def load(p, cond, J, need):
    rows = {}
    for r in map(json.loads, open(p, encoding="utf-8")):
        q, g, arith = r["query_id"], G[r["query_id"]], r["kind"] == "arith"
        s = int(included_v2(r["raw"], g, arith))
        if gold_kind(g) == "yesno" and not re.search(r"\b(yes|no)\b", r["raw"].lower()):
            if (q, cond) in J:
                s = J[(q, cond)]
            else:
                need.append((q, cond))
        pred, _ = extract(r["raw"])
        rows[q] = {"inc": s, "em": int(mh_exact_match(pred, g, arith)), "kind": r["kind"], "layer": r["layer"], "m": r["m"],
                   "question": Q[q]["question"], "query_id": q, "fin": int(r.get("finished", r.get("finish_reason") != "length")),
                   "n_out": r.get("n_tok_out"), "dk": int("don't know" in r["raw"].lower() or "do not know" in r["raw"].lower())}
    return rows


def mcnemar(a, b, ids):
    bb = sum(a[q]["inc"] and not b[q]["inc"] for q in ids)
    cc = sum(b[q]["inc"] and not a[q]["inc"] for q in ids)
    return {"n": len(ids), "b": bb, "c": cc, "p": float(binomtest(min(bb, cc), bb + cc, 0.5).pvalue) if bb + cc else 1.0}


def holm(d):
    order = sorted(d, key=lambda k: d[k]["p"])
    run = 0.0
    for k, name in enumerate(order):
        run = max(run, min(1.0, (len(d) - k) * d[name]["p"]))
        d[name]["p_holm"] = run
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--pair", action="append", default=[], help="이름=조건1,조건2 (조건 = arm:variant 또는 참고 행 이름)")
    ap.add_argument("--holm", default="", help="Holm 으로 묶을 쌍 이름(쉼표)")
    a = ap.parse_args()
    D = Path(a.out_dir)
    J = {}
    if (D / "judgments.csv").exists():
        J = {(r["query_id"], r["cond"]): int(r["judge"]) for r in csv.DictReader(open(D / "judgments.csv", encoding="utf-8"))}
    need = []
    R = {}
    for arm_dir in sorted(p for p in D.iterdir() if p.is_dir()):
        for p in sorted(arm_dir.glob("*.jsonl")):
            R[f"{arm_dir.name}:{p.stem}"] = load(p, f"{arm_dir.name}:{p.stem}", J, need)
    JR = {}
    if (ROOT / "results/yesno_judge_20261013/judgments.csv").exists():
        JR = {(r["query_id"], r["cond"]): int(r["judge"]) for r in csv.DictReader(open(ROOT / "results/yesno_judge_20261013/judgments.csv", encoding="utf-8"))}
    for name, p in REF.items():
        if p.exists():
            R[name] = load(p, REF_COND[name], JR, [])
    ids = sorted(Q)
    out = {"n": len(ids), "judgments_needed": len(need), "conditions": {}, "pairs": {}}
    print(f"| 조건 | 문항 | 정답 포함 | % | 조회 | 산술 | 공식 EM | 모른다 | 잘림 | 출력 토큰 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for c, rows in R.items():
        if len(rows) < len(ids):
            print(f"| {c} | {len(rows)}/882 진행 중 |"); continue
        v = [rows[q] for q in ids]
        s = {"inc": sum(x["inc"] for x in v), "pct": round(100 * sum(x["inc"] for x in v) / len(v), 1),
             "lookup": sum(x["inc"] for x in v if x["kind"] == "lookup"), "arith": sum(x["inc"] for x in v if x["kind"] == "arith"),
             "em": round(sum(x["em"] for x in v) / len(v), 4), "dont_know": sum(x["dk"] for x in v), "truncated": sum(1 - x["fin"] for x in v),
             "out_tok": round(sum(x["n_out"] or 0 for x in v) / len(v), 1),
             "by_layer": {L: sum(x["inc"] for x in v if x["layer"] == L) for L in ("lookup_m1", "lookup_m2+", "arith_m1", "arith_m2+")}}
        out["conditions"][c] = s
        print(f"| {c} | 882 | {s['inc']} | {s['pct']} | {s['lookup']}/511 | {s['arith']}/371 | {s['em']} | {s['dont_know']} | {s['truncated']} | {s['out_tok']} |")
    pairs = {}
    for spec in a.pair:
        name, cs = spec.split("=", 1)
        c1, c2 = cs.split(",", 1)
        if c1 in R and c2 in R and len(R[c1]) == len(R[c2]) == len(ids):
            pairs[name] = (c1, c2)
    hol = set(a.holm.split(",")) if a.holm else set()
    res = {k: mcnemar(R[c1], R[c2], ids) for k, (c1, c2) in pairs.items()}
    holm({k: v for k, v in res.items() if k in hol})
    for k, (c1, c2) in pairs.items():
        r = res[k]
        r["by_kind"] = {kind: mcnemar(R[c1], R[c2], [q for q in ids if R[c1][q]["kind"] == kind]) for kind in ("lookup", "arith")}
        out["pairs"][k] = {"conditions": [c1, c2], **r}
        print(f"\n[{k}] {c1} 대 {c2}: {out['conditions'][c1]['inc']} 대 {out['conditions'][c2]['inc']}, b:c {r['b']}:{r['c']} p={r['p']:.4f}"
              + (f" Holm p={r['p_holm']:.4f}" if "p_holm" in r else "") + f" | 조회 {r['by_kind']['lookup']['b']}:{r['by_kind']['lookup']['c']} p={r['by_kind']['lookup']['p']:.3f}"
              f" 산술 {r['by_kind']['arith']['b']}:{r['by_kind']['arith']['c']} p={r['by_kind']['arith']['p']:.3f}")
    # 유형별(10-13 PPT 근사 규칙)
    T = {q: by_type.qtype(R[next(iter(R))][q]) for q in ids} if R else {}
    groups = {t: [q for q in ids if T[q] == t] for t in by_type.TYPES}
    groups["전체"] = ids
    conds = [c for c in R if len(R[c]) == len(ids)]
    out["by_type"] = {}
    print("\n| 쿼리 유형 | 문항 수 | " + " | ".join(conds) + " |")
    print("|---|---|" + "---|" * len(conds))
    for t, g in groups.items():
        pct = {c: round(100 * sum(R[c][q]["inc"] for q in g) / len(g), 1) for c in conds}
        out["by_type"][t] = {"n": len(g), "pct": pct,
                             "pairs": {k: mcnemar(R[c1], R[c2], g) for k, (c1, c2) in pairs.items()}}
        print(f"| {t} | {len(g)} | " + " | ".join(str(pct[c]) for c in conds) + " |")
    if need:
        print(f"\n판정 필요(예/아니오 낱말 없음): {len(need)}건 → {D / 'to_judge.csv'}")
        with open(D / "to_judge.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(["query_id", "cond", "question", "gold", "raw"])
            for q, cond in need:
                arm, var = cond.split(":", 1)
                raw = next(r["raw"] for r in map(json.loads, open(D / arm / f"{var}.jsonl", encoding="utf-8")) if r["query_id"] == q)
                w.writerow([q, cond, Q[q]["question"], G[q], raw])
    (D / "report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
