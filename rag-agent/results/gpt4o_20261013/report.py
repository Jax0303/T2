"""GPT-4o 확인 실험 집계 — PREREG-2026-10-13-gpt4o.md (판정 규칙은 사전등록대로, 결과 전에 작성).

조건: closed(질문만) / fulldoc(표 전부) / ours(본 방법 RAG 출력) / langchain(LangChain RAG 출력), 각 882.
보조: 같은 입력을 Qwen3-8B 4비트로 돌린 결과(results/cellform_20261012/ans C·A).
채점(주) = 정답 포함(score_inclusion.included). 보조 = 공식 EM/F1(행에 저장됨), 마지막 숫자 채점.
  PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python results/gpt4o_20261013/report.py
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "results/cellform_20261012")]
from mh_arms import official_answers                                  # noqa: E402
from score_inclusion import gold_kind, included, last_num             # noqa: E402

D = ROOT / "results/gpt4o_20261013"
C = ROOT / "results/cellform_20261012"
FILES = {"closed": D / "ans_closed/langchain_chunk_answer.jsonl",
         "fulldoc": D / "ans_fulldoc/langchain_chunk_answer.jsonl",
         "ours": D / "ans_ours/langchain_chunk_answer.jsonl",
         "langchain": D / "ans_langchain/langchain_chunk_answer.jsonl",
         "qwen8b4_ours": C / "C_ours_budget/langchain_chunk_answer.jsonl",
         "qwen8b4_langchain": C / "A_chunk_md/langchain_chunk_answer.jsonl"}
G = official_answers("train")
LAYERS = ("lookup_m1", "lookup_m2+", "arith_m1", "arith_m2+")


def load(p):
    rows = {}
    for r in map(json.loads, open(p, encoding="utf-8")):
        arith = r["kind"] == "arith"
        r["inc"] = int(included(r["raw"], G[r["query_id"]], arith))
        r["last"] = int(last_num(r["raw"], G[r["query_id"]], arith))
        r["dont_know"] = int("don't know" in r["raw"].lower() or "do not know" in r["raw"].lower())
        rows[r["query_id"]] = r
    return rows


def mcnemar(a, b, ids):
    """b = a 만 맞힘, c = b 만 맞힘, 정확 이항(results/stats_20260926/stats.py 와 같은 계산)."""
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


def summarize(rows, ids):
    v = [rows[q] for q in ids]
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    by = defaultdict(list)
    for r in v:
        by[r["layer"]].append(r["inc"])
    return {"n": len(v), "inclusion": sum(r["inc"] for r in v), "inclusion_pct": round(100 * mean([r["inc"] for r in v]), 1),
            "lookup": sum(r["inc"] for r in v if r["kind"] == "lookup"), "lookup_n": sum(r["kind"] == "lookup" for r in v),
            "arith": sum(r["inc"] for r in v if r["kind"] == "arith"), "arith_n": sum(r["kind"] == "arith" for r in v),
            "by_layer": {k: {"n": len(by[k]), "inclusion": sum(by[k]), "pct": round(100 * mean(by[k]), 1)} for k in LAYERS if by[k]},
            "last_number": sum(r["last"] for r in v), "official_em": mean([r["answer_correct"] for r in v]),
            "official_f1": mean([r["f1"] for r in v]), "dont_know": sum(r["dont_know"] for r in v),
            "input_tokens_mean": mean([r["n_tok"] for r in v if r.get("n_tok") is not None]),
            "output_tokens_mean": mean([r["n_tok_out"] for r in v if r.get("n_tok_out") is not None]),
            "finish_length": sum(r.get("finish_reason") == "length" for r in v)}


def main():
    R = {k: load(p) for k, p in FILES.items() if p.exists()}
    ids = sorted(set.intersection(*(set(v) for v in R.values())))
    full = all(len(v) == 882 for v in R.values())
    out = {"n_common": len(ids), "complete_882": full, "conditions": {k: summarize(v, ids) for k, v in R.items()}}
    if "closed" in R:                                                   # ① 정답 유형별(예/아니오는 찍어도 50%)
        kinds = Counter(gold_kind(G[q]) for q in ids)
        out["closed_by_gold_kind"] = {k: {"n": kinds[k], "inclusion": sum(R["closed"][q]["inc"] for q in ids if gold_kind(G[q]) == k)}
                                      for k in ("yesno", "num", "text")}
    pairs = {"fulldoc_vs_ours": ("fulldoc", "ours"), "ours_vs_langchain": ("ours", "langchain")}
    out["mcnemar_holm"] = holm({k: mcnemar(R[a], R[b], ids) for k, (a, b) in pairs.items() if a in R and b in R})
    aux = {"ours_gpt4o_vs_qwen8b4": ("ours", "qwen8b4_ours"), "langchain_gpt4o_vs_qwen8b4": ("langchain", "qwen8b4_langchain")}
    out["mcnemar_aux"] = {k: mcnemar(R[a], R[b], ids) for k, (a, b) in aux.items() if a in R and b in R}
    out["mcnemar_by_kind"] = {k: {kind: mcnemar(R[a], R[b], [q for q in ids if R[a][q]["kind"] == kind]) for kind in ("lookup", "arith")}
                              for k, (a, b) in {**pairs, **aux}.items() if a in R and b in R}
    three = [k for k in ("fulldoc", "ours", "langchain") if k in R]
    if len(three) == 3:
        wrong = [q for q in ids if not any(R[k][q]["inc"] for k in three)]
        out["all_three_wrong"] = {"n": len(wrong), "by_kind": dict(Counter(R["ours"][q]["kind"] for q in wrong)),
                                  "by_layer": dict(Counter(R["ours"][q]["layer"] for q in wrong))}
    (D / "report.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"공통 문항 {len(ids)} (882 완료: {full})")
    print("| 조건 | 정답 포함 | % | 조회 | 산술 | 공식 EM | 모른다 | 입력 토큰 | 출력 토큰 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k, s in out["conditions"].items():
        print(f"| {k} | {s['inclusion']} | {s['inclusion_pct']} | {s['lookup']}/{s['lookup_n']} | {s['arith']}/{s['arith_n']} | "
              f"{s['official_em']} | {s['dont_know']} | {s['input_tokens_mean']} | {s['output_tokens_mean']} |")
    for k, s in out["conditions"].items():
        print(k, " ".join(f"{l}={v['inclusion']}/{v['n']}({v['pct']})" for l, v in s["by_layer"].items()))
    print("McNemar(Holm):", json.dumps(out["mcnemar_holm"]))
    print("보조:", json.dumps(out["mcnemar_aux"]))
    print("질문만 정답 유형별:", json.dumps(out.get("closed_by_gold_kind")))
    print("셋 다 오답:", json.dumps(out.get("all_three_wrong"), ensure_ascii=False))


if __name__ == "__main__":
    main()
