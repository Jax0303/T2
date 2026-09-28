"""본 방법 대 비교군(원방법) 집계 — PREREG-2026-09-29-baselines-original.md 의 판정 규칙. 새 실행 없음.

입력(있는 것만 집계):
  본 방법  검색 results/mh_only_20260928/retrieval_r100/mh_train_s3c_records.jsonl, 답변 results/mh_arms/cap300_20260924/cell_uniq.jsonl
  비교군   results/mh_only_20260928/baselines/<name>/<name>_records.jsonl, <name>_answer*.jsonl (raptor 는 _answer_2000·_answer_575)
  참고     표 전체 입력 results/mh_arms/cap300_20260924/fulltable.jsonl (답변만)
출력: final/compare.json, final/compare.md
  검색: Recall@k·Precision@k(k=1,5,10,20,30)·MRR(rpm.py 정의), 2,885 와 882 부분집합. 단위 사실: 단위당 평균 셀 수·토큰 수(순위 목록 전체 평균).
        Recall@20 문항별 차이 부호 검정(본 방법 − 비교군, 정확 이항).
  답변: EM(공식)·F1(공식) 882, 본 방법 대비 McNemar 정확(b = 본 방법만 맞힘, c = 비교군만), Holm(가족 = 비교군 답변 파일 전부, 참고 조건 제외).
실행: .venv/bin/python results/mh_only_20260928/final/compare.py [--root <저장소>]
"""
import argparse
import json
import os
import math
import statistics
import sys
from pathlib import Path

from scipy.stats import binomtest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from rpm import per_query, units_of  # noqa: E402

KS = (1, 5, 10, 20, 30)
BASE = "본 방법 (셀 문장)"
OURS_RET = "results/mh_only_20260928/retrieval_r100/mh_train_s3c_records.jsonl"
OURS_ANS = "results/mh_arms/cap300_20260924/cell_uniq.jsonl"
FULL_ANS = "results/mh_arms/cap300_20260924/fulltable.jsonl"
BASELINES = {"langchain_chunk": "LangChain 튜토리얼 청크", "raptor": "RAPTOR", "tablerag_chen": "TableRAG (Chen 2024)",
             "tablerag_yu": "TableRAG (Yu 2025)"}


def jl(p):
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def mc(ref, x, ids):
    b = sum(1 for q in ids if ref[q] and not x[q])
    c = sum(1 for q in ids if x[q] and not ref[q])
    return {"b": b, "c": c, "p": binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 4), round(c + h, 4)


def holm(rows):
    order = sorted(rows, key=lambda k: rows[k]["p"])
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(rows) - i) * rows[k]["p"]))
        rows[k]["p_holm"] = run


def retrieval_block(path, ids882, ids200=None):
    per, facts = {}, {"cells_per_unit": [], "tok_per_unit": []}
    for r in jl(path):
        u = units_of(r)
        if u is None or not r.get("gold_ids"):
            continue
        per[r["query_id"]] = per_query(set(r["gold_ids"]), u)
        raw = r.get("ranked_units") or r["doc"]["ranked_units"]
        facts["cells_per_unit"] += [len(x["cells"]) if isinstance(x, dict) else len(x) for x in raw]
        facts["tok_per_unit"] += [x["n_tok"] for x in raw if isinstance(x, dict) and "n_tok" in x]
    out = {}
    for tag, keep in (("2885", None), ("882", ids882), ("200", ids200)):
        rows = [v for q, v in per.items() if keep is None or q in keep]
        if rows:
            out[tag] = {"n": len(rows), **{k: round(statistics.mean(x[k] for x in rows), 4) for k in rows[0]}}
    out["unit_facts"] = {k: (round(statistics.mean(v), 1) if v else None) for k, v in facts.items()}
    return out, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("DATA_ROOT", str(HERE.parents[2])))
    a = ap.parse_args()
    os.chdir(a.root)
    sys.path.insert(0, a.root)
    from rag_agent.eval.multihiertt_f1 import mh_f1
    from rag_agent.eval.multihiertt_em import docmath_match
    kind_of = {json.loads(l)["query_id"]: json.loads(l)["kind"] for l in open(OURS_ANS, encoding="utf-8") if l.strip()}
    dm = lambda r: int(bool(docmath_match(r.get("pred"), r["answer"], kind_of[r["query_id"]] == "arith")))  # 보조: DocMath 허용 채점(같은 규칙을 모든 방법에)
    f1of = lambda r: float(r["f1"]) if "f1" in r else mh_f1(r.get("pred"), r["answer"], r.get("kind") == "arith")
    ours_ans = {r["query_id"]: r for r in jl(OURS_ANS)}
    ids882 = set(ours_ans)
    ids200 = {json.loads(l)["query_id"] for l in open("results/mh_only_20260928/baselines/ids200.jsonl", encoding="utf-8") if l.strip()}
    res = {"retrieval": {}, "answer": {}, "answer200": {}, "sources": {}}

    ours_ret, ours_per = retrieval_block(OURS_RET, ids882, ids200)
    m = json.load(open("results/mh_only_20260928/metrics.json"))["s3c"]  # GATE 1: 20셀 누적 토큰 평균 / 20 = 셀 문장 1개 토큰
    ours_ret["unit_facts"]["tok_per_unit"] = round(m["2885"]["토큰예산1500"]["누적 토큰 평균"] / 20, 1)
    res["retrieval"][BASE] = ours_ret
    res["sources"][BASE] = {"retrieval": OURS_RET, "answer": OURS_ANS}
    layer = {q: r["layer"] for q, r in ours_ans.items()}   # 유형은 본 방법 파일에서 (query_id 로 붙임)
    GROUPS = ["lookup", "arith", "lookup_m1", "lookup_m2+", "arith_m1", "arith_m2+"]   # 앞 둘 = 2분할(주), 뒤 넷 = 4분할(탐색적)
    def by_group(em, ids):  # 유형별 맞힘/n·비율·Wilson 95%, 본 방법 대비 McNemar
        out = {}
        for g in GROUPS:
            gi = [q for q in ids if layer[q] == g or layer[q].split("_")[0] == g]
            if gi:
                k = sum(em[q] for q in gi)
                out[g] = {"n": len(gi), "EM": k, "rate": round(k / len(gi), 4), "ci95": wilson(k, len(gi)), **mc(ours_em, em, gi)}
        return out
    res["by_group"] = {}
    ours_em = {q: int(r["answer_correct"]) for q, r in ours_ans.items()}
    ours_f1 = {q: f1of(r) for q, r in ours_ans.items()}
    res["answer200"][BASE] = {"n": len(ids200), "EM": sum(ours_em[q] for q in ids200), "EM_rate": round(sum(ours_em[q] for q in ids200) / len(ids200), 4),
                              "F1": round(statistics.mean(ours_f1[q] for q in ids200), 4)}
    res["by_group"][BASE] = {"882": by_group(ours_em, sorted(ids882)), "200": by_group(ours_em, sorted(ids200))}
    res["answer"][BASE] = {"n": 882, "EM": sum(ours_em.values()), "EM_rate": round(sum(ours_em.values()) / 882, 4), "ci95": wilson(sum(ours_em.values()), 882),
                           "DocMath": sum(dm(r) for r in ours_ans.values()),
                           "F1": round(statistics.mean(ours_f1.values()), 4)}

    for name, label in BASELINES.items():
        d = Path("results/mh_only_20260928/baselines") / name
        rec = d / f"{name}_records.jsonl"
        if rec.exists():
            blk, per = retrieval_block(rec, ids882, ids200)
            common = [q for q in per if q in ours_per]
            diff = [ours_per[q]["R@20"] - per[q]["R@20"] for q in common]
            pos, neg = sum(x > 0 for x in diff), sum(x < 0 for x in diff)
            blk["R@20_sign_test_vs_ours"] = {"n": len(common), "ours_higher": pos, "baseline_higher": neg,
                                             "p": binomtest(min(pos, neg), pos + neg, 0.5).pvalue if pos + neg else 1.0}
            res["retrieval"][label] = blk
            res["sources"].setdefault(label, {})["retrieval"] = str(rec)
        for ans in sorted(d.glob(f"{name}_answer*.jsonl")):
            key = label if ans.stem == f"{name}_answer" else f"{label} [{ans.stem.split('_answer_')[1]}]"
            rows = {r["query_id"]: r for r in jl(ans)}
            ids = sorted(set(rows) & ids882)
            if not ids:
                continue
            em = {q: int(rows[q]["answer_correct"]) for q in ids}
            res["answer"][key] = {"n": len(ids), "EM": sum(em.values()), "EM_rate": round(sum(em.values()) / len(ids), 4) if ids else None,
                                  "F1": round(statistics.mean(f1of(rows[q]) for q in ids), 4) if ids else None, "ci95": wilson(sum(em.values()), len(ids)),
                                  "DocMath": sum(dm(rows[q]) for q in ids),
                                  "n_missing_of_882": 882 - len(ids), **mc(ours_em, em, ids)}
            res["sources"].setdefault(key, {})["answer"] = str(ans)
            res["by_group"][key] = {"882": by_group(em, ids), "200": by_group(em, sorted(set(rows) & ids200))}
            i2 = sorted(set(rows) & ids200)
            if i2:
                em2 = {q: int(rows[q]["answer_correct"]) for q in i2}
                res["answer200"][key] = {"n": len(i2), "EM": sum(em2.values()), "EM_rate": round(sum(em2.values()) / len(i2), 4),
                                         "F1": round(statistics.mean(f1of(rows[q]) for q in i2), 4), **mc(ours_em, em2, i2)}

    if Path(FULL_ANS).exists():
        rows = {r["query_id"]: int(r["answer_correct"]) for r in jl(FULL_ANS)}
        ids = sorted(set(rows) & ids882)
        i2 = sorted(set(rows) & ids200)
        res["answer200"]["참고: 표 전체 입력"] = {"n": len(i2), "EM": sum(rows[q] for q in i2), "EM_rate": round(sum(rows[q] for q in i2) / len(i2), 4), "F1": None, **mc(ours_em, rows, i2), "reference_only": True}
        res["by_group"]["참고: 표 전체 입력"] = {"882": by_group(rows, ids), "200": by_group(rows, i2)}
        res["answer"]["참고: 표 전체 입력"] = {"n": len(ids), "EM": sum(rows[q] for q in ids), "EM_rate": round(sum(rows[q] for q in ids) / len(ids), 4),
                                         "F1": None, **mc(ours_em, rows, ids), "reference_only": True}
    for blk in ("answer", "answer200"):
        fam = {k: v for k, v in res[blk].items() if k != BASE and not v.get("reference_only") and "p" in v}
        if fam:
            holm(fam)

    (HERE / "compare.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    L = ["# 본 방법 대 비교군(원방법) — compare.py", ""]
    for tag in ("2885", "882", "200"):
        L += [f"## 검색 (query count {tag}) — Recall@k / Precision@k, k = 검색 단위 개수", "",
              "| 방법 | " + " | ".join(f"R@{k}" for k in KS) + " | MRR | " + " | ".join(f"P@{k}" for k in KS) + " | 단위당 셀 | 단위당 토큰 | R@20 부호검정(본>비:비>본, p) |",
              "|---" * (4 + 2 * len(KS)) + "|"]
        for m, v in res["retrieval"].items():
            if tag not in v:
                continue
            t = v[tag]; f = v["unit_facts"]; s = v.get("R@20_sign_test_vs_ours")
            L.append(f"| {m} | " + " | ".join(f"{t[f'R@{k}']:.3f}" for k in KS) + f" | {t['MRR']:.3f} | " + " | ".join(f"{t[f'P@{k}']:.3f}" for k in KS)
                     + f" | {f['cells_per_unit']} | {f['tok_per_unit']} | " + (f"{s['ours_higher']}:{s['baseline_higher']}, {s['p']:.2g}" if s and tag == "2885" else "—") + " |")
        L.append("")
    for blk, title in (("answer", "답변 (882)"), ("answer200", "답변 (앞 200 문항, 모든 방법 같은 문항)")):
        L += [f"## {title} — 공식 EM·F1(주), DocMath 허용채점(보조), 본 방법 대비 McNemar 정확 + Holm", "", "| 방법 | 맞힘/n | EM | F1 | 허용채점 맞힘 | 본 방법만:상대만 | p | Holm p |", "|---|---|---|---|---|---|---|---|"]
        for m, v in res[blk].items():
            L.append(f"| {m} | {v['EM']}/{v['n']} | {v['EM_rate']:.4f} | " + (f"{v['F1']:.4f}" if v.get("F1") is not None else "—") + f" | {v.get('DocMath', '—')}"
                     + (" | — | — | — |" if m == BASE else f" | {v['b']}:{v['c']} | {v['p']:.2g} | " + (f"{v['p_holm']:.2g}" if "p_holm" in v else "—") + " |"))
        L.append("")
    for tag in ("882", "200"):
        L += [f"## 유형별 답변 EM ({tag}) — [Wilson 95%], 괄호: 본 방법만:상대만, p — 2분할(조회/산술)이 주, 4분할은 탐색적", "", "| 방법 | " + " | ".join(GROUPS) + " |", "|---" * (1 + len(GROUPS)) + "|"]
        for m, v in res["by_group"].items():
            g = v.get(tag, {})
            L.append(f"| {m} | " + " | ".join((f"{g[k]['EM']}/{g[k]['n']} {g[k]['rate']:.3f} [{g[k]['ci95'][0]:.3f},{g[k]['ci95'][1]:.3f}]" + ("" if m == BASE else f" ({g[k]['b']}:{g[k]['c']}, {g[k]['p']:.2g})")) if k in g else "—" for k in GROUPS) + " |")
        L.append("")
    (HERE / "compare.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
