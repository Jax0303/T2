"""GATE 2: T1~T8. 정의는 rag-agent/PREREG-2026-09-28-lookup-arith-posthoc.md (83408b3) 그대로. 새 검색·임베딩 없음.
출력: posthoc.json, posthoc_rows_v3.3u.jsonl, posthoc_rows_v1.jsonl, t6_manual_judgement.csv, REPORT.md
실행: HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=. .venv/bin/python results/lookup_vs_arith_20260928/posthoc/posthoc.py
"""
import csv
import json
import math
import statistics

import numpy as np
from scipy.stats import fisher_exact

from common import HERE, NONE, rows

Z = 1.959964
B, SEED = 10_000, 42
F = "results/lookup_vs_arith_20260928/posthoc/posthoc.json"
A = "results/lookup_vs_arith_20260928/analyze.json"
cmp_ = lambda x: x["comparison"] != NONE
hit = lambda xs: sum(x["rank1"] for x in xs)


def wilson(k, n):
    p, d = k / n, 1 + Z * Z / n
    c, h = (p + Z * Z / (2 * n)) / d, Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return [c - h, c + h]


def rate(xs):
    return {"n": len(xs), "hit": hit(xs), "rate": hit(xs) / len(xs) if xs else None}


def compare(a, b):
    """a − b: 비율, Fisher 양측 p, Newcombe hybrid score 95% CI."""
    (k1, n1), (k2, n2) = (hit(a), len(a)), (hit(b), len(b))
    p1, p2 = k1 / n1, k2 / n2
    (l1, u1), (l2, u2) = wilson(k1, n1), wilson(k2, n2)
    d = p1 - p2
    return {"a": rate(a), "b": rate(b), "diff": d,
            "newcombe95": [d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)],
            "fisher_p": float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])}


def poisson_binomial_p(ps, k):
    dist = np.zeros(len(ps) + 1)
    dist[0] = 1.0
    for p in ps:
        new = dist * (1 - p)
        new[1:] += dist[:-1] * p
        dist = new
    return float(dist[dist <= dist[k] * (1 + 1e-7)].sum())


def t1_to_t4(rs):
    L, Ar = [x for x in rs if x["type"] == "조회"], [x for x in rs if x["type"] == "산술"]
    yn = lambda b: "예" if b else "아니오"
    T1 = {f"머리글_{yn(h)}": {f"비교_{'있음' if c else '없음'}": rate([x for x in L if x["header_answer"] == h and cmp_(x) == c])
                            for c in (True, False)} for h in (True, False)}
    Lc, Ln = [x for x in L if cmp_(x)], [x for x in L if not cmp_(x)]
    Ac, An = [x for x in Ar if cmp_(x)], [x for x in Ar if not cmp_(x)]
    w_c, w_n = len(Ac) / len(Ar), len(An) / len(Ar)
    rng = np.random.default_rng(SEED)
    draw = [rng.binomial(len(s), hit(s) / len(s), size=B) for s in (Lc, Ln, Ac, An)]
    std_b = w_c * draw[0] / len(Lc) + w_n * draw[1] / len(Ln)
    res_b = (draw[2] + draw[3]) / len(Ar) - std_b
    std = w_c * hit(Lc) / len(Lc) + w_n * hit(Ln) / len(Ln)
    T2 = {"i_산술_대_조회": compare(Ar, L), "ii_조회_비교있음_대_비교없음": compare(Lc, Ln),
          "iii_비교없음_산술_대_조회": compare(An, Ln),
          "iv_표준화": {"w_비교": w_c, "w_없음": w_n, "n_산술_비교": len(Ac), "n_산술_없음": len(An),
                     "조회_비교_rate": hit(Lc) / len(Lc), "조회_없음_rate": hit(Ln) / len(Ln),
                     "조회_표준화": std, "조회_표준화_boot95": np.percentile(std_b, [2.5, 97.5]).tolist(),
                     "산술_전체": hit(Ar) / len(Ar), "잔차": hit(Ar) / len(Ar) - std,
                     "잔차_boot95": np.percentile(res_b, [2.5, 97.5]).tolist(),
                     "잔차_boot_p": min(1.0, 2 * min(float((res_b <= 0).mean()), float((res_b >= 0).mean()))),
                     "B": B, "seed": SEED}}
    T3 = {"m분포": {}, "m층_1위": {}}
    for tp, xs in (("조회", L), ("산술", Ar)):
        T3["m분포"][tp] = {}
        for c in (True, False):
            ys = [x for x in xs if cmp_(x) == c]
            m1 = sum(x["m"] == 1 for x in ys)
            T3["m분포"][tp][f"비교_{'있음' if c else '없음'}"] = {"n": len(ys), "m1": m1, "m1_share": m1 / len(ys),
                                                                "m_median": statistics.median(x["m"] for x in ys)}
        T3["m층_1위"][tp] = {"m1": rate([x for x in xs if x["m"] == 1]), "m_gt1": rate([x for x in xs if x["m"] > 1])}
    T4 = {}
    for key, ys in (("비교_있음_전체", Lc), ("비교_있음_m_gt1", [x for x in Lc if x["m"] > 1])):
        ps = [1 / x["m"] for x in ys]
        T4[key] = {**rate(ys), "wilson95": wilson(hit(ys), len(ys)), "mean_1_over_m": sum(ps) / len(ps),
                   "expected_hits": sum(ps), "poisson_binomial_p": poisson_binomial_p(ps, hit(ys))}
    return {"T1": T1, "T2": T2, "T3": T3, "T4": T4}


def t5(rs):
    out = {}
    for tp in ("조회", "산술"):
        out[tp] = {}
        for c in (True, False):
            miss = [x for x in rs if x["type"] == tp and cmp_(x) == c and not x["rank1"]]
            d = {"실패_n": len(miss)}
            for w in ("가_후보_집합_안", "나_같은_표_집합_밖", "다_다른_표"):
                k = sum(x["top1_where"] == w for x in miss)
                d[w] = {"n": k, "share": k / len(miss) if miss else None}
            out[tp][f"비교_{'있음' if c else '없음'}"] = d
    return out


def t6(rs):
    path = HERE / "t6_manual_judgement.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["qid", "유형", "질문", "정답 칸 행 경로", "정답 칸 열 경로", "비교 단어 분류",
                    "질문만으로 정답 칸이 하나로 정해지는가: 예/아니오/판단불가"])
        for x in sorted(rs, key=lambda x: x["query_id"]):
            w.writerow([x["query_id"], x["type"], x["question"], " > ".join(x["row_path"]), " > ".join(x["col_path"]),
                        x["comparison"], ""])
    return {"file": "results/lookup_vs_arith_20260928/posthoc/t6_manual_judgement.csv", "rows": len(rs)}


def t8(rs):
    an = json.load(open(HERE.parent / "analyze.json"))["multihiertt"]["groups"]
    out = {}
    for g, tp in (("조회1", "조회"), ("산술1", "산술")):
        out[g] = {}
        for name, key, sel in (("전체", "path_overlap", lambda x: True), ("예산20_성공", "path_overlap_success", lambda x: x["b20"]),
                               ("예산20_실패", "path_overlap_fail", lambda x: not x["b20"])):
            ys = [x for x in rs if x["type"] == tp and sel(x)]
            po = an[g][key]
            assert po["n"] == len(ys), (g, name)
            out[g][name] = {"n": len(ys), "path_overlap_mean": po["ratio_mean"], "path_overlap_median": po["ratio_median"],
                            "path_overlap_key": f"multihiertt.groups.{g}.{key}",
                            "m_median": statistics.median(x["m"] for x in ys), "m_mean": statistics.mean(x["m"] for x in ys),
                            "m1_share": sum(x["m"] == 1 for x in ys) / len(ys)}
    return out


def report(res):
    f3 = lambda v: "—" if v is None else f"{v:.4f}"
    k = lambda key: f"`{F}:{key}`"
    ci = lambda v: f"[{v[0]:.4f}, {v[1]:.4f}]"
    L = ["# 조회 대 산술 1위 적중 사후 분석 — 결과 표", "",
         "정의: `rag-agent/PREREG-2026-09-28-lookup-arith-posthoc.md`. 모든 키의 파일은 표마다 적었다.", ""]
    for ver, title in (("final", "최종 버전(v3.3u)"), ("v1", "처음 버전(v1), T7")):
        r = res[ver]
        L += [f"## {title}", "", "### T1 조회: 머리글 답 × 비교 단어 (1위 적중 수 / 문항 수)", "",
              "| 머리글 답 | 비교 있음 | 비교 없음 |", "|---|---|---|"]
        for h in ("머리글_예", "머리글_아니오"):
            cells = [f"{r['T1'][h][c]['hit']}/{r['T1'][h][c]['n']} {k(f'{ver}.T1.{h}.{c}.{{hit,n}}')}" for c in ("비교_있음", "비교_없음")]
            L.append(f"| {h[4:]} | " + " | ".join(cells) + " |")
        L += ["", "### T2 통계", "", "| 항목 | a | b | a − b | Newcombe 95% CI | Fisher p | 키 |", "|---|---|---|---|---|---|---|"]
        for it in ("i_산술_대_조회", "ii_조회_비교있음_대_비교없음", "iii_비교없음_산술_대_조회"):
            c = r["T2"][it]
            L.append(f"| {it} | {c['a']['hit']}/{c['a']['n']} ({f3(c['a']['rate'])}) | {c['b']['hit']}/{c['b']['n']} ({f3(c['b']['rate'])}) | "
                     f"{f3(c['diff'])} | {ci(c['newcombe95'])} | {c['fisher_p']:.3g} | {k(f'{ver}.T2.{it}')} |")
        s = r["T2"]["iv_표준화"]
        L += ["", "| (iv) 표준화 | 값 | 부트스트랩 95% CI | p | 키 |", "|---|---|---|---|---|",
              f"| 가중치 w_비교 / w_없음 | {f3(s['w_비교'])} / {f3(s['w_없음'])} | | | {k(f'{ver}.T2.iv_표준화.{{w_비교,w_없음}}')} |",
              f"| 조회 표준화 1위 | {f3(s['조회_표준화'])} | {ci(s['조회_표준화_boot95'])} | | {k(f'{ver}.T2.iv_표준화.{{조회_표준화,조회_표준화_boot95}}')} |",
              f"| 산술 전체 1위 | {f3(s['산술_전체'])} | | | {k(f'{ver}.T2.iv_표준화.산술_전체')} |",
              f"| 잔차 (산술 − 조회 표준화) | {f3(s['잔차'])} | {ci(s['잔차_boot95'])} | {s['잔차_boot_p']:.3g} | {k(f'{ver}.T2.iv_표준화.{{잔차,잔차_boot95,잔차_boot_p}}')} |",
              "", "### T3 m 분포", "", "| 유형 | 비교 단어 | 문항 수 | m=1 수 | m=1 비율 | m 중앙값 | 키 |", "|---|---|---|---|---|---|---|"]
        for tp in ("조회", "산술"):
            for c in ("비교_있음", "비교_없음"):
                d = r["T3"]["m분포"][tp][c]
                L.append(f"| {tp} | {c[3:]} | {d['n']} | {d['m1']} | {f3(d['m1_share'])} | {d['m_median']:g} | {k(f'{ver}.T3.m분포.{tp}.{c}')} |")
        L += ["", "| 유형 | m=1 층 1위 | m>1 층 1위 | 키 |", "|---|---|---|---|"]
        for tp in ("조회", "산술"):
            d = r["T3"]["m층_1위"][tp]
            L.append(f"| {tp} | {d['m1']['hit']}/{d['m1']['n']} ({f3(d['m1']['rate'])}) | {d['m_gt1']['hit']}/{d['m_gt1']['n']} ({f3(d['m_gt1']['rate'])}) | "
                     f"{k(f'{ver}.T3.m층_1위.{tp}.{{m1,m_gt1}}')} |")
        L += ["", "### T4 조회 비교 있음: 관측 1위 대 1/m 평균", "",
              "| 대상 | 관측 1위 | Wilson 95% CI | 1/m 평균 | 기대 적중 수 | 푸아송 이항 p | 키 |", "|---|---|---|---|---|---|---|"]
        for key in ("비교_있음_전체", "비교_있음_m_gt1"):
            d = r["T4"][key]
            L.append(f"| {key} | {d['hit']}/{d['n']} ({f3(d['rate'])}) | {ci(d['wilson95'])} | {f3(d['mean_1_over_m'])} | {d['expected_hits']:.2f} | "
                     f"{d['poisson_binomial_p']:.3g} | {k(f'{ver}.T4.{key}')} |")
        if ver == "final":
            L += ["", "### T5 1위 실패 문항의 1위 칸 위치 (최종 버전)", "",
                  "| 유형 | 비교 단어 | 실패 수 | (가) 후보 집합 안 | (나) 같은 표 집합 밖 | (다) 다른 표 | 키 |", "|---|---|---|---|---|---|---|"]
            for tp in ("조회", "산술"):
                for c in ("비교_있음", "비교_없음"):
                    d = r["T5"][tp][c]
                    cell = lambda w: f"{d[w]['n']} ({f3(d[w]['share'])})"
                    L.append(f"| {tp} | {c[3:]} | {d['실패_n']} | {cell('가_후보_집합_안')} | {cell('나_같은_표_집합_밖')} | {cell('다_다른_표')} | "
                             f"{k(f'final.T5.{tp}.{c}')} |")
        L.append("")
    L += ["## T6", "", f"{res['T6']['rows']}행 {k('T6.{file,rows}')}", "",
          "## T8 path overlap(기존) 대 m (최종 버전)", "",
          f"path overlap 값의 파일은 `{A}`, m 값의 파일은 `{F}`.", "",
          "| 그룹 | 대상 | 문항 수 | path overlap 평균 | path overlap 중앙값 | path overlap 키 | m 중앙값 | m 평균 | m=1 비율 | m 키 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for g in ("조회1", "산술1"):
        for name in ("전체", "예산20_성공", "예산20_실패"):
            d = res["T8"][g][name]
            L.append(f"| {g} | {name} | {d['n']} | {f3(d['path_overlap_mean'])} | {f3(d['path_overlap_median'])} | "
                     f"`{A}:{d['path_overlap_key']}.{{n,ratio_mean,ratio_median}}` | {d['m_median']:g} | {d['m_mean']:.2f} | {f3(d['m1_share'])} | "
                     f"{k(f'T8.{g}.{name}.{{m_median,m_mean,m1_share}}')} |")
    (HERE / "REPORT.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    res = {}
    for ver, key in (("v3.3u", "final"), ("v1", "v1")):
        rs = rows(ver)
        with open(HERE / f"posthoc_rows_{ver}.jsonl", "w") as f:
            f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in rs)
        n = (sum(x["type"] == "조회" for x in rs), sum(x["type"] == "산술" for x in rs))
        assert n == ((212, 71) if ver == "v3.3u" else (211, 71)), n
        res[key] = t1_to_t4(rs)
        if ver == "v3.3u":
            res[key]["T5"] = t5(rs)
            res["T6"] = t6(rs)
            res["T8"] = t8(rs)
    (HERE / "posthoc.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    report(res)
    print((HERE / "REPORT.md").read_text())
