# SPDX-License-Identifier: MIT
"""MultiHiertt 답변 EM 불일치를 검색 성공 여부로 나눈다 — 본 방법 대 MT2Net (사후 분석).

같은 질의 id 에서 두 arm 의 (검색 성공, EM) 을 짝지어
  둘 다 검색 성공 / 한쪽만 성공 / 둘 다 실패
로 나누고, 각 칸의 EM 승패와 gold 문맥 EM 을 센다. 레코드에서 센다(요약 JSON 을 읽지 않는다).

해석의 한계 (2026-09-13 사용자 정정): **둘 다 검색 성공이어도 두 arm 의 문맥은 같지 않다.**
gold 셀과 함께 들어간 나머지 셀의 구성·순서와 셀 문장의 표기(헤더 경로, 단위)는 색인 표현이
정한다. 그러므로 '둘 다 성공' 칸의 EM 차이는 "검색 성공 여부로 설명되지 않는다"까지만 말할 수
있고, "검색 표현과 무관하다"는 인과 해석은 성립하지 않는다. 이 설계는 표현 효과를 문맥 구성
경로와 리더 민감도로 가르지 못한다.

  PYTHONPATH=. .venv/bin/python analysis/mh_em_split.py \
      --out results/mh_arms/em_split_cell_hv2_vs_mt2net.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rag_agent.eval.artifacts import file_digest, provenance, write_pair  # noqa: E402
from analysis.mh_final_report import holm                                # noqa: E402

D = ROOT / "results/mh_arms"
PRIMARY = ("ALL", "lookup_m2+", "arith_m2+")   # REPORT §3 과 같은 세 칸, Holm 묶음
GROUPS = ("both_retrieved", "ours_only_retrieved", "mt2net_only_retrieved", "neither_retrieved")
INTERPRETATION = (
    "EM 불일치 중 '둘 다 검색 성공' 칸의 몫은 검색 성공 여부로 설명되지 않는 몫이다. "
    "그 칸에서도 두 arm 의 문맥(함께 들어간 셀의 구성·순서, 문장 표기)은 색인 표현이 정하므로 "
    "검색 표현과 무관하다고 해석하지 않는다. 표현 효과가 문맥 구성을 거쳐 나타난 것인지, "
    "같은 정보에 대한 리더의 민감도인지는 이 분석으로 식별되지 않는다.")


def load(path):
    return {r["query_id"]: r for r in map(json.loads, Path(path).open(encoding="utf-8"))}


def p_exact(w, l):
    return binomtest(w, w + l).pvalue if w + l else 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", default=str(D / "mh_cell_hv2_answer_doc.jsonl"))
    ap.add_argument("--mt2net", default=str(D / "mh_mt2net_answer_doc.jsonl"))
    ap.add_argument("--gold", default=str(D / "mh_GOLD_hv2_doc.jsonl"))
    ap.add_argument("--retrieval", nargs="*", default=[
        str(D / "mh_cell_hv2_records.jsonl"), str(D / "mh_mt2net_records.jsonl"),
        str(D / "mh_mt2net_header_s3c_records.jsonl")],
        help="검색 짝지음(사후, 탐색용)에 쓸 레코드. 비우면 생략.")
    ap.add_argument("--old", nargs=3, metavar=("OURS", "MT2NET", "GOLD"),
                    help="같은 레그의 이전 리더 설정 답변 파일 — 새 설정 대 이전 설정 짝지음")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ours, mt2, gold = load(a.ours), load(a.mt2net), load(a.gold)
    ids = sorted(ours)
    if set(ids) != set(mt2) or set(ids) != set(gold):
        raise SystemExit("세 답변 파일의 질의 id 집합이 다르다")

    rows = []
    for i in ids:
        ro, rm = int(ours[i]["retrieval_correct"]), int(mt2[i]["retrieval_correct"])
        group = GROUPS[0] if ro and rm else GROUPS[3] if not (ro or rm) else GROUPS[1] if ro else GROUPS[2]
        rows.append({"query_id": i, "layer": ours[i]["layer"], "group": group,
                     "ours_retrieval": ro, "mt2net_retrieval": rm,
                     "ours_em": int(ours[i]["answer_correct"]), "mt2net_em": int(mt2[i]["answer_correct"]),
                     "gold_em": int(gold[i]["answer_correct"]),
                     "answer": ours[i]["answer"], "ours_pred": ours[i]["pred"],
                     "mt2net_pred": mt2[i]["pred"], "gold_pred": gold[i]["pred"]})

    table = {}
    for layer in sorted({r["layer"] for r in rows}) + ["ALL"]:
        rs = [r for r in rows if layer == "ALL" or r["layer"] == layer]
        table[layer] = {"n": len(rs), **{f"em_{k}": round(sum(r[f"{k}_em"] for r in rs) / len(rs), 4)
                                         for k in ("ours", "mt2net", "gold")}}
        for g in GROUPS:
            gs = [r for r in rs if r["group"] == g]
            w = sum(r["ours_em"] > r["mt2net_em"] for r in gs)
            l = sum(r["ours_em"] < r["mt2net_em"] for r in gs)
            table[layer][g] = {
                "n": len(gs), "em_ours_wins": w, "em_mt2net_wins": l,
                "discordant_gold_em_1": sum(r["gold_em"] for r in gs if r["ours_em"] != r["mt2net_em"])}
        w = sum(table[layer][g]["em_ours_wins"] for g in GROUPS)
        l = sum(table[layer][g]["em_mt2net_wins"] for g in GROUPS)
        table[layer]["all_em_ours_wins"], table[layer]["all_em_mt2net_wins"] = w, l
        table[layer]["em_exact_p_unadjusted"] = p_exact(w, l)
    for c, p in zip(PRIMARY, holm([table[c]["em_exact_p_unadjusted"] for c in PRIMARY])):
        table[c]["em_p_holm_primary3"] = p

    vs_old = {}
    for leg, new, old_path in zip(("ours", "mt2net", "gold"), (ours, mt2, gold), a.old or ()):
        old = load(old_path)
        if set(old) != set(ids):
            raise SystemExit(f"{old_path} 의 질의 id 집합이 다르다")
        groups = {L: [i for i in ids if L == "ALL" or new[i]["layer"] == L] for L in table}
        groups["ARITH"] = [i for i in ids if new[i]["kind"] == "arith"]
        vs_old[leg] = {}
        for L, q in groups.items():
            en = [int(new[i]["answer_correct"]) for i in q]
            eo = [int(old[i]["answer_correct"]) for i in q]
            w, l = sum(x > y for x, y in zip(en, eo)), sum(x < y for x, y in zip(en, eo))
            vs_old[leg][L] = {"n": len(q), "em_old": round(sum(eo) / len(q), 4),
                              "em_new": round(sum(en) / len(q), 4), "new_wins": w, "old_wins": l,
                              "exact_p_unadjusted": p_exact(w, l)}

    pairing = {}
    if a.retrieval:
        recs = {Path(p).name: load(p) for p in a.retrieval}
        rid = sorted(i for i in set.intersection(*(set(v) for v in recs.values()))
                     if all("doc" in v[i] and "corpus" in v[i] for v in recs.values()))
        names = list(recs)
        for scope in ("doc", "corpus"):
            pairing[scope] = {"n": len(rid), "accuracy": {
                k: round(sum(v[i][scope]["correct"] for i in rid) / len(rid), 4) for k, v in recs.items()}}
            for x in range(len(names)):
                for y in range(x + 1, len(names)):
                    A, B = recs[names[x]], recs[names[y]]
                    w = sum(1 for i in rid if A[i][scope]["correct"] and not B[i][scope]["correct"])
                    l = sum(1 for i in rid if B[i][scope]["correct"] and not A[i][scope]["correct"])
                    pairing[scope][f"{names[x]} vs {names[y]}"] = {
                        "wins": w, "losses": l, "exact_p_unadjusted": p_exact(w, l)}

    summary = {
        "analysis": "post hoc, not preregistered; p values unadjusted",
        "inputs_sha256": {Path(p).name: file_digest(p)
                          for p in [a.ours, a.mt2net, a.gold, *(a.old or ()), *a.retrieval]},
        "interpretation": INTERPRETATION,
        "em_split": table, "vs_old_setting": vs_old, "retrieval_pairing_exploratory": pairing,
        "provenance": provenance(ROOT)}
    write_pair(Path(a.out), rows, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "provenance"},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
