"""표준 Recall@k · Precision@k · MRR (k = 검색 단위 개수, 사용자 지시 2026-09-29). 문항 평균.

정의 (한 문항, G = 정답 셀 좌표 집합, 단위 u_1..u_n = 방법의 순위 목록, 단위가 배달하는 셀 집합 c(u)):
  hit(g, k) = ∃ i ≤ k, g ∈ c(u_i)
  Recall@k    = |{g ∈ G : hit(g, k)}| / |G|
  Precision@k = |{i ≤ k : c(u_i) ∩ G ≠ ∅}| / k          (단위가 셀 1개면 정답 셀 수/k)
  MRR         = 1 / (정답 셀을 담은 첫 단위의 순위), 저장된 순위 안에 없으면 0
입력: records jsonl. 단위 목록은 r["doc"]["ranked_units"](mh_arms: 셀 id 리스트의 리스트) 또는
      r["ranked_units"](비교군: {"cells": [...]} 의 리스트). gold 는 r["gold_ids"].
실행: .venv/bin/python results/mh_only_20260928/final/rpm.py <records.jsonl> [--ids <query_id 목록 jsonl>] [--name 이름]
출력: JSON 한 줄(stdout) — n, k별 Recall/Precision, MRR, 순위 목록 길이 최소/평균.
"""
import argparse
import json
import statistics

KS = (1, 5, 10, 20, 30)


def units_of(r):
    u = r.get("ranked_units") or (r.get("doc") or {}).get("ranked_units")
    if u is None:
        return None
    return [set(x["cells"]) if isinstance(x, dict) else set(x) for x in u]


def per_query(gold, units):
    out = {}
    for k in KS:
        top = units[:k]
        got = set().union(*top) if top else set()
        out[f"R@{k}"] = len(gold & got) / len(gold)
        out[f"P@{k}"] = sum(1 for u in top if u & gold) / k
    out["MRR"] = next((1 / (i + 1) for i, u in enumerate(units) if u & gold), 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--ids")
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    keep = {json.loads(l)["query_id"] for l in open(a.ids, encoding="utf-8") if l.strip()} if a.ids else None
    rows, lens, skipped = [], [], 0
    for l in open(a.records, encoding="utf-8"):
        r = json.loads(l)
        if keep is not None and r["query_id"] not in keep:
            continue
        u = units_of(r)
        if u is None:
            skipped += 1
            continue
        gold = set(r["gold_ids"])
        if not gold:
            skipped += 1
            continue
        lens.append(len(u))
        rows.append(per_query(gold, u))
    agg = {key: round(statistics.mean(x[key] for x in rows), 4) for key in rows[0]}
    print(json.dumps({"name": a.name, "records": a.records, "n": len(rows), "skipped_no_units_or_gold": skipped,
                      "rank_list_len_min": min(lens), "rank_list_len_mean": round(statistics.mean(lens), 1), **agg},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
