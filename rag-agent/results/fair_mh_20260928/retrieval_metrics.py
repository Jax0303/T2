"""필터 전 검색 지표(A): Recall/Precision/F1/Hit/All-Evidence@k, k=1,5,10,20, 질문별 계산 뒤 macro 평균.

실행:  .venv/bin/python results/fair_mh_20260928/retrieval_metrics.py   (rag-agent/ 에서)
입력:  results/fair_mh_20260928/retrieval/mh_train_<arm>_records.jsonl (doc.ranked_units, gold_ids)
출력:  retrieval_metrics.json, retrieval_metrics.md

정의 (PREREG-2026-09-28-fair-mh.md §3):
  G = 정답 근거 셀 좌표 집합. R_k = 상위 k개 셀 좌표(좌표 기준 중복 제거).
  Recall@k=|G∩R_k|/|G|, Precision@k=|G∩R_k|/k (k개 미만 반환은 빈 자리를 비관련으로), F1=질문별 조화평균,
  Hit@k=|G∩R_k|>0, All-Evidence@k=G⊆R_k (기존 "correct").
  두 가지 k:
   셀 순위 k  — 단위를 순위대로 펼쳐 첫 k개 서로 다른 셀 (단위 안 순서는 표 순서, 청크 arm 은 이 순서가 의미 없어 참고용).
   단위 순위 k — 상위 k개 단위가 담은 셀 전부. Precision 분모는 실제 전달 셀 수(별도 지표 Precision_delivered).
  셀 문장 arm 은 단위 = 셀이라 두 정의가 같다.
"""
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
KS = (1, 5, 10, 20)
ARMS = ["s3c", "chunk", "trag_hetero", "table", "row"]
POP882 = {json.loads(l)["query_id"] for l in open("results/mh_arms/cap300_20260924/cell_uniq.jsonl")}


def metrics(gold, ranked, k, by_unit):
    if by_unit == "budget":      # 예산 20 전달 집합: 서로 다른 셀 20개에 이를 때까지 단위를 통째로 (기존 correct 의 문맥)
        got = []
        for u in ranked:
            got += [c for c in u if c not in got]
            if len(got) >= k:
                break
        denom = len(got)
    elif by_unit:
        got = []
        for u in ranked[:k]:
            got += [c for c in u if c not in got]
        denom = len(got)
    else:
        got = []
        for u in ranked:
            for c in u:
                if c not in got:
                    got.append(c)
                if len(got) == k:
                    break
            if len(got) == k:
                break
        denom = k
    inter = len(gold & set(got))
    rec, prec = inter / len(gold), (inter / denom if denom else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"recall": rec, "precision": prec, "f1": f1, "hit": int(inter > 0), "all": int(gold <= set(got)),
            "n_delivered": len(got)}


def macro(rows):
    keys = ("recall", "precision", "f1", "hit", "all", "n_delivered")
    return {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in keys} | {"n": len(rows)}


out, md = {}, ["# 필터 전 검색 지표 (MultiHiertt train, 문서 안, 최종 머리글 규칙, macro 평균)\n"]
for arm in ARMS:
    f = HERE / "retrieval" / f"mh_train_{arm}_records.jsonl"
    if not f.exists():
        out[arm] = "미실행"; continue
    recs = [r for r in map(json.loads, open(f, encoding="utf-8")) if "excluded" not in r and "doc" in r]
    excluded = sum("excluded" in json.loads(l) for l in open(f, encoding="utf-8"))
    empty_gold = sum(1 for r in recs if not r["gold_ids"])
    recs = [r for r in recs if r["gold_ids"]]
    res = {"n_scored": len(recs), "n_excluded_records": excluded, "n_empty_gold": empty_gold,
           "check_all_at20_equals_correct": None}
    mism = 0
    for pop_name, pop in (("2885", None), ("882", POP882)):
        rs = [r for r in recs if pop is None or r["query_id"] in pop]
        for by_unit in (False, True, "budget"):
            tag = f"{pop_name} {'예산20전달집합' if by_unit == 'budget' else '단위순위' if by_unit else '셀순위'}"
            per_k = {}
            for k in KS:
                rows = [metrics(set(r["gold_ids"]), r["doc"]["ranked_units"], k, by_unit) for r in rs]
                per_k[str(k)] = macro(rows)
                per_k[f"{k}_by_layer"] = {L: macro([x for x, r in zip(rows, rs) if r["layer"] == L])
                                          for L in sorted({r["layer"] for r in rs})}
                if k == 20 and by_unit is False and pop is None:
                    # 셀 arm 은 셀순위 20 = 예산 20 문맥이라 기존 correct 와 같아야 한다 (청크 arm 은 다름이 정상)
                    mism = sum(x["all"] != r["doc"]["correct"] for x, r in zip(rows, rs))
            res[tag] = per_k
    res["check_all_at20_equals_correct"] = {"n_mismatch_cell_rank20_vs_correct": mism,
                                            "note": "셀 arm 은 0 이어야 함; 청크·표·행 arm 은 단위 통째 예산이라 다를 수 있음"}
    out[arm] = res
    for pop_name in ("2885", "882"):
        md.append(f"\n## {arm} — query count={res['n_scored'] if pop_name == '2885' else len(POP882)} ({pop_name})\n")
        for by in ("셀순위", "단위순위", "예산20전달집합"):
            md.append(f"\n{by} k | Recall | Precision | F1 | Hit | All-Evidence | 전달 셀 수\n---|---|---|---|---|---|---")
            for k in (KS if by != "예산20전달집합" else (20,)):
                m = res[f"{pop_name} {by}"][str(k)]
                md.append(f"{k} | {m['recall']:.4f} | {m['precision']:.4f} | {m['f1']:.4f} | {m['hit']:.4f} | {m['all']:.4f} | {m['n_delivered']:.1f}")
(HERE / "retrieval_metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
(HERE / "retrieval_metrics.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
