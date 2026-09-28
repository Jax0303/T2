"""GATE 1 검색 지표 (PREREG-2026-09-28-mh-only.md §3). 문항 평균(macro). 수치는 retrieval/*_records.jsonl 에서만.

정의
  셀 단위 arm(s3c, tablerag_leaf/path): G = 정답 셀 좌표 집합, R_k = 상위 k 셀(좌표 중복 제거).
    Recall@k = |G∩R_k|/|G|, Precision@k = |G∩R_k|/k, MRR = 1/(첫 정답 셀 순위), 저장된 순위(20) 안에 없으면 0.
  청크·행·표·행열 arm: 단위 u 가 g 를 포함하면 g 회수.
    Recall@k = |{g∈G: ∃u∈R_k, g∈u}|/|G|, Precision@k = |{u∈R_k: u∩G≠∅}|/k, MRR = 1/(정답 셀을 포함한 첫 단위 순위).
    (rowcol 은 순위 목록이 행·열 단위이고 실제 전달은 행∩열 교집합이라, 이 표의 값은 "행 또는 열 단위가 정답 셀을 포함"으로 읽는다.)
  다중 정답 셀은 부분 회수를 그대로 점수로. 기존 "전부 포함" = Recall@20 = 1.0 인 문항 비율(열 하나).
  (b) 토큰 예산 표: 순위 순으로 단위를 더해 누적 토큰(단위 텍스트만, Qwen3-8B 토크나이저)이 1,500 을 넘기 직전까지의 집합.
     Precision 분모 = 넣은 단위 수. 저장 순위 20 단위 안에 1,500 이 안 차면 20 단위 전부(그 문항 수를 적는다).
실행:  .venv/bin/python results/mh_only_20260928/metrics.py
"""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
R = HERE / "retrieval"
BUDGET = 1500
KS = (1, 5, 10, 20)
ARMS = [("s3c", "본 방법(셀 문장)", "cell"), ("chunk", "1,000자 청크", "unit"), ("trag_hetero", "TableRAG(Yu) 청킹 이식", "unit"),
        ("table", "표 단위", "unit"), ("rowcol", "행·열 단위", "unit"),
        ("tablerag_leaf", "TableRAG(Chen) 셀 검색 leaf(공식 숫자열 규칙)", "cell"), ("tablerag_path", "TableRAG(Chen) 셀 검색 path", "cell")]
POP882 = {json.loads(l)["query_id"] for l in open("results/mh_arms/cap300_20260924/cell_uniq.jsonl")}


def per_query(gold, units, kind, k):
    gold = set(gold)
    if kind == "cell":
        cells = []
        for u in units:
            for c in u:
                if c not in cells:
                    cells.append(c)
        top = cells[:k]
        inter = len(gold & set(top))
        rank = next((i for i, c in enumerate(cells, 1) if c in gold), None)
        return inter / len(gold), inter / k, (1 / rank if rank else 0.0)
    top = units[:k]
    recovered = {g for g in gold if any(g in u for u in top)}
    hit_units = sum(1 for u in top if gold & set(u))
    rank = next((i for i, u in enumerate(units, 1) if gold & set(u)), None)
    return len(recovered) / len(gold), hit_units / k, (1 / rank if rank else 0.0)


def budget_set(gold, units, toks, kind):
    gold, n, cum = set(gold), 0, 0
    for t in toks:
        if cum + t > BUDGET:
            break
        cum += t; n += 1
    if n == 0:                       # 첫 단위가 이미 1,500 초과 — 첫 단위 하나는 넣는다(표시)
        n, cum, first_over = 1, toks[0], True
    else:
        first_over = False
    top = units[:n]
    if kind == "cell":
        cells = {c for u in top for c in u}
        rec, prec = len(gold & cells) / len(gold), len(gold & cells) / max(len(cells), 1)
    else:
        rec = len({g for g in gold if any(g in u for u in top)}) / len(gold)
        prec = sum(1 for u in top if gold & set(u)) / n
    return {"recall": rec, "precision": prec, "n_units": n, "n_cells": len({c for u in top for c in u}), "tokens": cum,
            "capped_at_20": int(n == len(units) and len(units) == 20 and cum + (toks[19] if len(toks) > 19 else 0) <= BUDGET and False) or int(n == 20 and cum <= BUDGET),
            "first_unit_over": int(first_over)}


def mean(xs):
    return round(sum(xs) / len(xs), 4) if xs else None


out, csv_rows, md = {}, [], ["# GATE 1 검색 지표 (MultiHiertt train 2,885 / 답변 표본 882, 문서 안, 최종 머리글 규칙, 문항 평균)\n"]
for arm, name, kind in ARMS:
    f = R / f"mh_train_{arm}_records.jsonl"
    if not f.exists():
        out[arm] = "미실행"; continue
    recs = [r for r in map(json.loads, open(f, encoding="utf-8")) if "excluded" not in r and "doc" in r and r.get("gold_ids")]
    res = {"name": name, "kind": kind, "n": len(recs), "source": str(f)}
    for pop, sel in (("2885", recs), ("882", [r for r in recs if r["query_id"] in POP882])):
        block = {}
        pq = {k: [per_query(r["gold_ids"], r["doc"]["ranked_units"], kind, k) for r in sel] for k in KS}
        for k in KS:
            block[f"Recall@{k}"] = mean([x[0] for x in pq[k]])
        block["MRR"] = mean([x[2] for x in pq[20]])
        for k in (5, 10, 20):
            block[f"Precision@{k}"] = mean([x[1] for x in pq[k]])
        block["Recall@20=1.0 비율(기존 전부 포함)"] = mean([int(x[0] == 1.0) for x in pq[20]])
        block["기존 correct(예산20 문맥) 비율"] = mean([r["doc"]["correct"] for r in sel])
        bs = [budget_set(r["gold_ids"], r["doc"]["ranked_units"], r["doc"]["ranked_tokens"], kind) for r in sel]
        block[f"토큰예산{BUDGET}"] = {"Recall": mean([b["recall"] for b in bs]), "Precision(분모=단위 수)": mean([b["precision"] for b in bs]),
                                   "단위 수 평균": mean([b["n_units"] for b in bs]), "셀 수 평균": mean([b["n_cells"] for b in bs]),
                                   "누적 토큰 평균": mean([b["tokens"] for b in bs]),
                                   "20단위에서 잘린 문항 수(예산 미달)": sum(b["capped_at_20"] for b in bs),
                                   "첫 단위가 예산 초과인 문항 수": sum(b["first_unit_over"] for b in bs)}
        block["층별 Recall@20"] = {L: mean([x[0] for x, r in zip(pq[20], sel) if r["layer"] == L]) for L in sorted({r["layer"] for r in sel})}
        block["층별 MRR"] = {L: mean([x[2] for x, r in zip(pq[20], sel) if r["layer"] == L]) for L in sorted({r["layer"] for r in sel})}
        res[pop] = block
        for key, v in block.items():
            if isinstance(v, float):
                csv_rows.append({"arm": name, "population": pop, "metric": key, "value": v, "source_file": str(f)})
        for key, v in block[f"토큰예산{BUDGET}"].items():
            csv_rows.append({"arm": name, "population": pop, "metric": f"토큰예산{BUDGET} {key}", "value": v, "source_file": str(f)})
    out[arm] = res

for pop in ("2885", "882"):
    md.append(f"\n## (a) 단위 k 표 — {pop}\n\n| arm | 단위 | R@1 | R@5 | R@10 | R@20 | MRR | P@5 | P@10 | P@20 | R@20=1.0 비율 | 기존 correct |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arm, name, kind in ARMS:
        if out[arm] == "미실행":
            md.append(f"| {name} | | 미실행 |"); continue
        b = out[arm][pop]
        md.append(f"| {name} | {'셀' if kind == 'cell' else '단위'} | {b['Recall@1']:.3f} | {b['Recall@5']:.3f} | {b['Recall@10']:.3f} | {b['Recall@20']:.3f} | {b['MRR']:.3f} | "
                  f"{b['Precision@5']:.3f} | {b['Precision@10']:.3f} | {b['Precision@20']:.3f} | {b['Recall@20=1.0 비율(기존 전부 포함)']:.3f} | {b['기존 correct(예산20 문맥) 비율']:.3f} |")
    md.append(f"\n## (b) 토큰 예산 ≤{BUDGET} 표 — {pop}\n\n| arm | Recall | Precision(단위) | 단위 수 | 셀 수 | 누적 토큰 | 20단위에서 잘림 | 첫 단위 초과 |\n|---|---|---|---|---|---|---|---|")
    for arm, name, kind in ARMS:
        if out[arm] == "미실행":
            continue
        t = out[arm][pop][f"토큰예산{BUDGET}"]
        md.append(f"| {name} | {t['Recall']:.3f} | {t['Precision(분모=단위 수)']:.3f} | {t['단위 수 평균']:.1f} | {t['셀 수 평균']:.1f} | {t['누적 토큰 평균']:.0f} | {t['20단위에서 잘린 문항 수(예산 미달)']} | {t['첫 단위가 예산 초과인 문항 수']} |")

# §3.5 dev 332 본 방법: |G| 분포, Recall@k·MRR 곡선, 누적 토큰
dev_f = R / "mh_dev_s3c_b50_records.jsonl"
if dev_f.exists():
    pop = {json.loads(l)["query_id"] for l in open("results/reader_format_20260927/mh_dev_pop332.jsonl")}
    dev = [r for r in map(json.loads, open(dev_f, encoding="utf-8")) if "excluded" not in r and r["query_id"] in pop]
    g = [len(r["gold_ids"]) for r in dev]
    curve = {}
    for k in (1, 3, 5, 10, 15, 20, 30, 50):
        pq = [per_query(r["gold_ids"], r["doc"]["ranked_units"], "cell", k) for r in dev]
        toks = [sum(r["doc"]["ranked_tokens"][:k]) for r in dev]
        curve[str(k)] = {"Recall": mean([x[0] for x in pq]), "MRR(순위 k 안)": mean([x[2] if x[2] >= 1 / k else 0.0 for x in pq]),
                         "누적 토큰 평균": mean(toks), "누적 토큰 중앙값": statistics.median(toks)}
    out["dev332_s3c"] = {"n": len(dev), "source": str(dev_f), "|G| 최대": max(g), "|G| 중앙값": statistics.median(g),
                         "|G| 분포": dict(sorted({str(v): g.count(v) for v in set(g)}.items(), key=lambda kv: int(kv[0]))),
                         "curve": curve}
    md.append(f"\n## §3.5 dev 332 본 방법 (query count={len(dev)}) — |G| 최대 {max(g)}, 중앙값 {statistics.median(g)}, 분포 {out['dev332_s3c']['|G| 분포']}\n\n| k | Recall@k | 증가(pp) | MRR(k 안) | 누적 토큰 평균 | 중앙값 |\n|---|---|---|---|---|---|")
    prev = None
    for k, v in curve.items():
        inc = "" if prev is None else f"{(v['Recall'] - prev) * 100:+.2f}"
        md.append(f"| {k} | {v['Recall']:.4f} | {inc} | {v['MRR(순위 k 안)']:.4f} | {v['누적 토큰 평균']:.0f} | {v['누적 토큰 중앙값']:.0f} |")
        prev = v["Recall"]
    for r in dev:
        csv_rows.append({"arm": "dev332 s3c", "population": "dev332", "metric": "|G|", "value": len(r["gold_ids"]), "source_file": str(dev_f)})
    for k, v in curve.items():
        for kk, vv in v.items():
            csv_rows.append({"arm": "dev332 s3c", "population": "dev332", "metric": f"k={k} {kk}", "value": vv, "source_file": str(dev_f)})

(HERE / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
(HERE / "metrics.md").write_text("\n".join(md) + "\n", encoding="utf-8")
with open(HERE / "metrics.csv", "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=["arm", "population", "metric", "value", "source_file"]); w.writeheader(); w.writerows(csv_rows)
print("\n".join(md))
