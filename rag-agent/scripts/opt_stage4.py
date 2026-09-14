#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""STAGE 4 — 크로스인코더 재정렬 (results/opt/PREREGISTER_OPT.md §7). dev 질의만 채점한다.

기준 순위 = STAGE 1 baseline 기록(STAGE 1·2 채택 없음)의 상위 20 단위. 그 20개만 (질문, baseline 셀 문장)
CE 점수로 내림차순 재정렬(동점은 원 순위), 21위 이하 불변. CE 점수는 로짓(activation 없음) — 시그모이드 포화가
만드는 가짜 동점을 top-m 엄격 판정에 넣지 않기 위해서다.

  PYTHONPATH=. .venv/bin/python scripts/opt_stage4.py
  PYTHONPATH=. .venv/bin/python scripts/opt_stage4.py --limit 20 --out-dir <scratch>   # 스모크
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rag_agent.eval.artifacts import digest, file_digest, provenance, read_records  # noqa: E402
from scripts.opt_stage1 import SPLIT, TYPES, paired                              # noqa: E402
from scripts.retrieval_accuracy import PAGE_TITLES, build_corpus, load_queries  # noqa: E402

CE_MODEL, CE_REV = "cross-encoder/ms-marco-MiniLM-L-6-v2", "c5ee24cb16019beea0893ab7796b1df96625c6b8"
WIDTH, MAX_LEN, BATCH, WARMUP = 20, 512, 20, 5
STAGE1 = ROOT / "results/opt/stage1"
NEG = re.compile(r"\b(not|no|never|none|neither|nor|without|except|excluding|other than|non)\b|n't\b")
TIME = re.compile(r"\b(1[89]\d{2}|20\d{2})\b|\b(january|february|march|april|june|july|august|september|october|"
                  r"november|december|years?|months?|quarters?|decades?|century|before|after|since|until|during|"
                  r"earlier|later|previous|recent|recently|season)\b")


def judge(top20, new, score, gold, mode, base_rr, unit_tid, qtid):
    """재정렬 순위 한 질의분. gold 는 gold 단위 집합(색인 안 된 gold 셀은 None)."""
    in20 = [u for u in gold if u in score]
    mo = max((score[u] for u in top20 if u not in gold), default=float("-inf"))
    if mode == "all":
        strict = len(in20) == len(gold) and min(score[u] for u in in20) > mo
        stable = set(new[:len(gold)]) == gold
    else:
        strict = bool(in20) and max(score[u] for u in in20) > mo
        stable = new[0] in gold
    out = {"strict": int(strict), "stable": int(stable),
           "rr": 1.0 / (1 + min(new.index(u) for u in in20)) if in20 else base_rr, "top20": new}
    if mode == "all" and len(gold) == 1:
        g = next(iter(gold))
        before = round(1 / base_rr)
        after = new.index(g) + 1 if g in score else before
        out["rank_before"], out["rank_after"] = before, after
        out["rank_change"] = ("outside_20" if before > WIDTH else "1->1" if before == after == 1 else
                              "1->demoted" if before == 1 else "promoted->1" if after == 1 else
                              "improved_not_1" if after < before else "unchanged" if after == before else "worse")
        if not strict:
            top = new[0]
            out["fail"] = "tie" if top == g else ("type1" if unit_tid[top] == qtid else "type5")
            out["gap"] = mo - score[g] if g in score else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--out-dir", default="results/opt/stage4")
    ap.add_argument("--limit", type=int, default=0, help="스모크 전용: dev 질의 앞 N건")
    a = ap.parse_args()
    out = ROOT / a.out_dir
    if out.exists() and any(out.iterdir()):
        ap.error(f"{out} is not empty — outputs are never overwritten")
    if a.limit and out.resolve() == (ROOT / "results/opt/stage4").resolve():
        ap.error("--limit writes only outside results/opt/stage4")
    t0 = time.time()

    s1_sum = json.loads((STAGE1 / "summary.json").read_text())
    assert s1_sum["records_sha256"] == file_digest(STAGE1 / "records.jsonl")
    stage1 = read_records(STAGE1 / "records.jsonl")
    split = json.loads(SPLIT.read_text())
    dev_type = {q["query_id"]: q["type"] for q in split["dev_queries"]}
    queries = load_queries(a.data_dir, "test", {})
    tids = sorted({q["table_id"] for q in queries})
    scored = [q for q in queries if q["query_id"] in dev_type and not q["excluded"]]
    assert {q["query_id"] for q in scored} == set(stage1)
    if a.limit:
        scored = scored[:a.limit]
    texts, covers, _, unit_tids, _ = build_corpus(a.data_dir, tids, "s3c", "cell",
                                                   json.loads(PAGE_TITLES.read_text()))
    assert digest(texts) == s1_sum["corpus"]["text_sha256"]["baseline"], "baseline sentences differ from STAGE 1"
    unit_of = {next(iter(c)): k for k, c in enumerate(covers)}

    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(CE_MODEL, revision=CE_REV, device="cuda", max_length=MAX_LEN, activation_fn=torch.nn.Identity())
    tok = ce.tokenizer
    print(f"[ce] loaded ({time.time() - t0:.0f}s)", flush=True)

    recs, lat, overflow = [], [], 0
    for k, q in enumerate(scored, 1):
        s1 = stage1[q["query_id"]]
        top20 = s1["baseline"]["top20"][:WIDTH]
        pairs = [(q["question"], texts[u]) for u in top20]
        overflow += sum(len(tok(x, y)["input_ids"]) > MAX_LEN for x, y in pairs)
        torch.cuda.synchronize()
        t = time.perf_counter()
        ce_scores = ce.predict(pairs, batch_size=BATCH, convert_to_numpy=True, show_progress_bar=False)
        new = [top20[i] for i in sorted(range(len(top20)), key=lambda i: (-float(ce_scores[i]), i))]
        torch.cuda.synchronize()
        if k > WARMUP:
            lat.append(time.perf_counter() - t)
        score = {u: float(v) for u, v in zip(top20, ce_scores)}
        gold = {unit_of.get(c) for c in q["gold"]}
        neg, tm = bool(NEG.search(q["question"].lower())), bool(TIME.search(q["question"].lower()))
        recs.append({"query_id": q["query_id"], "type": dev_type[q["query_id"]], "mode": q["mode"], "m": len(gold),
                     "neg": neg, "time": tm, "ce_scores": [float(v) for v in ce_scores],
                     "baseline": {x: s1["baseline"][x] for x in ("strict", "stable", "rr")},
                     "rerank": judge(top20, new, score, gold, q["mode"], s1["baseline"]["rr"], unit_tids, q["table_id"])})
        if k % 100 == 0:
            print(f"  {k}/{len(scored)} ({time.time() - t0:.0f}s)", flush=True)

    def block(rows):
        e = {"n": len(rows)}
        for c in ("baseline", "rerank"):
            e[c] = {"strict": sum(r[c]["strict"] for r in rows), "stable": sum(r[c]["stable"] for r in rows),
                    "mrr": float(np.mean([r[c]["rr"] for r in rows]))}
        e["vs_baseline"] = {m: paired([r["rerank"][m] for r in rows], [r["baseline"][m] for r in rows])
                            for m in ("strict", "stable")}
        mrr = paired([r["rerank"]["rr"] for r in rows], [r["baseline"]["rr"] for r in rows])
        e["vs_baseline"]["mrr"] = {"delta": mrr["delta"], "ci95": mrr["ci95"]}
        return e

    groups = {t: [r for r in recs if r["type"] == t] for t in TYPES}
    groups = {g: v for g, v in groups.items() if v}
    table = {g: block(v) for g, v in groups.items()}
    subgroups = {}
    for g, v in groups.items():
        parts = {"NEG": [r for r in v if r["neg"]], "TIME": [r for r in v if r["time"]],
                 "type3": [r for r in v if r["neg"] or r["time"]], "rest": [r for r in v if not (r["neg"] or r["time"])]}
        subgroups[g] = {p: block(x) for p, x in parts.items() if x}
    single = groups.get("single", [])
    changes = Counter(r["rerank"]["rank_change"] for r in single)
    b, c = changes["promoted->1"], changes["1->demoted"]
    from scipy.stats import binom
    promo = {"promoted": b, "demoted": c, "p_mcnemar": 1.0 if b + c == 0 else float(min(1.0, 2 * binom.cdf(min(b, c), b + c, .5)))}
    j = table["single"]["vs_baseline"]["strict"] if single else None
    judgment = None if j is None else {**j, "criterion_met": j["delta"] > 0 and j["ci95"][0] > 0}
    fails = Counter(r["rerank"]["fail"] for r in single if "fail" in r["rerank"])
    gaps = [r["rerank"]["gap"] for r in single if r["rerank"].get("gap") is not None]
    lat_s = {"n": len(lat), "mean": float(np.mean(lat)), "median": float(np.median(lat)),
             "p95": float(np.percentile(lat, 95))} if lat else None

    out.mkdir(parents=True, exist_ok=True)
    with (out / "records.jsonl").open("x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
    ce_first = next(ce.parameters())
    summary = {
        "prereg": "results/opt/PREREGISTER_OPT.md §7", "prereg_sha256": file_digest(ROOT / "results/opt/PREREGISTER_OPT.md"),
        "provenance": provenance(ROOT), "arguments": vars(a), "split_sha256": file_digest(SPLIT),
        "baseline_source": {"records": "results/opt/stage1/records.jsonl", "sha256": s1_sum["records_sha256"]},
        "cross_encoder": {"model": CE_MODEL, "revision_requested": CE_REV, "max_length": MAX_LEN, "batch_size": BATCH,
                          "activation": "identity (logits)", "device": str(ce_first.device), "dtype": str(ce_first.dtype)},
        "rerank_width": WIDTH, "pairs_over_max_length": overflow,
        "latency_seconds_excl_warmup": lat_s, "warmup_queries": WARMUP,
        "population": {g: len(v) for g, v in groups.items()},
        "table": table, "subgroups_type3": subgroups, "single_rank_change": dict(changes),
        "single_top1_promotion_vs_demotion": promo, "judgment_single_strict": judgment,
        "single_fail_classes_rerank": dict(fails),
        "single_gap_rerank": {"n": len(gaps), "median": float(np.median(gaps)) if gaps else None,
                              "lt_.01": sum(x < .01 for x in gaps) / len(gaps) if gaps else None},
        "records_sha256": file_digest(out / "records.jsonl"), "seconds": round(time.time() - t0),
    }
    with (out / "summary.json").open("x", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    pct = lambda k2, n: f"{k2}/{n} = {k2 / n:.4f}"
    f3 = lambda s: f"{s['delta']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]"
    lines = ["# STAGE 4 결과 — dev (해석 없음)", "",
             f"사전등록 `PREREGISTER_OPT.md` §7. CE 입력 512 초과 쌍 {overflow}. 지연(워밍업 {WARMUP} 제외) {lat_s}.", "",
             "## 판정 — single top-m 엄격", ""]
    if judgment:
        lines += ["| Δ | 95% CI | b:c | McNemar p | 기준 충족 |", "|---:|---|---:|---:|---|",
                  f"| {judgment['delta']:+.4f} | [{judgment['ci95'][0]:+.4f}, {judgment['ci95'][1]:+.4f}] | "
                  f"{judgment['b']}:{judgment['c']} | {judgment['p_mcnemar']:.3g} | {'예' if judgment['criterion_met'] else '아니오'} |"]
    lines += ["", "## 전 유형", "",
              "| 유형 | base 엄격 | 재정렬 엄격 | 엄격 Δ [CI] b:c p | base 안정 | 재정렬 안정 | 안정 Δ [CI] b:c p | MRR base→재정렬, Δ [CI] |",
              "|---|---:|---:|---|---:|---:|---|---|"]
    for g, e in table.items():
        v = e["vs_baseline"]
        lines.append(f"| {g} | {pct(e['baseline']['strict'], e['n'])} | {pct(e['rerank']['strict'], e['n'])} | "
                     f"{f3(v['strict'])} {v['strict']['b']}:{v['strict']['c']} {v['strict']['p_mcnemar']:.3g} | "
                     f"{pct(e['baseline']['stable'], e['n'])} | {pct(e['rerank']['stable'], e['n'])} | "
                     f"{f3(v['stable'])} {v['stable']['b']}:{v['stable']['c']} {v['stable']['p_mcnemar']:.3g} | "
                     f"{e['baseline']['mrr']:.4f}→{e['rerank']['mrr']:.4f}, {f3(v['mrr'])} |")
    lines += ["", "## single gold 순위 변화", "", f"{dict(changes)}", "",
              f"1위 승격:강등 {b}:{c}, McNemar p {promo['p_mcnemar']:.3g}", "",
              f"재정렬 후 실패 분류 {dict(fails)} · 점수 차(로짓, gold 가 20 안) {summary['single_gap_rerank']}", "",
              "## 유형3 하위집단 (single)", "", "| 하위집단 | n | base 엄격 | 재정렬 엄격 | 엄격 Δ [CI] b:c p |", "|---|---:|---:|---:|---|"]
    for p, e in subgroups.get("single", {}).items():
        v = e["vs_baseline"]["strict"]
        lines.append(f"| {p} | {e['n']} | {e['baseline']['strict']} | {e['rerank']['strict']} | "
                     f"{f3(v)} {v['b']}:{v['c']} {v['p_mcnemar']:.3g} |")
    (out / "STAGE4_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
