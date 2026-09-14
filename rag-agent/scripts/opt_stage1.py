#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""STAGE 1 — 셀 문장 템플릿 (results/opt/PREREGISTER_OPT.md §4). dev 질의만 채점한다.

조건: baseline(s3c) · T1_leaf_first · T2_field_split 격자 6점 · T3_no_caption.
지표 §2: top-m 엄격(주) · top-m 안정 정렬 · MRR · single 실패 분류 · 점수 차. 통계·판정 §3.
baseline 순위 상위 20 이 v2 기록(context_units)과 1건이라도 다르면 결과를 쓰지 않고 멈춘다.

  PYTHONPATH=. .venv/bin/python scripts/opt_stage1.py
  PYTHONPATH=. .venv/bin/python scripts/opt_stage1.py --limit 20 --out-dir <scratch>   # 스모크
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import binom

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rag_agent.bench import hitab_grid as hg                                    # noqa: E402
from rag_agent.eval.artifacts import digest, file_digest, provenance, read_records  # noqa: E402
from rag_agent.retrieve.encoders import default_encoder                          # noqa: E402
from rag_agent.retrieve.hybrid_index import _minmax, _tokenize                   # noqa: E402
from rag_agent.retrieve.sparse_bm25 import SparseBM25                            # noqa: E402
from rag_agent.serialization.base import fmt_value, join_path                    # noqa: E402
from rag_agent.serialization.caption import caption_sentence, with_page_title    # noqa: E402
from rag_agent.serialization.templates import STRUCTURAL_COMPACT                 # noqa: E402
from scripts.retrieval_accuracy import (PAGE_TITLES, build_corpus, cell_unit,    # noqa: E402
                                        load_queries, query_type)

SPLIT = ROOT / "results/opt/split.json"
V2 = ROOT / "results/evaluation_v2/s3c_v2_records.jsonl"
MODEL, REVISION, ALPHA = "BAAI/bge-base-en-v1.5", "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a", 0.7
GRID = [(.4, .2, .4), (.4, .3, .3), (.5, .2, .3), (.5, .3, .2), (.6, .2, .2), (.6, .3, .1)]
TYPES = ("single", "multi", "arith", "any")
TYPE_OF = {"single_cell": "single", "multi_cell": "multi", "arithmetic": "arith", "header_answer": "any"}


def t1_parts(title, rp, cp, v):
    value = fmt_value(v)
    head = " ".join(x for x in ((rp[-1] if rp else ""), (cp[-1] if cp else "")) if x)
    ps = " / ".join(x for x in (join_path(rp), join_path(cp)) if x)
    tail = ([ps] if ps else []) + ([f"table '{title}'"] if title else [])
    text = (f"{head} — {value}." if head else f"{value}.") + (f" ({', '.join(tail)})" if tail else "")
    return text, f"{head} — {value}", ps


def evaluate(s, gold, mode, unit_tid, qtid):
    """§2 지표 한 질의분. gold = 색인 단위 인덱스 목록(색인에 없는 gold 셀은 None)."""
    n = len(s)
    order = np.argsort(-s, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(1, n + 1)
    g = [u for u in gold if u is not None]
    missing = len(g) < len(gold)
    mask = np.ones(n, dtype=bool)
    mask[g] = False
    max_other = float(s[mask].max())
    if mode == "all":
        strict = not missing and float(s[g].min()) > max_other
        stable = not missing and set(order[:len(gold)].tolist()) == set(g)
    else:
        strict = bool(g) and float(s[g].max()) > max_other
        stable = bool(g) and int(order[0]) in set(g)
    out = {"strict": int(strict), "stable": int(stable),
           "rr": 1.0 / int(rank[g].min()) if g else 0.0, "gold_not_indexed": missing,
           "top20": [int(u) for u in order[:20]]}
    if mode == "all" and len(gold) == 1 and not strict and g:
        top = int(order[0])
        out["fail"] = "tie" if top == g[0] else ("type1" if unit_tid[top] == qtid else "type5")
        out["gap"] = max_other - float(s[g[0]])
    return out


def paired(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(x), (10000, len(x)))
    d = x[idx].mean(1) - y[idx].mean(1)
    b, c = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    return {"delta": float(x.mean() - y.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "b": b, "c": c, "p_mcnemar": 1.0 if b + c == 0 else float(min(1.0, 2 * binom.cdf(min(b, c), b + c, .5)))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--cache-dir", default=".cache/retrieval_accuracy")
    ap.add_argument("--out-dir", default="results/opt/stage1")
    ap.add_argument("--limit", type=int, default=0, help="스모크 전용: dev 질의 앞 N건")
    a = ap.parse_args()
    out = ROOT / a.out_dir
    if out.exists() and any(out.iterdir()):
        ap.error(f"{out} is not empty — outputs are never overwritten")
    if a.limit and out.resolve() == (ROOT / "results/opt/stage1").resolve():
        ap.error("--limit writes only outside results/opt/stage1")
    t0 = time.time()

    split = json.loads(SPLIT.read_text())
    dev_type = {q["query_id"]: q["type"] for q in split["dev_queries"]}
    page_titles = json.loads(PAGE_TITLES.read_text())
    tabs: dict = {}
    queries = load_queries(a.data_dir, "test", tabs)
    tids = sorted({q["table_id"] for q in queries})
    dev = [q for q in queries if q["query_id"] in dev_type]
    assert len(dev) == len(dev_type) == split["n_queries"]["dev"]
    for q in dev:
        assert (q["excluded"] and dev_type[q["query_id"]] == "excluded") or \
               TYPE_OF.get(query_type(q)) == dev_type[q["query_id"]], q["query_id"]
    scored = [q for q in dev if not q["excluded"]]
    if a.limit:
        scored = scored[:a.limit]

    # --- 코퍼스: baseline 은 v2 실행과 같은 build_corpus, 나머지는 같은 셀 순서로 렌더 ---
    base_texts, covers, _, unit_tids, _ = build_corpus(a.data_dir, tids, "s3c", "cell", page_titles)
    cells, t1, leaf, path, cap, t3, s2 = [], [], [], [], [], [], []
    for tid in tids:
        tab = hg.load_table(tid, a.data_dir)
        t = tab.table
        title = with_page_title(tab.title, page_titles.get(tid))
        for i in range(t.n_rows):
            for j in range(t.n_cols):
                if not str(t.data[i][j]).strip():
                    continue
                rp, cp, v = t.row_path(i), t.col_path(j), t.data[i][j]
                cells.append((tid, i, j))
                text, lf, ps = t1_parts(title, rp, cp, v)
                t1.append(text), leaf.append(lf), path.append(ps), cap.append(title)
                t3.append(caption_sentence("", rp, cp, value=v, template=STRUCTURAL_COMPACT))
                s2.append(cell_unit("", rp, cp, v, "s2"))
    assert [frozenset([c]) for c in cells] == covers, "cell order differs from build_corpus"
    unit_of = {c: k for k, c in enumerate(cells)}
    t3_vs_s2 = {"t3_sha256": digest(t3), "s2_sha256": digest(s2), "identical": t3 == s2,
                "n_texts_differ": sum(x != y for x, y in zip(t3, s2))}
    path_u = sorted({p for p in path if p})
    cap_u = sorted({c for c in cap if c})
    pmap, cmap = {p: k for k, p in enumerate(path_u)}, {c: k for k, c in enumerate(cap_u)}
    path_idx = np.array([pmap.get(p, -1) for p in path])
    cap_idx = np.array([cmap.get(c, -1) for c in cap])
    print(f"[corpus] {len(tids)} tables / {len(cells)} cells / paths {len(path_u)} / captions {len(cap_u)} "
          f"({time.time() - t0:.0f}s)", flush=True)

    enc = default_encoder(model_name=MODEL, revision=REVISION)
    cache = ROOT / a.cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    audits = {}

    def embed(name, texts):
        audits[name] = enc.audit_inputs(texts, overflow="error")
        key = digest({"texts": texts, "encoder": enc.metadata(), "overflow": "error"})[:24]
        f = cache / f"{enc.name.replace('/', '_')}_{len(texts)}_{key}.npy"
        if f.exists():
            e = np.load(f)
            print(f"[dense] {name} cache hit {f.name}", flush=True)
        else:
            e = enc.encode(texts)
            np.save(f, e)
            print(f"[dense] {name} encoded {len(texts)} ({time.time() - t0:.0f}s)", flush=True)
        assert e.ndim == 2 and len(e) == len(texts) and np.isfinite(e).all(), name
        return e

    E = {"baseline": embed("baseline", base_texts), "T1": embed("T1", t1), "T3": embed("T3", t3)}
    E_leaf, E_path, E_cap = embed("T2_leaf", leaf), embed("T2_path", path_u), embed("T2_caption", cap_u)
    audits["queries"] = enc.audit_inputs([q["question"] for q in scored], query=True, overflow="error")
    BM = {"baseline": SparseBM25(_tokenize(x) for x in base_texts),
          "T1": SparseBM25(_tokenize(x) for x in t1), "T3": SparseBM25(_tokenize(x) for x in t3)}
    print(f"[bm25] built ({time.time() - t0:.0f}s)", flush=True)

    names = ["baseline", "T1", *[f"T2_{w[0]}_{w[1]}_{w[2]}" for w in GRID], "T3"]
    v2 = read_records(V2)
    repro_mismatch, recs = [], []
    for k, q in enumerate(scored, 1):
        qv = enc.encode_query([q["question"]])[0].astype(np.float32)
        gold = [unit_of.get(c) for c in sorted(q["gold"])]
        bm = {c: BM[c].get_scores(_tokenize(q["question"])) for c in BM}
        scores = {c: ALPHA * _minmax(E[c] @ qv) + (1 - ALPHA) * _minmax(bm[c]) for c in ("baseline", "T1", "T3")}
        cos_leaf = E_leaf @ qv
        cos_path = np.where(path_idx >= 0, (E_path @ qv)[path_idx], 0.0).astype(np.float32)
        cos_cap = np.where(cap_idx >= 0, (E_cap @ qv)[cap_idx], 0.0).astype(np.float32)
        for w in GRID:
            d = w[0] * cos_leaf + w[1] * cos_path + w[2] * cos_cap
            scores[f"T2_{w[0]}_{w[1]}_{w[2]}"] = ALPHA * _minmax(d) + (1 - ALPHA) * _minmax(bm["baseline"])
        row = {"query_id": q["query_id"], "type": dev_type[q["query_id"]], "mode": q["mode"], "m": len(gold)}
        for c in names:
            row[c] = evaluate(scores[c], gold, q["mode"], unit_tids, q["table_id"])
        want = [tuple(x) for u in v2[q["query_id"]]["context_units"] for x in u["cells"]]
        got = [cells[u] for u in row["baseline"]["top20"][:len(want)]]
        if got != want:
            repro_mismatch.append(q["query_id"])
        recs.append(row)
        if k % 100 == 0:
            print(f"  {k}/{len(scored)} ({time.time() - t0:.0f}s)", flush=True)
    if repro_mismatch:
        print(f"STOP: baseline top-20 differs from v2 records on {len(repro_mismatch)} dev queries: "
              f"{repro_mismatch[:10]}", flush=True)
        return 2

    # --- 집계 ---
    def by_type(t):
        return [r for r in recs if r["type"] == t]

    table = {}
    for c in names:
        table[c] = {}
        for t in TYPES:
            rows = by_type(t)
            if not rows:
                continue
            e = {"n": len(rows), "strict": sum(r[c]["strict"] for r in rows),
                 "stable": sum(r[c]["stable"] for r in rows),
                 "mrr": float(np.mean([r[c]["rr"] for r in rows])),
                 "gold_not_indexed": sum(r[c]["gold_not_indexed"] for r in rows)}
            e["strict_acc"], e["stable_acc"] = e["strict"] / e["n"], e["stable"] / e["n"]
            if t == "single":
                f = [r[c] for r in rows if "fail" in r[c]]
                t1f = [x for x in f if x["fail"] == "type1"]
                e["fail"] = {k2: sum(x["fail"] == k2 for x in f) for k2 in ("tie", "type1", "type5")}
                e["gap_all"] = {"median": float(np.median([x["gap"] for x in f])) if f else None,
                                "lt_.01": sum(x["gap"] < .01 for x in f) / len(f) if f else None}
                e["gap_type1"] = {"median": float(np.median([x["gap"] for x in t1f])) if t1f else None,
                                  "lt_.01": sum(x["gap"] < .01 for x in t1f) / len(t1f) if t1f else None}
            table[c][t] = e

    grid_names = [c for c in names if c.startswith("T2_")]
    single_n = len(by_type("single"))
    t2_star = max(grid_names, key=lambda c: (table[c]["single"]["strict"], table[c]["single"]["mrr"],
                                             -grid_names.index(c))) if single_n else None
    comps = {}
    for c in names[1:]:
        comps[c] = {}
        for t in TYPES:
            rows = by_type(t)
            if not rows:
                continue
            base = {m: [r["baseline"][m] for r in rows] for m in ("strict", "stable", "rr")}
            cond = {m: [r[c][m] for r in rows] for m in ("strict", "stable", "rr")}
            comps[c][t] = {m: paired(cond[m], base[m]) for m in ("strict", "stable")}
            comps[c][t]["mrr"] = {k2: v for k2, v in paired(cond["rr"], base["rr"]).items() if k2 in ("delta", "ci95")}
    family = [c for c in ("T1", t2_star, "T3") if c and "single" in comps.get(c, {})]
    ps = sorted(family, key=lambda c: comps[c]["single"]["strict"]["p_mcnemar"])
    run, judgment = 0.0, {}
    for i, c in enumerate(ps):
        run = max(run, min(1.0, (len(ps) - i) * comps[c]["single"]["strict"]["p_mcnemar"]))
        s = comps[c]["single"]["strict"]
        judgment[c] = {"delta": s["delta"], "ci95": s["ci95"], "b": s["b"], "c": s["c"],
                       "p_mcnemar": s["p_mcnemar"], "p_holm": run,
                       "criterion_met": s["delta"] > 0 and s["ci95"][0] > 0 and run < .05}

    out.mkdir(parents=True, exist_ok=True)
    with (out / "records.jsonl").open("x", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "prereg": "results/opt/PREREGISTER_OPT.md §4", "prereg_sha256": file_digest(ROOT / "results/opt/PREREGISTER_OPT.md"),
        "provenance": provenance(ROOT), "arguments": vars(a), "split_sha256": file_digest(SPLIT),
        "population": {"dev_queries": len(dev), "scored": len(scored), "excluded": len(dev) - len([q for q in dev if not q["excluded"]]),
                       "by_type": {t: len(by_type(t)) for t in TYPES}},
        "encoder": enc.metadata(), "alpha": ALPHA, "embedding_input_audit": audits,
        "corpus": {"n_tables": len(tids), "n_cells": len(cells), "n_unique_paths": len(path_u),
                   "n_unique_captions": len(cap_u), "n_empty_head": sum(x.startswith(" — ") for x in leaf),
                   "text_sha256": {"baseline": digest(base_texts), "T1": digest(t1), "T3": digest(t3),
                                   "T2_leaf": digest(leaf), "T2_path": digest(path_u), "T2_caption": digest(cap_u)}},
        "t3_vs_s2_template": t3_vs_s2,
        "reproduction_check": {"reference": str(V2.relative_to(ROOT)), "reference_sha256": file_digest(V2),
                               "queries_checked": len(recs), "mismatch": 0},
        "t2_star": t2_star, "table": table, "vs_baseline": comps, "judgment_single_strict": judgment,
        "records_sha256": file_digest(out / "records.jsonl"), "seconds": round(time.time() - t0),
    }
    (out / "summary.json").open("x", encoding="utf-8").write(json.dumps(summary, ensure_ascii=False, indent=2))

    pct = lambda e, k: f"{e[k]}/{e['n']} = {e[k] / e['n']:.4f}"
    lines = ["# STAGE 1 결과 — dev (해석 없음)", "",
             f"사전등록 `PREREGISTER_OPT.md` §4. 원자료 `summary.json`, `records.jsonl`. T2* = `{t2_star}`.",
             f"재현 점검: baseline 상위 20 순위 = v2 기록, {len(recs)}건 불일치 0. T3 = s2 텍스트: {t3_vs_s2['identical']} "
             f"(다른 텍스트 {t3_vs_s2['n_texts_differ']}).", "",
             "## top-m 엄격", "", "| 조건 | " + " | ".join(TYPES) + " | single MRR |", "|---|" + "---:|" * 5]
    for c in names:
        lines.append(f"| {c}{' (T2*)' if c == t2_star else ''} | "
                     + " | ".join(pct(table[c][t], "strict") if t in table[c] else "—" for t in TYPES)
                     + f" | {table[c]['single']['mrr']:.4f} |")
    lines += ["", "## top-m 안정 정렬", "", "| 조건 | " + " | ".join(TYPES) + " |", "|---|" + "---:|" * 4]
    for c in names:
        lines.append(f"| {c} | " + " | ".join(pct(table[c][t], "stable") if t in table[c] else "—" for t in TYPES) + " |")
    lines += ["", "## 판정 — single top-m 엄격, 기준 대비 (§3)", "",
              "| 조건 | Δ | 95% CI | b:c | McNemar p | Holm p | 기준 충족 |", "|---|---:|---|---:|---:|---:|---|"]
    for c, j in judgment.items():
        lines.append(f"| {c} | {j['delta']:+.4f} | [{j['ci95'][0]:+.4f}, {j['ci95'][1]:+.4f}] | {j['b']}:{j['c']} | "
                     f"{j['p_mcnemar']:.3g} | {j['p_holm']:.3g} | {'예' if j['criterion_met'] else '아니오'} |")
    lines += ["", "## 기준 대비 전 조건 × 전 유형 (판정 외)", "",
              "| 조건 | 유형 | 엄격 Δ [CI] b:c p | 안정 Δ [CI] b:c p | MRR Δ [CI] |", "|---|---|---|---|---|"]
    for c in names[1:]:
        for t, v in comps[c].items():
            f = lambda s: f"{s['delta']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] {s['b']}:{s['c']} {s['p_mcnemar']:.3g}"
            lines.append(f"| {c} | {t} | {f(v['strict'])} | {f(v['stable'])} | "
                         f"{v['mrr']['delta']:+.4f} [{v['mrr']['ci95'][0]:+.4f}, {v['mrr']['ci95'][1]:+.4f}] |")
    lines += ["", "## single 실패 분류·점수 차", "",
              "| 조건 | 동점 | 유형1 | 유형5 | 점수 차 중앙값(전체) | <.01 (전체) | 중앙값(유형1) | <.01 (유형1) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    fm = lambda x: "—" if x is None else f"{x:.4f}"
    for c in names:
        e = table[c]["single"]
        lines.append(f"| {c} | {e['fail']['tie']} | {e['fail']['type1']} | {e['fail']['type5']} | "
                     f"{fm(e['gap_all']['median'])} | {fm(e['gap_all']['lt_.01'])} | "
                     f"{fm(e['gap_type1']['median'])} | {fm(e['gap_type1']['lt_.01'])} |")
    (out / "STAGE1_RESULTS.md").open("x", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
