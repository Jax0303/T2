#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""STAGE 2 — 질의 분해 + RRF (results/opt/PREREGISTER_OPT.md §5). dev 질의만 채점한다.

1단계: Qwen2.5-7B 4bit 로 dev 질의마다 하위 질의 생성 → generations.jsonl (질의마다 바로 기록, 다시 돌리면 이어서).
2단계: 모델을 내린 뒤 baseline(s3c) 셀 문장 하이브리드로 하위 질의마다 전 코퍼스를 정렬하고 RRF 로 합친다.
STAGE 1 채택 없음 → 기준은 baseline. baseline 을 다시 계산해 STAGE 1 기록과 다르면 결과를 쓰지 않고 멈춘다.

  PYTHONPATH=. .venv/bin/python scripts/opt_stage2.py
  PYTHONPATH=. .venv/bin/python scripts/opt_stage2.py --limit 10 --out-dir <scratch>   # 스모크
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rag_agent.eval.artifacts import digest, file_digest, provenance, read_records  # noqa: E402
from rag_agent.retrieve.encoders import default_encoder                          # noqa: E402
from rag_agent.retrieve.hybrid_index import _minmax, _tokenize                   # noqa: E402
from rag_agent.retrieve.sparse_bm25 import SparseBM25                            # noqa: E402
from scripts.opt_stage1 import ALPHA, MODEL, REVISION, SPLIT, TYPES, evaluate, paired  # noqa: E402
from scripts.retrieval_accuracy import PAGE_TITLES, build_corpus, load_queries  # noqa: E402

READER, READER_REV, MAX_TOKENS, SEED = "Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28", 256, 42
SYSTEM = "You decompose questions about statistical tables into search queries for retrieving table cells."
USER = ("Question: {question}\n\n"
        "Each value needed to answer the question is one table cell. Write one short search query per cell, "
        "naming what the cell's row and column describe. If one cell is enough, write exactly one query. "
        'Output only a JSON array of strings, e.g. ["query 1", "query 2"].')
RRF_K = 60
STAGE1 = ROOT / "results/opt/stage1/records.jsonl"


def parse(text):
    """첫 '[' 부터 마지막 ']' 까지 json.loads. 문자열만 담은 리스트여야 하고, strip 후 빈 문자열은 버린다.
    남는 것이 없거나 형식이 다르면 (None, 0) — 호출자가 원 질문 하나로 대체한다.
    완전 중복은 순서를 지켜 제거하고 제거 수를 돌려준다."""
    i, j = text.find("["), text.rfind("]")
    if i < 0 or j < i:
        return None, 0
    try:
        v = json.loads(text[i:j + 1])
    except ValueError:
        return None, 0
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        return None, 0
    v = [x.strip() for x in v if x.strip()]
    if not v:
        return None, 0
    kept = list(dict.fromkeys(v))
    return kept, len(v) - len(kept)


def generate(scored, out):
    cfg = {"reader": READER, "revision": READER_REV, "quantization": "4bit", "max_new_tokens": MAX_TOKENS,
           "temperature": 0.0, "seed": SEED, "system_sha256": digest(SYSTEM), "user_template_sha256": digest(USER)}
    cfg_path, gen_path = out / "generation_config.json", out / "generations.jsonl"
    if cfg_path.exists():
        assert json.loads(cfg_path.read_text()) == cfg, "generation config changed — use a new --out-dir"
    else:
        cfg_path.write_text(json.dumps(cfg, indent=2))
    done = read_records(gen_path) if gen_path.exists() else {}
    todo = [q for q in scored if q["query_id"] not in done]
    if todo:
        import torch
        from rag_agent.llm.local_qwen import LocalQwenLLM
        torch.manual_seed(SEED)
        llm = LocalQwenLLM(model_name=READER, quantization="4bit", revision=READER_REV)
        meta_path = out / "reader_metadata.json"
        meta = llm.metadata()
        if meta_path.exists():
            assert json.loads(meta_path.read_text())["revision_resolved"] == meta["revision_resolved"]
        else:
            meta_path.write_text(json.dumps(meta, indent=2))
        t0 = time.time()
        with gen_path.open("a", encoding="utf-8") as fh:
            for k, q in enumerate(todo, 1):
                text = llm.complete(SYSTEM, USER.replace("{question}", q["question"]),
                                    max_tokens=MAX_TOKENS, temperature=0.0)
                fh.write(json.dumps({"query_id": q["query_id"], "output": text}, ensure_ascii=False) + "\n")
                fh.flush()
                if k % 25 == 0:
                    print(f"  gen {k}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    return read_records(gen_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--cache-dir", default=".cache/retrieval_accuracy")
    ap.add_argument("--out-dir", default="results/opt/stage2")
    ap.add_argument("--limit", type=int, default=0, help="스모크 전용: dev 질의 앞 N건")
    a = ap.parse_args()
    out = ROOT / a.out_dir
    if (out / "summary.json").exists():
        ap.error(f"{out}/summary.json exists — outputs are never overwritten")
    if a.limit and out.resolve() == (ROOT / "results/opt/stage2").resolve():
        ap.error("--limit writes only outside results/opt/stage2")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    split = json.loads(SPLIT.read_text())
    dev_type = {q["query_id"]: q["type"] for q in split["dev_queries"]}
    tabs: dict = {}
    queries = load_queries(a.data_dir, "test", tabs)
    tids = sorted({q["table_id"] for q in queries})
    scored = [q for q in queries if q["query_id"] in dev_type and not q["excluded"]]
    if a.limit:
        scored = scored[:a.limit]

    gens = generate(scored, out)
    print(f"[generate] {len(gens)} outputs ({time.time() - t0:.0f}s)", flush=True)

    page_titles = json.loads(PAGE_TITLES.read_text())
    texts, covers, _, unit_tids, _ = build_corpus(a.data_dir, tids, "s3c", "cell", page_titles)
    cells = [next(iter(c)) for c in covers]
    unit_of = {c: k for k, c in enumerate(cells)}
    enc = default_encoder(model_name=MODEL, revision=REVISION)
    key = digest({"texts": texts, "encoder": enc.metadata(), "overflow": "error"})[:24]
    f = ROOT / a.cache_dir / f"{enc.name.replace('/', '_')}_{len(texts)}_{key}.npy"
    assert f.exists(), f"baseline embedding cache missing: {f.name}"
    E = np.load(f)
    bm = SparseBM25(_tokenize(x) for x in texts)
    stage1 = read_records(STAGE1)
    print(f"[retrieval] corpus {len(cells)} cells ready ({time.time() - t0:.0f}s)", flush=True)

    def hybrid(text):
        qv = enc.encode_query([text])[0].astype(np.float32)
        return ALPHA * _minmax(E @ qv) + (1 - ALPHA) * _minmax(bm.get_scores(_tokenize(text)))

    sub_texts = []
    recs, mismatch = [], []
    for k, q in enumerate(scored, 1):
        gold = [unit_of.get(c) for c in sorted(q["gold"])]
        base = evaluate(hybrid(q["question"]), gold, q["mode"], unit_tids, q["table_id"])
        s1 = stage1[q["query_id"]]["baseline"]
        if any(base[x] != s1[x] for x in ("strict", "stable", "rr", "top20")):
            mismatch.append(q["query_id"])
        subs, dupes = parse(gens[q["query_id"]]["output"])
        failed = subs is None
        subs = [q["question"]] if failed else subs
        sub_texts.extend(subs)
        n = len(cells)
        rrf = np.zeros(n, dtype=np.float64)
        for sq in subs:
            order = np.argsort(-hybrid(sq), kind="stable")
            rank = np.empty(n, dtype=np.int64)
            rank[order] = np.arange(1, n + 1)
            rrf += 1.0 / (RRF_K + rank)
        recs.append({"query_id": q["query_id"], "type": dev_type[q["query_id"]], "mode": q["mode"], "m": len(gold),
                     "subqueries": subs, "parse_failed": failed, "duplicates_removed": dupes,
                     "baseline": base, "decomp": evaluate(rrf, gold, q["mode"], unit_tids, q["table_id"])})
        if k % 100 == 0:
            print(f"  {k}/{len(scored)} ({time.time() - t0:.0f}s)", flush=True)
    if mismatch:
        print(f"STOP: baseline differs from STAGE 1 records on {len(mismatch)} queries: {mismatch[:10]}", flush=True)
        return 2
    audit = enc.audit_inputs(sub_texts, query=True, overflow="error")

    def block(rows, c):
        return {"n": len(rows), "strict": sum(r[c]["strict"] for r in rows),
                "stable": sum(r[c]["stable"] for r in rows), "mrr": float(np.mean([r[c]["rr"] for r in rows]))}

    groups = {t: [r for r in recs if r["type"] == t] for t in TYPES}
    groups["multi+arith"] = groups["multi"] + groups["arith"]
    groups = {g: v for g, v in groups.items() if v}
    table = {c: {g: block(v, c) for g, v in groups.items()} for c in ("baseline", "decomp")}
    comps = {}
    for g, v in groups.items():
        comps[g] = {m: paired([r["decomp"][m] for r in v], [r["baseline"][m] for r in v]) for m in ("strict", "stable")}
        mrr = paired([r["decomp"]["rr"] for r in v], [r["baseline"]["rr"] for r in v])
        comps[g]["mrr"] = {"delta": mrr["delta"], "ci95": mrr["ci95"]}
    j = comps.get("multi+arith", {}).get("strict")
    judgment = None if j is None else {**j, "criterion_met": j["delta"] > 0 and j["ci95"][0] > 0}
    fails = {c: Counter(r[c]["fail"] for r in groups.get("single", []) if "fail" in r[c]) for c in ("baseline", "decomp")}
    n_sub = {g: dict(sorted(Counter(len(r["subqueries"]) for r in v).items())) for g, v in groups.items()}

    with (out / "records.jsonl").open("x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
    summary = {
        "prereg": "results/opt/PREREGISTER_OPT.md §5", "prereg_sha256": file_digest(ROOT / "results/opt/PREREGISTER_OPT.md"),
        "provenance": provenance(ROOT), "arguments": vars(a), "split_sha256": file_digest(SPLIT),
        "generation_config": json.loads((out / "generation_config.json").read_text()),
        "generations_sha256": file_digest(out / "generations.jsonl"),
        "encoder": enc.metadata(), "alpha": ALPHA, "rrf_k": RRF_K, "subquery_input_audit": audit,
        "baseline_check": {"reference": str(STAGE1.relative_to(ROOT)), "reference_sha256": file_digest(STAGE1),
                           "queries_checked": len(recs), "mismatch": 0},
        "population": {g: len(v) for g, v in groups.items()},
        "parse_failed": sum(r["parse_failed"] for r in recs),
        "duplicates_removed": sum(r["duplicates_removed"] for r in recs),
        "subquery_count_distribution": n_sub, "table": table, "vs_baseline": comps,
        "judgment_multi_arith_strict": judgment, "single_fail_classes": {c: dict(v) for c, v in fails.items()},
        "records_sha256": file_digest(out / "records.jsonl"), "seconds": round(time.time() - t0),
    }
    with (out / "summary.json").open("x", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    pct = lambda e, k2: f"{e[k2]}/{e['n']} = {e[k2] / e['n']:.4f}"
    f3 = lambda s: f"{s['delta']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]"
    lines = ["# STAGE 2 결과 — dev (해석 없음)", "",
             f"사전등록 `PREREGISTER_OPT.md` §5. baseline 재계산 = STAGE 1 기록 {len(recs)}건 불일치 0. "
             f"파싱 실패 {summary['parse_failed']} · 중복 제거 {summary['duplicates_removed']} · 하위 질의 입력 초과 {audit['n_overflow']}.",
             "", "## 판정 — multi+arith top-m 엄격", ""]
    if judgment:
        lines += ["| Δ | 95% CI | b:c | McNemar p | 기준 충족 |", "|---:|---|---:|---:|---|",
                  f"| {judgment['delta']:+.4f} | [{judgment['ci95'][0]:+.4f}, {judgment['ci95'][1]:+.4f}] | "
                  f"{judgment['b']}:{judgment['c']} | {judgment['p_mcnemar']:.3g} | {'예' if judgment['criterion_met'] else '아니오'} |"]
    lines += ["", "## 전 유형", "",
              "| 유형 | baseline 엄격 | 분해 엄격 | 엄격 Δ [CI] b:c p | baseline 안정 | 분해 안정 | 안정 Δ [CI] b:c p | MRR base→분해, Δ [CI] |",
              "|---|---:|---:|---|---:|---:|---|---|"]
    for g in groups:
        b, d, c = table["baseline"][g], table["decomp"][g], comps[g]
        lines.append(f"| {g} | {pct(b, 'strict')} | {pct(d, 'strict')} | {f3(c['strict'])} {c['strict']['b']}:{c['strict']['c']} "
                     f"{c['strict']['p_mcnemar']:.3g} | {pct(b, 'stable')} | {pct(d, 'stable')} | {f3(c['stable'])} "
                     f"{c['stable']['b']}:{c['stable']['c']} {c['stable']['p_mcnemar']:.3g} | "
                     f"{b['mrr']:.4f}→{d['mrr']:.4f}, {f3(c['mrr'])} |")
    lines += ["", "## 하위 질의 수 분포", "", "| 유형 | 분포 {개수: 질의 수} |", "|---|---|"]
    lines += [f"| {g} | {v} |" for g, v in n_sub.items()]
    lines += ["", "## single 실패 분류", "", f"baseline {dict(fails['baseline'])} · 분해 {dict(fails['decomp'])}"]
    (out / "STAGE2_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
