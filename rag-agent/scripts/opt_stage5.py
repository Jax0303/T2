#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""STAGE 5 — 표 먼저, 그 표의 셀만 (results/opt/PREREGISTER_OPT.md §8). dev 질의만 채점한다.

5-1: Qwen2.5-7B 4bit 로 538표 캡션을 1회 생성 → captions.jsonl (표마다 바로 기록, 다시 돌리면 이어서). 질의·gold 미사용.
5-2: 모델을 내린 뒤 표 텍스트(제목·캡션·열 경로·행 잎)에 하이브리드 → 상위 T 표(T=1,3,5). 그 표의 셀만 남겨
     min-max 를 다시 하고 안정 정렬한다(`retrieval_accuracy.py --corpus gold` 마스킹과 같은 방식).
기준 = baseline(STAGE 1·2·4 채택 없음). baseline 을 다시 계산해 STAGE 1 기록과 다르면 결과를 쓰지 않고 멈춘다.
gold 표가 상위 T 밖이면 top-m 엄격·안정 실패이고, 사전등록에 정의가 없는 MRR 은 0 으로 둔다.

  PYTHONPATH=. .venv/bin/python scripts/opt_stage5.py
  PYTHONPATH=. .venv/bin/python scripts/opt_stage5.py --limit 20 --caption-limit 5 --out-dir <scratch>   # 스모크
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
from rag_agent.bench import hitab_grid as hg                                    # noqa: E402
from rag_agent.eval.artifacts import digest, file_digest, provenance, read_records  # noqa: E402
from rag_agent.retrieve.encoders import default_encoder                          # noqa: E402
from rag_agent.retrieve.hybrid_index import _minmax, _tokenize                   # noqa: E402
from rag_agent.retrieve.sparse_bm25 import SparseBM25                            # noqa: E402
from rag_agent.serialization.base import fmt_value, join_path                    # noqa: E402
from rag_agent.serialization.caption import with_page_title                      # noqa: E402
from scripts.opt_stage1 import ALPHA, MODEL, REVISION, SPLIT, TYPES, evaluate, paired  # noqa: E402
from scripts.opt_stage2 import READER, READER_REV, SEED, STAGE1                  # noqa: E402
from scripts.retrieval_accuracy import PAGE_TITLES, build_corpus, load_queries  # noqa: E402

CAP_SYSTEM = "You write short descriptive captions for statistical tables."
CAP_MAX_TOKENS = 96
TS = (1, 3, 5)


def cap_user(title, cols, rows, vals):
    return (f"Title: {title}\nColumn headers: {cols}\nRow headers: {rows}\nExample values: {vals}\n\n"
            "Write one sentence (at most 40 words) describing what this table reports: the subject, the measures, "
            "the breakdowns, and the time period if shown. Use only the information above. Output only the sentence.")


def fields(tab):
    """열 경로(열 순서), 행 잎 라벨(행 순서), 비어 있지 않은 데이터 셀 값(행 우선). 빈 라벨은 뺀다."""
    t = tab.table
    cols = [join_path(t.col_path(j)) for j in range(t.n_cols)]
    rows = [fmt_value(t.row_path(i)[-1]) if t.row_path(i) else "" for i in range(t.n_rows)]
    vals = [fmt_value(t.data[i][j]) for i in range(t.n_rows) for j in range(t.n_cols) if str(t.data[i][j]).strip()]
    return [c for c in cols if c], [r for r in rows if r], vals


def table_text(title, caption, cols, rows):
    return f"{title}. {caption} Columns: {'; '.join(cols)}. Rows: {'; '.join(rows)}."


def generate_captions(tids, info, out, caption_limit):
    cfg = {"reader": READER, "revision": READER_REV, "quantization": "4bit", "max_new_tokens": CAP_MAX_TOKENS,
           "temperature": 0.0, "seed": SEED, "system_sha256": digest(CAP_SYSTEM),
           "user_template_sha256": digest(cap_user("{title}", "{COLS}", "{ROWS}", "{VALS}"))}
    cfg_path, cap_path = out / "caption_config.json", out / "captions.jsonl"
    if cfg_path.exists():
        assert json.loads(cfg_path.read_text()) == cfg, "caption config changed — use a new --out-dir"
    else:
        cfg_path.write_text(json.dumps(cfg, indent=2))
    load = lambda: {r["table_id"]: r for r in map(json.loads, cap_path.open())} if cap_path.exists() else {}
    todo = [tid for tid in tids if tid not in load()]
    if caption_limit:
        todo = todo[:max(0, caption_limit - len(load()))]
    if todo:
        import torch
        from rag_agent.llm.local_qwen import LocalQwenLLM
        torch.manual_seed(SEED)
        llm = LocalQwenLLM(model_name=READER, quantization="4bit", revision=READER_REV)
        meta_path = out / "reader_metadata.json"
        if meta_path.exists():
            assert json.loads(meta_path.read_text())["revision_resolved"] == llm.metadata()["revision_resolved"]
        else:
            meta_path.write_text(json.dumps(llm.metadata(), indent=2))
        t0 = time.time()
        with cap_path.open("a", encoding="utf-8") as fh:
            for k, tid in enumerate(todo, 1):
                title, cols, rows, vals = info[tid]
                raw = llm.complete(CAP_SYSTEM, cap_user(title, "; ".join(list(dict.fromkeys(cols))[:20]),
                                                        "; ".join(list(dict.fromkeys(rows))[:20]), "; ".join(vals[:5])),
                                   max_tokens=CAP_MAX_TOKENS, temperature=0.0)
                lines = raw.strip().splitlines()
                fh.write(json.dumps({"table_id": tid, "output": raw, "caption": lines[0].strip() if lines else ""},
                                    ensure_ascii=False) + "\n")
                fh.flush()
                if k % 50 == 0:
                    print(f"  caption {k}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    return load()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--cache-dir", default=".cache/retrieval_accuracy")
    ap.add_argument("--out-dir", default="results/opt/stage5")
    ap.add_argument("--limit", type=int, default=0, help="스모크 전용: dev 질의 앞 N건")
    ap.add_argument("--caption-limit", type=int, default=0, help="스모크 전용: 캡션을 앞 K표만 생성, 나머지는 빈 캡션")
    a = ap.parse_args()
    out = ROOT / a.out_dir
    if (out / "summary.json").exists():
        ap.error(f"{out}/summary.json exists — outputs are never overwritten")
    if (a.limit or a.caption_limit) and out.resolve() == (ROOT / "results/opt/stage5").resolve():
        ap.error("--limit/--caption-limit write only outside results/opt/stage5")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    split = json.loads(SPLIT.read_text())
    dev_type = {q["query_id"]: q["type"] for q in split["dev_queries"]}
    queries = load_queries(a.data_dir, "test", {})
    tids = sorted({q["table_id"] for q in queries})
    scored = [q for q in queries if q["query_id"] in dev_type and not q["excluded"]]
    if a.limit:
        scored = scored[:a.limit]
    page_titles = json.loads(PAGE_TITLES.read_text())
    info = {}
    for tid in tids:
        tab = hg.load_table(tid, a.data_dir)
        info[tid] = (with_page_title(tab.title, page_titles.get(tid)), *fields(tab))

    caps = generate_captions(tids, info, out, a.caption_limit)
    assert a.caption_limit or set(caps) == set(tids), "captions missing for some tables"
    print(f"[captions] {len(caps)} ({time.time() - t0:.0f}s)", flush=True)

    enc = default_encoder(model_name=MODEL, revision=REVISION)
    tokenizer, limit = enc.model.tokenizer, int(enc.model.max_seq_length)
    n_tok = lambda s: len(tokenizer(enc.passage_prefix + s, truncation=False, padding=False,
                                    add_special_tokens=True)["input_ids"])
    tab_texts, trunc = [], {}
    for tid in tids:
        title, cols, rows, _ = info[tid]
        cols, rows = list(cols), list(rows)
        cap = caps[tid]["caption"] if tid in caps else ""
        n_cols, n_rows = len(cols), len(rows)
        text = table_text(title, cap, cols, rows)
        while n_tok(text) > limit and (rows or cols):
            (rows if rows else cols).pop()                # 행 라벨 뒤에서부터, 그다음 열 경로 뒤에서부터
            text = table_text(title, cap, cols, rows)
        if (len(cols), len(rows)) != (n_cols, n_rows):
            trunc[tid] = {"rows_dropped": n_rows - len(rows), "cols_dropped": n_cols - len(cols)}
        tab_texts.append(text)
    audit = {"tables": enc.audit_inputs(tab_texts, overflow="error")}
    with (out / "table_texts.jsonl").open("w", encoding="utf-8") as fh:
        fh.writelines(json.dumps({"table_id": tid, "text": x}, ensure_ascii=False) + "\n" for tid, x in zip(tids, tab_texts))
    E_tab = enc.encode(tab_texts)
    bm_tab = SparseBM25(_tokenize(x) for x in tab_texts)

    texts, covers, _, unit_tids, _ = build_corpus(a.data_dir, tids, "s3c", "cell", page_titles)
    unit_of = {next(iter(c)): k for k, c in enumerate(covers)}
    units_of = {}
    for k, tid in enumerate(unit_tids):
        units_of.setdefault(tid, []).append(k)
    units_of = {tid: np.array(v) for tid, v in units_of.items()}
    key = digest({"texts": texts, "encoder": enc.metadata(), "overflow": "error"})[:24]
    f = ROOT / a.cache_dir / f"{enc.name.replace('/', '_')}_{len(texts)}_{key}.npy"
    assert f.exists(), f"baseline embedding cache missing: {f.name}"
    E = np.load(f)
    bm = SparseBM25(_tokenize(x) for x in texts)
    audit["queries"] = enc.audit_inputs([q["question"] for q in scored], query=True, overflow="error")
    stage1 = read_records(STAGE1)
    print(f"[retrieval] tables {len(tids)} (truncated {len(trunc)}) · cells {len(texts)} ({time.time() - t0:.0f}s)", flush=True)

    n = len(texts)
    recs, mismatch = [], []
    for k, q in enumerate(scored, 1):
        qv = enc.encode_query([q["question"]])[0].astype(np.float32)
        toks = _tokenize(q["question"])
        d, b = E @ qv, bm.get_scores(toks)
        gold = [unit_of.get(c) for c in sorted(q["gold"])]
        base = evaluate(ALPHA * _minmax(d) + (1 - ALPHA) * _minmax(b), gold, q["mode"], unit_tids, q["table_id"])
        s1 = stage1[q["query_id"]]["baseline"]
        if any(base[x] != s1[x] for x in ("strict", "stable", "rr", "top20")):
            mismatch.append(q["query_id"])
        s_tab = ALPHA * _minmax(E_tab @ qv) + (1 - ALPHA) * _minmax(bm_tab.get_scores(toks))
        order_tab = np.argsort(-s_tab, kind="stable")
        row = {"query_id": q["query_id"], "type": dev_type[q["query_id"]], "mode": q["mode"], "m": len(gold),
               "gold_table_rank": int(np.flatnonzero(order_tab == tids.index(q["table_id"]))[0]) + 1,
               "baseline": base}
        for T in TS:
            top = [tids[i] for i in order_tab[:T]]
            sel = np.sort(np.concatenate([units_of[t] for t in top]))
            s = np.full(n, -np.inf, dtype=np.float64)
            s[sel] = ALPHA * _minmax(d[sel]) + (1 - ALPHA) * _minmax(b[sel])
            e = evaluate(s, gold, q["mode"], unit_tids, q["table_id"])
            if not any(g is not None and np.isfinite(s[g]) for g in gold):
                e["rr"] = 0.0
            if e.get("gap") is not None and not np.isfinite(e["gap"]):
                e["gap"] = None
            e["gold_table_in_top"] = int(q["table_id"] in top)
            e["top_tables"] = top
            row[f"T{T}"] = e
        recs.append(row)
        if k % 100 == 0:
            print(f"  {k}/{len(scored)} ({time.time() - t0:.0f}s)", flush=True)
    if mismatch:
        print(f"STOP: baseline differs from STAGE 1 records on {len(mismatch)} queries: {mismatch[:10]}", flush=True)
        return 2

    conds = ["baseline", *[f"T{T}" for T in TS]]
    groups = {t: [r for r in recs if r["type"] == t] for t in TYPES}
    groups = {g: v for g, v in groups.items() if v}
    table, comps = {}, {}
    for g, v in groups.items():
        table[g] = {c: {"n": len(v), "strict": sum(r[c]["strict"] for r in v), "stable": sum(r[c]["stable"] for r in v),
                        "mrr": float(np.mean([r[c]["rr"] for r in v])),
                        **({"gold_table_in_top": sum(r[c]["gold_table_in_top"] for r in v)} if c != "baseline" else {})}
                    for c in conds}
        comps[g] = {}
        for c in conds[1:]:
            comps[g][c] = {m: paired([r[c][m] for r in v], [r["baseline"][m] for r in v]) for m in ("strict", "stable")}
            mrr = paired([r[c]["rr"] for r in v], [r["baseline"]["rr"] for r in v])
            comps[g][c]["mrr"] = {"delta": mrr["delta"], "ci95": mrr["ci95"]}
    judgment = {}
    if "single" in comps:
        ps = sorted(conds[1:], key=lambda c: comps["single"][c]["strict"]["p_mcnemar"])
        run = 0.0
        for i, c in enumerate(ps):
            s = comps["single"][c]["strict"]
            run = max(run, min(1.0, (len(ps) - i) * s["p_mcnemar"]))
            judgment[c] = {**s, "p_holm": run, "criterion_met": s["delta"] > 0 and s["ci95"][0] > 0 and run < .05}
        judgment = {c: judgment[c] for c in conds[1:]}
    fails = {c: dict(Counter(r[c]["fail"] for r in groups.get("single", []) if "fail" in r[c])) for c in conds}

    with (out / "records.jsonl").open("x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
    empty_caps = sum(not caps[t]["caption"] for t in caps)
    summary = {
        "prereg": "results/opt/PREREGISTER_OPT.md §8", "prereg_sha256": file_digest(ROOT / "results/opt/PREREGISTER_OPT.md"),
        "provenance": provenance(ROOT), "arguments": vars(a), "split_sha256": file_digest(SPLIT),
        "caption_config": json.loads((out / "caption_config.json").read_text()),
        "captions": {"n": len(caps), "empty": empty_caps, "sha256": file_digest(out / "captions.jsonl")},
        "table_texts": {"sha256": file_digest(out / "table_texts.jsonl"), "n_truncated": len(trunc),
                        "rows_dropped": sum(x["rows_dropped"] for x in trunc.values()),
                        "cols_dropped": sum(x["cols_dropped"] for x in trunc.values()), "truncated": trunc},
        "encoder": enc.metadata(), "alpha": ALPHA, "input_audit": audit,
        "mrr_rule_gold_table_outside_top_T": 0.0,
        "baseline_check": {"reference": str(STAGE1.relative_to(ROOT)), "reference_sha256": file_digest(STAGE1),
                           "queries_checked": len(recs), "mismatch": 0},
        "population": {g: len(v) for g, v in groups.items()}, "table": table, "vs_baseline": comps,
        "judgment_single_strict_holm3": judgment, "single_fail_classes": fails,
        "records_sha256": file_digest(out / "records.jsonl"), "seconds": round(time.time() - t0),
    }
    with (out / "summary.json").open("x", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    pct = lambda k2, n2: f"{k2}/{n2} = {k2 / n2:.4f}"
    f3 = lambda s: f"{s['delta']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] {s['b']}:{s['c']} {s['p_mcnemar']:.3g}"
    lines = ["# STAGE 5 결과 — dev (해석 없음)", "",
             f"사전등록 `PREREGISTER_OPT.md` §8. baseline 재계산 = STAGE 1 기록 {len(recs)}건 불일치 0. 캡션 {len(caps)} (빈 캡션 {empty_caps}), "
             f"토큰 한도로 줄인 표 텍스트 {len(trunc)}. gold 표가 상위 T 밖이면 MRR 0.", "",
             "## 판정 — single top-m 엄격 (Holm 3)", "",
             "| 조건 | Δ | 95% CI | b:c | McNemar p | Holm p | 기준 충족 |", "|---|---:|---|---:|---:|---:|---|"]
    for c, j in judgment.items():
        lines.append(f"| {c} | {j['delta']:+.4f} | [{j['ci95'][0]:+.4f}, {j['ci95'][1]:+.4f}] | {j['b']}:{j['c']} | "
                     f"{j['p_mcnemar']:.3g} | {j['p_holm']:.3g} | {'예' if j['criterion_met'] else '아니오'} |")
    for m, name in (("strict", "top-m 엄격"), ("stable", "top-m 안정 정렬")):
        lines += ["", f"## {name}", "", "| 유형 | " + " | ".join(conds) + " |", "|---|" + "---:|" * len(conds)]
        lines += [f"| {g} | " + " | ".join(pct(table[g][c][m], table[g][c]["n"]) for c in conds) + " |" for g in table]
    lines += ["", "## MRR · gold 표가 상위 T 안", "", "| 유형 | " + " | ".join(conds) + " | " +
              " | ".join(f"gold 표 ∈ {c}" for c in conds[1:]) + " |", "|---|" + "---:|" * (2 * len(conds) - 1)]
    lines += [f"| {g} | " + " | ".join(f"{table[g][c]['mrr']:.4f}" for c in conds) + " | " +
              " | ".join(pct(table[g][c]["gold_table_in_top"], table[g][c]["n"]) for c in conds[1:]) + " |" for g in table]
    lines += ["", "## 기준 대비 전 조건 × 전 유형 (판정 외)", "",
              "| 조건 | 유형 | 엄격 Δ [CI] b:c p | 안정 Δ [CI] b:c p |", "|---|---|---|---|"]
    lines += [f"| {c} | {g} | {f3(comps[g][c]['strict'])} | {f3(comps[g][c]['stable'])} |" for g in comps for c in conds[1:]]
    lines += ["", "## single 실패 분류", "", f"{fails}"]
    (out / "STAGE5_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
