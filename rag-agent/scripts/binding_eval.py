#!/usr/bin/env python3
"""교수님 가설 검증: 표 구조 메타데이터(헤더경로 인벤토리) 주입이 binding fault를 줄이는가?

method_grounded.py 의 naive vs grounded 비교를 *모든 벤치마크*(hitab/finqa/wikisql)로
확장한다. 단일 변수 = 메타데이터 유무:
  --mode naive    : 헤더 인벤토리 없음 (모델이 헤더 문자열 자유추측)
  --mode grounded : ROW/COL 헤더경로 인벤토리 주입 + grounding-trace 자가수정

검색은 gold(정답표 제공)로 고정 → 답변/바인딩 단계만 격리. 동일 LLM·seed로 두 모드를
돌리므로 비교는 내적으로 유효하다.

측정:
  acc              최종 정확도(numeric/exact match)
  nm_rate          NO_MATCH(바인딩 실패) 발생 쿼리 비율  ← 교수님 가설의 핵심 타깃
  empty_rate       EMPTY(빈 셀) 발생 비율
  clean_bind       NO_MATCH/EMPTY 둘 다 없는 비율
  acc_given_clean  바인딩 깨끗한 쿼리에서의 정확도  ← 잔여 병목(operand/계산) 노출

사용: .venv/bin/python scripts/binding_eval.py --benches hitab,finqa,wikisql --n 50
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_agent.bench import registry                                  # noqa: E402
from rag_agent.bench.schema import BenchTable                         # noqa: E402
from rag_agent.stores.original_store import OriginalTable             # noqa: E402
from rag_agent.eval.metrics import numeric_match, exact_match         # noqa: E402
# method_grounded 의 검증된 부품 재사용 (재발명 금지)
from scripts.method_grounded import (                                 # noqa: E402
    TracedTable, run_code, strip_code, build_user, trace_feedback,
    needs_repair, NAIVE_SYS, GROUNDED_SYS,
)

SEED = 42


def bench_to_original(bt: BenchTable) -> OriginalTable:
    """BenchTable → OriginalTable (필드 동일; 매칭/해소 로직 그대로 재사용)."""
    return OriginalTable(
        table_id=bt.table_id, title=bt.title or "", data=bt.data,
        top_paths=bt.top_paths, left_paths=bt.left_paths,
    )


def score(pred: str, answer) -> bool:
    """gold answer(list 가능)의 어느 원소와라도 매칭되면 정답."""
    golds = answer if isinstance(answer, (list, tuple)) else [answer]
    for g in golds:
        try:
            if numeric_match(pred, g) or exact_match(pred, str(g)):
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def build_user_from_meta(title, question, meta):
    """교수님 방식: 디스크에 *저장된* 헤더경로 메타(cap 없음)로 프롬프트 구성."""
    s = f"Table title: {title}\nQuestion: {question}\n"
    s += f"\n(table depth: row={meta.get('row_depth')}, col={meta.get('col_depth')})"
    s += "\nROW HEADERS (row_header 후보):\n- " + "\n- ".join(meta.get("row_paths") or ["(none)"])
    s += "\n\nCOL HEADERS (col_header 후보):\n- " + "\n- ".join(meta.get("col_paths") or ["(none)"])
    s += "\n\nWrite python that sets `result` to the answer value."
    return s


def run_one(llm, q, ot, grounded, repairs, max_tokens, meta=None):
    tt = TracedTable(ot)
    api = {"cell": tt.cell, "col_values": tt.col_values, "row_values": tt.row_values,
           "list_rows": tt.list_rows, "list_cols": tt.list_cols}
    sys_p = GROUNDED_SYS if grounded else NAIVE_SYS
    if grounded and meta is not None:
        user = build_user_from_meta(ot.title, q.question, meta)   # 저장된 메타 사용
    else:
        user = build_user(ot.title, q.question, tt, grounded=grounded, binding_hint="")
    code = strip_code(llm.complete(sys_p, user, max_tokens=max_tokens))
    tt.trace = []
    result, err = run_code(code, api)
    n_repair = 0
    if grounded:
        while n_repair < repairs and needs_repair(tt.trace, result, err):
            fb = trace_feedback(code, tt.trace, result, err)
            code = strip_code(llm.complete(sys_p, user + "\n\n" + fb, max_tokens=max_tokens))
            tt.trace = []
            result, err = run_code(code, api)
            n_repair += 1
    pred = "" if result is None else str(result)
    flags = [t["flag"] for t in tt.trace if t["flag"]]
    return {
        "query_id": q.query_id, "query": q.question, "gold": q.answer, "pred": pred,
        "correct": score(pred, q.answer), "err": err, "n_repair": n_repair,
        "trace_flags": flags, "code": code,
    }


def _llm_failed(r) -> bool:
    """LLM 호출 자체가 실패한 행(레이트리밋 소진 등) — 바인딩 지표에서 제외."""
    e = str(r.get("err") or "")
    return e.startswith("HARNESS") or "retries exhausted" in e or "code" not in r


def summarize(rows):
    n_all = len(rows)
    rows = [r for r in rows if not _llm_failed(r)]   # 유효(LLM 응답) 행만
    n = len(rows)
    if n == 0:
        return {"n": 0, "n_all": n_all, "n_llm_failed": n_all}
    def has(r, f):
        return f in (r.get("trace_flags") or [])
    acc = sum(r["correct"] for r in rows) / n
    nm = sum(1 for r in rows if has(r, "NO_MATCH")) / n
    empty = sum(1 for r in rows if has(r, "EMPTY")) / n
    clean = [r for r in rows if not (r.get("trace_flags") or [])]
    clean_bind = len(clean) / n
    acc_clean = (sum(r["correct"] for r in clean) / len(clean)) if clean else None
    return {"n": n, "n_all": n_all, "n_llm_failed": n_all - n,
            "acc": round(acc, 4), "nm_rate": round(nm, 4),
            "empty_rate": round(empty, 4), "clean_bind": round(clean_bind, 4),
            "acc_given_clean": round(acc_clean, 4) if acc_clean is not None else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", default="hitab,finqa,wikisql")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--modes", default="naive,grounded")
    ap.add_argument("--repairs", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--model", default="llama-3.3-70b-versatile")
    ap.add_argument("--backend", choices=["groq", "local"], default="groq")
    ap.add_argument("--meta-store", default=None,
                    help="저장된 표 메타 디렉토리(data/table_meta). 주면 grounded가 즉석추출 대신 저장본 사용")
    ap.add_argument("--out-dir", default="results/binding_eval")
    args = ap.parse_args()

    if args.backend == "local":
        from rag_agent.llm.local_qwen import LocalQwenLLM
        llm = LocalQwenLLM(default_max_tokens=args.max_tokens)
    else:
        from rag_agent.llm.groq_llm import GroqLLM
        llm = GroqLLM(model_name=args.model)
    print(f"[binding_eval] LLM={llm.name} n={args.n} modes={args.modes}", flush=True)

    outdir = ROOT / args.out_dir
    outdir.mkdir(parents=True, exist_ok=True)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    summary = {"config": {"model": llm.name, "n": args.n, "repairs": args.repairs,
                          "seed": SEED, "retrieval": "gold"}, "results": {}}

    for bench in [b.strip() for b in args.benches.split(",") if b.strip()]:
        print(f"\n=== {bench} ===", flush=True)
        queries, tables = registry.load(bench, max_samples=None)
        rng = random.Random(SEED)
        pool = [q for q in queries
                if q.gold_table_id in tables and tables[q.gold_table_id].data]
        rng.shuffle(pool)
        chosen = pool[:args.n]
        print(f"  pool={len(pool)} chosen={len(chosen)}", flush=True)

        summary["results"][bench] = {}
        for mode in modes:
            grounded = mode == "grounded"
            rows = []
            t0 = time.time()
            for i, q in enumerate(chosen, 1):
                ot = bench_to_original(tables[q.gold_table_id])
                meta = None
                if args.meta_store:
                    from scripts.build_table_meta import safe_name
                    mp = ROOT / args.meta_store / bench / f"{safe_name(q.gold_table_id)}.json"
                    if mp.exists():
                        meta = json.loads(mp.read_text())
                try:
                    rows.append(run_one(llm, q, ot, grounded, args.repairs, args.max_tokens, meta=meta))
                except Exception as e:  # noqa: BLE001
                    rows.append({"query_id": q.query_id, "query": q.question,
                                 "correct": False, "err": f"HARNESS:{type(e).__name__}:{e}",
                                 "trace_flags": [], "n_repair": 0})
                if i % 10 == 0 or i == len(chosen):
                    acc = sum(r["correct"] for r in rows) / len(rows)
                    print(f"  [{bench}/{mode}] {i}/{len(chosen)} acc={acc:.3f} {time.time()-t0:.0f}s",
                          flush=True)
            s = summarize(rows)
            summary["results"][bench][mode] = s
            (outdir / f"{bench}_{mode}.json").write_text(
                json.dumps({"bench": bench, "mode": mode, "summary": s, "rows": rows},
                           ensure_ascii=False, indent=2))
            print(f"  [{bench}/{mode}] {json.dumps(s, ensure_ascii=False)}", flush=True)

    # delta (grounded - naive) per bench
    if "naive" in modes and "grounded" in modes:
        summary["delta"] = {}
        for bench, r in summary["results"].items():
            nv, gr = r.get("naive"), r.get("grounded")
            if nv and gr:
                summary["delta"][bench] = {
                    k: round(gr[k] - nv[k], 4)
                    for k in ["acc", "nm_rate", "clean_bind", "acc_given_clean"]
                    if nv.get(k) is not None and gr.get(k) is not None
                }
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary.get("delta", summary["results"]), ensure_ascii=False, indent=2), flush=True)
    print(f"→ {outdir/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
