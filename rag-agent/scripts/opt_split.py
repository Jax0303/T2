#!/usr/bin/env python3
"""STAGE 0 — HiTab test 를 표 단위로 dev 40% / test 60% 로 나눈다 (seed=42).

성능은 재지 않는다. 데이터셋에서 질의 유형만 읽어 split 별로 센다.
유형: excluded(gold 해석 불가) / any(답이 헤더) / all 모드 중
single(aggregation none, gold 1셀) / multi(none, gold ≥2셀) / arith(aggregation ≠ none).

  PYTHONPATH=. .venv/bin/python scripts/opt_split.py
"""
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from retrieval_accuracy import load_queries                          # noqa: E402
from rag_agent.data.loader import _find_data_root                    # noqa: E402
from rag_agent.eval.artifacts import file_digest                     # noqa: E402

SEED, DEV_RATIO, DATA_DIR, SPLIT = 42, 0.4, "data/hitab", "test"
OUT = ROOT / "results/opt"


def qtype(q):
    if q["excluded"]:
        return "excluded"
    if q["mode"] == "any":
        return "any"
    if q.get("aggregation") != "none":
        return "arith"
    return "single" if len(q["gold"]) == 1 else "multi"


def main():
    split_f, frozen_f = OUT / "split.json", OUT / "TEST_FROZEN.md"
    if split_f.exists() or frozen_f.exists():
        sys.exit(f"refuse to overwrite {split_f} / {frozen_f}")
    tabs = {}
    qs = load_queries(DATA_DIR, SPLIT, tabs)
    tids = sorted({q["table_id"] for q in qs})
    order = tids[:]
    random.Random(SEED).shuffle(order)
    dev_t = set(order[:round(DEV_RATIO * len(tids))])
    side = {tid: ("dev" if tid in dev_t else "test") for tid in tids}

    ids = {"dev": [], "test": []}
    counts = {"dev": Counter(), "test": Counter()}
    agg = Counter()
    for q in qs:
        s, t = side[q["table_id"]], qtype(q)
        ids[s].append({"query_id": q["query_id"], "table_id": q["table_id"], "type": t})
        counts[s][t] += 1
        agg[(t, str(q.get("aggregation")))] += 1
    for s in ids:
        ids[s].sort(key=lambda r: r["query_id"])

    # 누수 점검(관측만): 제목이 dev 표와 글자 그대로 같은 test 표 수
    title = {tid: (tabs[tid].title if tabs.get(tid) else None) for tid in tids}
    dev_titles = {title[t] for t in dev_t}
    same_title = sorted(t for t in tids if side[t] == "test" and title[t] in dev_titles)

    test_ids = [r["query_id"] for r in ids["test"]]
    test_sha = hashlib.sha256("\n".join(test_ids).encode()).hexdigest()
    now = datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")
    data_root = _find_data_root(DATA_DIR)
    summary = {
        "created": now, "seed": SEED, "dev_ratio": DEV_RATIO, "unit": "table",
        "procedure": "tids=sorted(set(table_id)); random.Random(42).shuffle(copy); "
                     "dev=first round(0.4*n)",
        "split_sha256": file_digest(data_root / "data" / f"{SPLIT}_samples.jsonl"),
        "script_sha256": file_digest(Path(__file__)),
        "n_tables": {"all": len(tids), "dev": len(dev_t), "test": len(tids) - len(dev_t)},
        "n_queries": {s: len(ids[s]) for s in ids},
        "by_type": {s: dict(sorted(counts[s].items())) for s in counts},
        "type_x_aggregation": {f"{t}|{a}": n for (t, a), n in sorted(agg.items())},
        "test_tables_with_title_identical_to_a_dev_table": len(same_title),
        "test_query_ids_sha256": test_sha,
        "dev_tables": sorted(dev_t), "test_tables": sorted(set(tids) - dev_t),
        "dev_queries": ids["dev"], "test_queries": ids["test"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    split_f.write_text(json.dumps(summary, ensure_ascii=False, indent=1))

    lines = [
        "# TEST_FROZEN — STAGE 0 동결 test split", "",
        f"- 동결 시각: {now}",
        f"- 분할: 표 단위, seed={SEED}, dev {DEV_RATIO:.0%} / test {1 - DEV_RATIO:.0%}"
        f" — `scripts/opt_split.py` (sha256 {summary['script_sha256']})",
        f"- 원본: `{DATA_DIR}` {SPLIT}_samples.jsonl sha256 {summary['split_sha256']}",
        f"- test 표 {summary['n_tables']['test']} / 질의 {len(test_ids)}",
        f"- test 유형별: {summary['by_type']['test']}",
        f"- test 질의 ID 목록(정렬, 개행 결합) sha256: `{test_sha}`",
        f"- 전체 목록·표 ID: `results/opt/split.json` 의 `test_queries` / `test_tables`",
        "- STAGE 1~5 동안 이 목록의 질의에 대해 어떤 측정도 하지 않는다.", "",
        "## test 질의 ID (query_id  table_id  type)", "", "```",
        *(f"{r['query_id']}  {r['table_id']}  {r['type']}" for r in ids["test"]),
        "```", ""]
    frozen_f.write_text("\n".join(lines))
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("dev_tables", "test_tables", "dev_queries", "test_queries")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
