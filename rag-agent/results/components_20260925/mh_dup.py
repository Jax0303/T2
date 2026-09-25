"""2026-09-25 MultiHiertt c(값+경로) / e(값+경로+라벨) 검색 정확도·문장 중복률.
검색 결과는 재사용(results/mh_arms/mh_cell_hv2*, 머리글 v2, train). 중복률은 같은 코드로 코퍼스를
다시 만들어 요약 JSON 의 corpus_text_sha256 과 같은지 확인한 뒤 센다.
실행: .venv/bin/python results/components_20260925/mh_dup.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.argv = sys.argv[:1]
import mh_arms as mh                                                  # noqa: E402

OUT = Path(__file__).parent
RUNS = [("c", "none", "results/mh_arms/mh_cell_hv2"), ("e", "L1", "results/mh_arms/mh_cell_hv2_L1")]

queries, docs, _ = mh.load_population("train")
result = {}
for k, label, stem in RUNS:
    cfg = json.loads((ROOT / f"{stem}.json").read_text())
    tables, _ = mh.build_tables(docs, "v2", label)
    texts, covers, _, unit_tids, _ = mh.build_corpus(
        "", sorted(tables), "s3c", "cell", {}, load=lambda tid, _d: tables.get(tid))
    assert mh.digest(texts) == cfg["corpus_text_sha256"], stem
    uids = [t.split("::")[0] for t in unit_tids]
    whole = Counter(texts)
    in_doc = Counter(zip(uids, texts))
    per_doc = {}
    for u, t in zip(uids, texts):
        per_doc.setdefault(u, []).append(t)
    recs = [r for r in map(json.loads, open(ROOT / f"{stem}_records.jsonl")) if "excluded" not in r]
    result[k] = {
        "summary": f"{stem}.json", "records": f"{stem}_records.jsonl", "label_rule": label,
        "example": texts[0], "n_queries": len(recs),
        "doc_correct": sum(r["doc"]["correct"] for r in recs),
        "corpus_correct": sum(r["corpus"]["correct"] for r in recs),
        "doc_accuracy_json": cfg["by_layer"]["ALL"]["doc"]["accuracy_all"],
        "corpus_accuracy_json": cfg["by_layer"]["ALL"]["corpus"]["accuracy_all"],
        "cells": len(texts), "docs": len(set(uids)),
        "dup_cells_within_doc": sum(1 for u, t in zip(uids, texts) if in_doc[u, t] > 1),
        "dup_cells_whole_index": sum(1 for t in texts if whole[t] > 1),
        "distinct_texts_whole_index": len(whole),
        "distinct_doc_contents": len({tuple(v) for v in per_doc.values()}),
    }
    print(k, result[k], flush=True)
(OUT / "mh_dup.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
