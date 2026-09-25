"""GATE 0 추가 3번: '표 검색 97' 후보와 t_table_hybrid 의 조건을 파일에서 읽는다. 추론 없음."""
import collections, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
blob = subprocess.run(["git", "-C", str(ROOT), "show",
                       "f4e6865:rag-agent/results/retrieval_accuracy/t_table_hybrid_records.jsonl"],
                      capture_output=True, text=True, check=True).stdout
sc = [r for r in map(json.loads, blob.splitlines()) if "correct" in r]
pr = [r for r in sc if r["mode"] == "all" and r["m"] == 1 and (r.get("aggregation") or "none") == "none"]
mh = json.load(open(ROOT / "results/strict_fixed_budget/mh_validation_cell_hv3.3_none_doc_fixed_budget20_summary.json"))
print(json.dumps({
    "t_table_hybrid (f4e6865:rag-agent/results/retrieval_accuracy/t_table_hybrid_records.jsonl)": {
        "scored": len(sc),
        "tables_in_context_hist": dict(sorted(collections.Counter(len(r["context"]) for r in sc).items())),
        "primary_991": {"query_count": len(pr), "correct": sum(r["correct"] for r in pr),
                        "accuracy": round(sum(r["correct"] for r in pr) / len(pr), 4),
                        "gold_table_in_context": sum(r["gold_table_in_context"] for r in pr)}},
    "mh_validation DTC (results/strict_fixed_budget/mh_validation_cell_hv3.3_none_doc_fixed_budget20_summary.json)": {
        k: mh[k] for k in ("dataset", "split", "metric", "selector", "table_metric", "header_rule", "encoder",
                           "alpha", "n_docs", "n_tables")} | {"overall": {k: mh["groups"]["overall"][k] for k in
                           ("N", "strict_recall", "derived_table_strict_recall", "derived_table_recall",
                            "derived_table_precision")}},
}, indent=1, ensure_ascii=False))
