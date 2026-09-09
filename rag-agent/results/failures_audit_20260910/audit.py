import json,hashlib
from pathlib import Path
from collections import Counter
from rag_agent.bench.hitab_grid import load_table
from rag_agent.eval.metrics import hitab_exact_match_text
from rag_agent.serialization.base import fmt_value
src=Path("/mnt/c/Users/user/Desktop/FAILURES.json")
d=json.loads(src.read_text())
recs=Path("results/retrieval_accuracy/t_s3c_hybrid_records.jsonl")
rs=[r for r in map(json.loads,recs.read_text().splitlines()) if "correct" in r and r["mode"]=="all" and (r.get("aggregation") or "none")=="none" and r["m"]==1]
cs={c["query_id"]:c for c in d["cases"]}; cache={}; rows=[]
for r in rs:
 tid=r["table_id"]
 if tid not in cache:cache[tid]=load_table(tid,"data/hitab")
 t=cache[tid].table
 _,i,j=r["gold_cells"][0]
 val=fmt_value(t.data[i][j])
 rows.append({"query_id":r["query_id"],"gold_cell":r["gold_cells"][0],"copied_value":val,"answer":r["answer"],"copy_correct":int(hitab_exact_match_text(val,r["answer"])),"original_bucket":cs.get(r["query_id"],{}).get("bucket")})
report={
 "diagnostic":"Oracle gold-cell value copy, not model inference or deployed EM",
 "limitation":"Uses annotated gold coordinates. This diagnoses value availability and scoring compatibility only; selecting the correct cell from retrieved candidates remains unsolved.",
 "n":len(rows),"correct":sum(r["copy_correct"] for r in rows),
 "reader_ceiling_copy_correct":sum(r["copy_correct"] for r in rows if r["original_bucket"]=="reader_ceiling"),
 "buckets":dict(Counter(c["bucket"] for c in cs.values())),
 "distractor_value_match_labels":dict(Counter(c.get("misread",{}).get("where") for c in cs.values() if c["bucket"]=="distractor")),
 "label_caveat":"Value matching does not prove which cell the model attended to; gold-dependent buckets use outputs before title bug fix.",
 "input_sha256":hashlib.sha256(src.read_bytes()).hexdigest(),
 "records_sha256":hashlib.sha256(recs.read_bytes()).hexdigest(),
 "scorer_sha256":hashlib.sha256(Path("rag_agent/eval/metrics.py").read_bytes()).hexdigest(),
 "rows":rows
}
out=Path("results/failures_audit_20260910");out.mkdir(exist_ok=True)
(out/"GOLD_COPY_DIAGNOSTIC.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!="rows"},ensure_ascii=False,indent=2))
