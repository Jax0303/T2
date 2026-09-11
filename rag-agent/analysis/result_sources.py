"""One source registry for corrected retrieval records and their reader legs."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "results/retrieval_accuracy"
FIXED = ROOT / "results/audit_fix_20260910"
ANSWERS = ROOT / "results/audit_answers_20260910"
FIXED_TAGS = {"t_s3c_hybrid", "t_rowcol_values", "t_chunk1000",
              "t_trag_hetero", "t_trag_hetero_tok", "t_tablerag_leaf", "t_tablerag_path"}
INVALIDATED = {"t_rowcol_values", "t_tablerag_leaf", "t_tablerag_path"}

def retrieval_path(tag, suffix=".json"):
    if tag in FIXED_TAGS:
        return FIXED / (tag + "_fix" + suffix)
    return ORIGINAL / (tag + suffix)

def answer_path(tag, condition="retrieved", suffix=".jsonl"):
    folder = ANSWERS if tag in INVALIDATED and condition == "retrieved" else ORIGINAL
    return folder / (tag + "_answer_" + condition + suffix)

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]

def validate_pair(records, answers):
    source = {r["query_id"]:r for r in records if "correct" in r}
    output = {r["query_id"]:r for r in answers}
    if len(source) != sum("correct" in r for r in records) or len(output) != len(answers):
        raise ValueError("duplicate query IDs")
    if source.keys() != output.keys():
        raise ValueError("reader/retrieval query populations differ")
    for q,a in output.items():
        r = source[q]
        expected = hashlib.md5("\n".join(r["context"]).encode()).hexdigest()[:12]
        if r.get("context_sha") != expected or a.get("context_sha") != expected:
            raise ValueError(f"context mismatch: {q}")
        if a["retrieval_correct"] != r["correct"] or a["mode"] != r["mode"] or a["answer"] != r["answer"]:
            raise ValueError(f"evaluation metadata mismatch: {q}")
    return answers

def current_answer_rows(tag, condition="retrieved"):
    path = answer_path(tag,condition)
    if not path.exists():
        return []
    if tag not in INVALIDATED or condition != "retrieved":
        return read_jsonl(path)
    summary_path = path.with_suffix(".json")
    if not summary_path.exists():
        return []  # an incremental JSONL is not a completed experiment
    summary = json.loads(summary_path.read_text())
    if not summary.get("complete"):
        return []
    records_path = retrieval_path(tag,"_records.jsonl")
    if summary.get("source_sha256") != hashlib.sha256(records_path.read_bytes()).hexdigest():
        raise ValueError(f"stale retrieval source: {tag}")
    rows = validate_pair(read_jsonl(records_path),read_jsonl(path))
    if summary["n"] != len(rows):
        raise ValueError(f"stale answer summary: {tag}")
    return rows
