"""dev 산술 264(본 방법 입력): 실패/성공별 특징 — 가설 점검용, GPU 없음."""
import json, re, sys, pickle
from collections import defaultdict, Counter
from pathlib import Path
ROOT = Path(".").resolve(); sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "results/cellform_20261012")]
from mh_arms import build_tables, cell_id, load_population
from retrieval_accuracy import build_corpus
from score_inclusion import included, nums
from rag_agent.eval.multihiertt_em import format_match
import mh_arms
from huggingface_hub import hf_hub_download
D = ROOT / "results/cellform_dev_20261012"
recs = {r["query_id"]: r for r in map(json.loads, open(D / "C_records.jsonl"))}
ans = {r["query_id"]: r for r in map(json.loads, open(D / "ans_C/langchain_chunk_answer.jsonl"))}
ids = list(recs)
p = hf_hub_download(mh_arms.OFFICIAL_REPO, "multihiertt_data/dev.json", repo_type="dataset", revision=mh_arms.OFFICIAL_REV)
QA = {x["uid"]: x["qa"] for x in json.load(open(p))}
_q, docs, _ = load_population("validation")
tables, hdr = build_tables({u: docs[u] for u in ids}, "v3.3u", "none")
texts, covers, _, unit_tids, _ = build_corpus("", sorted(tables), "s3c", "cell", {}, load=lambda t, _d: tables.get(t))
sent = defaultdict(dict)
for txt, cov, tid in zip(texts, covers, unit_tids):
    c = next(iter(cov)); sent[tid.split("::")[0]][cell_id(c, hdr).split("::", 1)[1]] = txt
pickle.dump({"sent": dict(sent), "QA": QA}, open(sys.argv[1], "wb"))
