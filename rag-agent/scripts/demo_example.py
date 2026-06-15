#!/usr/bin/env python3
"""Pick a real OWT query rescued by preprocessing and dump the actual
serialize() output (C0 vs C3) the pipeline produces for its gold table."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_agent.prep.conditions import from_openwikitable, serialize
from rag_agent.prep.synth import TemplateSynth

base = Path(__file__).resolve().parents[1]
res = json.load(open(base / "results/prep/owt_bm25_n1000.json"))
pq = res["per_query"]
qids, C0, C1, C3 = pq["question_ids"], pq["C0"], pq["C1"], pq["C3"]

qtext, qgold = {}, {}
with open(base / "data/openwikitable/queries_test.jsonl") as f:
    for line in f:
        r = json.loads(line)
        qtext[r["question_id"]] = r["question"]
        qgold[r["question_id"]] = r["gold_table_id"]

corpus = {}
with open(base / "data/openwikitable/corpus.jsonl") as f:
    for line in f:
        r = json.loads(line)
        corpus[str(r["table_id"])] = r

# rescued = C0 fails (None or rank>10) but C3 finds it at rank 1
cands = []
for i, qid in enumerate(qids):
    c0 = C0[i] if C0[i] is not None else 10**9
    if C3[i] == 1 and c0 > 10 and qid in qgold and str(qgold[qid]) in corpus:
        rec = corpus[str(qgold[qid])]
        ncols = len(rec.get("header", []))
        nrows = len(rec.get("rows", []))
        cands.append((i, qid, c0, C1[i], ncols, nrows))

# prefer a small, readable table
cands.sort(key=lambda x: (x[4] * 1000 + x[5]))
print(f"{len(cands)} rescued candidates\n")
synth = TemplateSynth(n_questions=5)
for i, qid, c0, c1, nc, nr in cands[:4]:
    rec = corpus[str(qgold[qid])]
    pt = from_openwikitable(rec)
    print("=" * 70)
    print("QUERY :", qtext[qid])
    print("GOLD TABLE id:", qgold[qid], f"| {nc} cols x {nr} rows")
    print(f"RANK  : C0={'>1000' if c0>=10**9 else c0}  C1={c1}  C3=1")
    print("\n--- serialize(C0) : raw table only ---")
    print(serialize(pt, "C0", max_rows=6))
    print("\n--- serialize(C3) : +metadata +schema +synthetic questions ---")
    print(serialize(pt, "C3", max_rows=6, synth_provider=synth))
    print()
