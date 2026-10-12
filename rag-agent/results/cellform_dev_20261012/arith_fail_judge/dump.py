import json, sys, pickle, random
sys.path[:0] = [".", "scripts", "results/cellform_20261012"]
from score_inclusion import included, nums
P = pickle.load(open(sys.argv[1], "rb")); sent, QA = P["sent"], P["QA"]
D = "results/cellform_dev_20261012"
recs = {r["query_id"]: r for r in map(json.loads, open(f"{D}/C_records.jsonl"))}
fails = []
for r in map(json.loads, open(f"{D}/ans_C/langchain_chunk_answer.jsonl")):
    if r["kind"] != "arith" or not recs[r["query_id"]]["correct"]: continue
    if not included(r["raw"], QA[r["query_id"]]["answer"], True): fails.append(r)
fails.sort(key=lambda r: r["query_id"])
random.seed(20261012); samp = random.sample(fails, 60)
def fl(t):
    try: return abs(float(t.strip("%()")))
    except ValueError: return None
out = []
for i, r in enumerate(samp, 1):
    q = r["query_id"]; R = recs[q]; qa = QA[q]; lines = R["context"].split("\n")
    gs = [sent[q][c] for c in R["gold_ids"]]
    ov = {fl(t) for t in nums(r["raw"])} - {None}
    used = [(j, l) for j, l in enumerate(lines) if l not in gs and any(fl(t) in ov for t in nums(l.rsplit(":", 1)[-1]) if fl(t))]
    out.append(dict(no=i, query_id=q, question=qa["question"], program=qa["program"], answer=qa["answer"],
                    gold=[(lines.index(s), s) for s in gs], used_nongold=used[:6], raw=r["raw"], n_lines=len(lines)))
json.dump(out, open(sys.argv[2], "w"), ensure_ascii=False, indent=0)
print(len(fails), len(out))
