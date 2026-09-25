"""2026-09-25 s3c_v2 같은 표 혼동 분석(리더 없음, 임베딩만).
정답 셀과 '정답보다 높은 순위를 받은 같은 표 셀'의 문서 임베딩 코사인, 행·열 경로 중 다른 단계.
모집단 = HiTab test 단일 셀 조회 991문항. 순서는 s3c_v2 와 같은 식(α=.7 hybrid, stable argsort)으로
다시 계산하고, 판정·정답 표 포함이 s3c_v2 레코드와 991문항 전부 같은지 확인한다.
실행: .venv/bin/python results/components_20260925/confusion.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.argv = sys.argv[:1]
import retrieval_accuracy as ra                                       # noqa: E402
from rag_agent.bench import hitab_grid as hg                          # noqa: E402
from rag_agent.eval.artifacts import digest                           # noqa: E402

OUT = Path(__file__).parent
SRC = ROOT / "results/evaluation_v2/s3c_v2.json"
cfg = json.loads(SRC.read_text())
args = cfg["arguments"]
recs = {r["query_id"]: r for r in map(json.loads, open(ROOT / "results/evaluation_v2/s3c_v2_records.jsonl"))
        if "excluded" not in r and r["mode"] == "all" and r["m"] == 1
        and (r.get("aggregation") or "none") == "none"}
assert len(recs) == 991

page_titles = json.loads(ra.PAGE_TITLES.read_text())
queries = ra.load_queries("data/hitab", "test", {})
tids = sorted({q["table_id"] for q in queries})
texts, covers, _, unit_tids, _ = ra.build_corpus("data/hitab", tids, "s3c", "cell", page_titles)
assert digest(texts) == cfg["corpus_text_sha256"]
cell = [next(iter(c)) for c in covers]
idx_of = {c: k for k, c in enumerate(cell)}
unit_tid = np.array(unit_tids)

bm = ra.SparseBM25(ra._tokenize(t) for t in texts)
enc = ra.default_encoder(model_name=args["embed_model"], revision=args["embed_revision"])
key = digest({"texts": texts, "encoder": enc.metadata(), "overflow": args["embed_overflow"]})[:24]
cache = ROOT / args["cache_dir"] / f"{enc.name.replace('/', '_')}_{len(texts)}_{key}.npy"
emb = np.load(cache) if cache.exists() else enc.encode(texts)
print("cache hit" if cache.exists() else "encoded", cache.name, flush=True)

tables = {}


def paths(c):
    tid, i, j = c
    if tid not in tables:
        tables[tid] = hg.load_table(tid, "data/hitab").table
    t = tables[tid]
    return list(t.row_path(i)), list(t.col_path(j))


def level_diff(a, b):
    """같음 / 잎만 다름 / 상위 단계 다름(깊이가 다르거나 잎이 아닌 단계가 다름)"""
    if a == b:
        return "same"
    if len(a) == len(b) and a[:-1] == b[:-1]:
        return "leaf_only"
    return "upper"


rows = []
for q in queries:
    r = recs.get(q["query_id"])
    if r is None:
        continue
    s = bm.get_scores(ra._tokenize(q["question"]))
    d = emb @ enc.encode_query([q["question"]])[0].astype(np.float32)
    s = args["alpha"] * ra._minmax(d) + (1 - args["alpha"]) * ra._minmax(s)
    order = np.argsort(-s, kind="stable")
    g = idx_of[tuple(r["gold_cells"][0])]
    rank = int(np.flatnonzero(order == g)[0])                     # 0-based
    top = order[:20]
    assert int(g in top) == r["correct"], q["query_id"]
    assert int(any(unit_tid[k] == q["table_id"] for k in top)) == r["gold_table_in_context"]
    above = [int(k) for k in order[:rank] if unit_tid[k] == q["table_id"]]
    cos = [float(emb[g] @ emb[k]) for k in above]
    gr, gc = paths(cell[g])
    diffs = []
    for k in above:
        kr, kc = paths(cell[k])
        diffs.append([level_diff(gr, kr), level_diff(gc, kc), texts[k] == texts[g]])
    rows.append({"query_id": q["query_id"], "correct": r["correct"],
                 "table_found": r["gold_table_in_context"], "gold_rank": rank + 1,
                 "n_same_table_above": len(above), "cos": cos, "diffs": diffs})

groups = {"fail_table_found": [x for x in rows if not x["correct"] and x["table_found"]],
          "success": [x for x in rows if x["correct"]]}
assert len(groups["fail_table_found"]) == 45 and len(groups["success"]) == 906


def summarize(rs):
    has = [x for x in rs if x["cos"]]
    per_mean = [float(np.mean(x["cos"])) for x in has]
    per_max = [max(x["cos"]) for x in has]
    top1 = [x["cos"][0] for x in has]
    pairs = [dd for x in has for dd in x["diffs"]]
    first = [x["diffs"][0] for x in has]
    tab = lambda ds: {"row": dict(Counter(d[0] for d in ds)), "col": dict(Counter(d[1] for d in ds)),
                      "row_col": dict(Counter(f"{d[0]}|{d[1]}" for d in ds)),
                      "identical_text": sum(d[2] for d in ds), "n": len(ds)}
    return {"n_queries": len(rs), "n_with_same_table_above": len(has),
            "n_same_table_above_median": float(np.median([x["n_same_table_above"] for x in has])) if has else None,
            "n_same_table_above_mean": float(np.mean([x["n_same_table_above"] for x in has])) if has else None,
            "cos_per_query_mean__mean": float(np.mean(per_mean)), "cos_per_query_mean__median": float(np.median(per_mean)),
            "cos_per_query_max__mean": float(np.mean(per_max)), "cos_per_query_max__median": float(np.median(per_max)),
            "cos_highest_ranked__mean": float(np.mean(top1)),
            "path_diff_all_pairs": tab(pairs), "path_diff_highest_ranked": tab(first)}


result = {"source_records": "results/evaluation_v2/s3c_v2_records.jsonl", "embedding_cache": str(cache.relative_to(ROOT)),
          **{k: summarize(v) for k, v in groups.items()}}
(OUT / "confusion.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
with open(OUT / "confusion_rows.jsonl", "w") as f:
    for x in rows:
        f.write(json.dumps(x) + "\n")
print(json.dumps(result, indent=2))
