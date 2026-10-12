"""RAPTOR 잎 청크 셀 단위 상한: 문항별 (잎 어딘가에 온전히 들어 있는 정답 셀 수 / 정답 셀 수) 의 문항 평균. 새 실행 없음(렌더링만)."""
import json, sys, statistics
from pathlib import Path
import importlib.util
spec = importlib.util.spec_from_file_location("raptor_baseline", "scripts/baselines/raptor.py"); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
import tiktoken
recs = {r["query_id"]: r for r in map(json.loads, R.RECORDS.open(encoding="utf-8")) if "doc" in r}
_q, docs, _ = R.load_population("train")
docs = {u: docs[u] for u in recs}
tables, hdr = R.build_tables(docs, "v3.3u")
tok_cl = tiktoken.get_encoding("cl100k_base")
ids200 = {json.loads(l)["query_id"] for l in open("results/mh_only_20260928/baselines/ids200.jsonl") if l.strip()}
frac, allok = {}, {}
for q, r in recs.items():
    text, cells = R.render_doc(q, tables, hdr)
    chunks = R.split_text_spans(text, tok_cl, R.CHUNK_TOKENS)
    ok = set().union(*(set(R.leaf_cells(sp, cells)) for _, sp in chunks))
    g = set(r["gold_ids"])
    frac[q] = len(g & ok) / len(g); allok[q] = g <= ok
for tag, keep in (("2885", None), ("200", ids200)):
    qs = [q for q in frac if keep is None or q in keep]
    print(tag, len(qs), "cell_frac_mean", round(100 * statistics.mean(frac[q] for q in qs), 2), "all_ok", round(100 * statistics.mean(allok[q] for q in qs), 2))
