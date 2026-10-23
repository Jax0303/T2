"""문서 원문 조건에서 LangChain 상위 4개 중 본문 청크가 정답을 글자 그대로 담는 빈도 — 검색만(예비)."""
import json, re, sys, time
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from mh_arms import build_tables, cell_id, load_population, NO_TITLE
from rag_agent.retrieve.encoders import default_encoder
from rag_agent.serialization.chunks import markdown_source
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
OUT = Path(__file__).resolve().parent
A = {r["query_id"]: r for r in map(json.loads, open(ROOT / "results/cellform_20261012/A_chunk_md_records.jsonl", encoding="utf-8"))}
ids = sorted(A); want = set(ids)
queries, docs, _ = load_population("train"); Q = {q["uid"]: q for q in queries if q["uid"] in want}; docs = {u: docs[u] for u in ids}
tables, hdr = build_tables(docs, "v3.3u", "none")
enc = default_encoder(model_name="BAAI/bge-base-en-v1.5")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
norm = lambda s: re.sub(r"[,\s\$%]", "", str(s)).lower()
def doc_text(uid):
    parts, spans, off, head = [], [], 0, "Table name: \n"
    for tid in sorted((t for t in tables if t.split("::")[0] == uid), key=lambda t: int(t.split("::")[1])):
        tab = tables[tid]; text, sp = markdown_source(tab, tab.table, NO_TITLE); text = text[len(head):]
        spans += [(off + s - len(head), off + e - len(head), cell_id((tid, i, j), hdr).split("::", 1)[1]) for s, e, (i, j) in sp]
        parts.append(text); off += len(text) + 2
    return "\n\n".join(parts), spans
rows = []; t0 = time.time()
for n, u in enumerate(ids, 1):
    ttext, spans = doc_text(u); pre = "\n\n".join(p.strip() for p in docs[u][2]) + "\n\n"
    text = pre + ttext; spans = [(s + len(pre), e + len(pre), c) for s, e, c in spans]
    splits = splitter.split_documents([Document(page_content=text)])
    cells = [sorted(c for s, t, c in spans if d.metadata["start_index"] <= s and t <= d.metadata["start_index"] + len(d.page_content)) for d in splits]
    ce = enc.encode([d.page_content for d in splits]).astype(np.float32); qv = enc.encode_query([Q[u]["question"]])[0].astype(np.float32)
    order = np.argsort(-(ce @ qv), kind="stable")[:4]
    g = Q[u]["answer"]; gs = norm(g) if isinstance(g, str) else norm(f"{g:g}")
    desc = docs[u][1]; desc = json.loads(desc) if isinstance(desc, str) else desc
    gold_vals = [norm(desc[e].strip().rstrip(".").rsplit(" is ", 1)[-1]) for e in A[u]["gold_ids"] if e in desc]
    text_chunks = [splits[i].page_content for i in order if not cells[i]]
    tc = norm(" ".join(text_chunks))
    rows.append({"query_id": u, "layer": A[u]["layer"], "n_text_chunks_top4": len(text_chunks),
                 "gold_in_top4": int(set(A[u]["gold_ids"]) <= set().union(*(set(cells[i]) for i in order))),
                 "answer_str_in_text_chunks": int(bool(gs) and gs in tc), "all_gold_vals_in_text_chunks": int(bool(gold_vals) and all(v and v in tc for v in gold_vals)),
                 "rank_of_first_table_chunk": int(next((k for k, i in enumerate(order) if cells[i]), -1))})
    if n % 200 == 0: print(f"  {n}/882 {time.time()-t0:.0f}s", flush=True)
with open(OUT / "fulldoc_textchunk_rows.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
N = len(rows); miss = [r for r in rows if not r["gold_in_top4"]]
print(f"\n문서 원문 LangChain 상위4: 정답 셀 누락 {len(miss)}/{N}; 본문 청크 수 평균 {np.mean([r['n_text_chunks_top4'] for r in rows]):.2f}; 표 청크가 1위인 문항 {sum(r['rank_of_first_table_chunk']==0 for r in rows)}")
print(f"본문 청크에 정답 문자열 그대로: 전체 {sum(r['answer_str_in_text_chunks'] for r in rows)}/{N}, 정답 셀 누락 문항 중 {sum(r['answer_str_in_text_chunks'] for r in miss)}/{len(miss)}; 정답 셀 값 전부: 전체 {sum(r['all_gold_vals_in_text_chunks'] for r in rows)}, 누락 중 {sum(r['all_gold_vals_in_text_chunks'] for r in miss)}")
for L in ("lookup_m1", "lookup_m2+", "arith_m1", "arith_m2+"):
    qs = [r for r in miss if r["layer"] == L]
    print(f"  {L}: 누락 {len(qs)}, 그중 본문에 정답 문자열 {sum(r['answer_str_in_text_chunks'] for r in qs)}, 셀 값 전부 {sum(r['all_gold_vals_in_text_chunks'] for r in qs)}")
