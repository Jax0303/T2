"""문서 원문(문단+표) 조건 예비 측정 — 검색만. PRESTATEMENT-fulldoc-feasibility.md 참조.
  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/loss_analysis_20261017/feasibility_fulldoc.py
"""
import json, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts/baselines")]
from mh_arms import build_tables, cell_id, load_population, NO_TITLE          # noqa: E402
from retrieval_accuracy import build_corpus                                   # noqa: E402
from rag_agent.retrieve.encoders import _tokenize, default_encoder            # noqa: E402
from rag_agent.retrieve.hybrid_index import _minmax                           # noqa: E402
from rag_agent.retrieve.sparse_bm25 import SparseBM25                         # noqa: E402
from rag_agent.serialization.chunks import markdown_source                    # noqa: E402
from langchain_core.documents import Document                                 # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter           # noqa: E402
from transformers import AutoTokenizer                                        # noqa: E402

OUT = Path(__file__).resolve().parent
A = {r["query_id"]: r for r in map(json.loads, open(ROOT / "results/cellform_20261012/A_chunk_md_records.jsonl", encoding="utf-8"))}
C = {r["query_id"]: r for r in map(json.loads, open(ROOT / "results/cellform_20261012/C_ours_budget_records.jsonl", encoding="utf-8"))}
ids = sorted(A); want = set(ids)
t0 = time.time()
queries, docs, _ = load_population("train")
Q = {q["uid"]: q for q in queries if q["uid"] in want}
docs = {u: docs[u] for u in ids}
tables, hdr = build_tables(docs, "v3.3u", "none")
texts, covers, _, unit_tids, _ = build_corpus("", sorted(tables), "s3c", "cell", {}, load=lambda t, _d: tables.get(t))
unit_uid = [t.split("::")[0] for t in unit_tids]
unit_cell = [cell_id(next(iter(cov)), hdr).split("::", 1)[1] for cov in covers]
print(f"[corpus] {len(tables)} tables, {len(texts)} cell units ({time.time()-t0:.0f}s)", flush=True)
# 문단 문장 단위
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'$\d])")
sent_texts, sent_uid = [], []
for u in ids:
    paras = docs[u][2] if len(docs[u]) > 2 else []
    for p in paras:
        for s in SENT.split(p.strip()):
            s = s.strip()
            if len(_tokenize(s)) >= 3:
                sent_texts.append(s); sent_uid.append(u)
print(f"[text] {len(sent_texts)} sentence units from {sum(len(docs[u][2]) for u in ids)} paragraphs", flush=True)
all_texts = texts + sent_texts
all_uid = np.array(unit_uid + sent_uid)
is_cell = np.array([True] * len(texts) + [False] * len(sent_texts))
enc = default_encoder(model_name="BAAI/bge-base-en-v1.5")
t1 = time.time(); emb = enc.encode(all_texts).astype(np.float32); print(f"[dense] {emb.shape} in {time.time()-t1:.0f}s", flush=True)
qv = {u: enc.encode_query([Q[u]["question"]])[0].astype(np.float32) for u in ids}
bm_all = SparseBM25(_tokenize(t) for t in all_texts)
bm_cell = SparseBM25(_tokenize(t) for t in texts)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
n_tok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
tok_len = np.array([n_tok(t) for t in all_texts])
print(f"[bm25] all {len(bm_all.vocab)} / cell {len(bm_cell.vocab)} terms ({time.time()-t0:.0f}s)", flush=True)

def ours(u, budget_tok, use_text):
    sel = np.flatnonzero((all_uid == u) & (is_cell | use_text))
    dn = emb[sel] @ qv[u]
    bm = bm_all if use_text else bm_cell
    sp = bm.get_scores(_tokenize(Q[u]["question"]))
    sp = sp[sel] if use_text else sp[sel]          # bm_cell 색인은 cell 단위가 앞에 있어 같은 인덱스
    sc = 0.7 * _minmax(dn) + 0.3 * _minmax(sp)
    order = sel[np.argsort(-sc, kind="stable")]
    got, used, n_units, n_text = set(), 0, 0, 0
    for i in order:
        if used + tok_len[i] > budget_tok: break
        used += tok_len[i]; n_units += 1
        if is_cell[i]: got.add(unit_cell[i])
        else: n_text += 1
    return got, n_units, n_text, used

# LangChain: 표만 / 문단+표
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
def doc_text(uid):
    parts, spans, off, head = [], [], 0, "Table name: \n"
    for tid in sorted((t for t in tables if t.split("::")[0] == uid), key=lambda t: int(t.split("::")[1])):
        tab = tables[tid]
        text, sp = markdown_source(tab, tab.table, NO_TITLE)
        assert text.startswith(head); text = text[len(head):]
        spans += [(off + s - len(head), off + e - len(head), cell_id((tid, i, j), hdr).split("::", 1)[1]) for s, e, (i, j) in sp]
        parts.append(text); off += len(text) + 2
    return "\n\n".join(parts), spans
def lc(u, with_text, ks=(1, 2, 4)):
    ttext, spans = doc_text(u)
    if with_text:
        pre = "\n\n".join(p.strip() for p in docs[u][2]) + "\n\n"
        text = pre + ttext; spans = [(s + len(pre), e + len(pre), c) for s, e, c in spans]
    else:
        text = ttext
    splits = splitter.split_documents([Document(page_content=text)])
    cells = []
    for d in splits:
        b, e = d.metadata["start_index"], d.metadata["start_index"] + len(d.page_content)
        cells.append(sorted(c for s, t, c in spans if b <= s and t <= e))
    ce = enc.encode([d.page_content for d in splits]).astype(np.float32)
    sims = ce @ qv[u] / (np.linalg.norm(ce, axis=1) * np.linalg.norm(qv[u]) + 1e-9)
    order = np.argsort(-sims, kind="stable")
    res = {}
    for k in ks:
        top = order[:k]
        got = set().union(*(set(cells[i]) for i in top)) if len(top) else set()
        res[k] = {"got": got, "tok": n_tok("\n\n".join(splits[i].page_content for i in top)),
                  "table_chunks": int(sum(bool(cells[i]) for i in top)), "n_chunks": len(splits)}
    return res

rows = []
for n, u in enumerate(ids, 1):
    gold = set(A[u]["gold_ids"])
    r = {"query_id": u, "layer": A[u]["layer"], "kind": A[u]["kind"], "m": A[u]["m"]}
    lt = lc(u, False); lf = lc(u, True, ks=(1, 2, 4, 8))
    for k in (1, 2, 4):
        r[f"lc_tab_k{k}"] = int(gold <= lt[k]["got"]); r[f"lc_tab_k{k}_tok"] = lt[k]["tok"]
        g, nu, ntx, used = ours(u, lt[k]["tok"], False)
        r[f"ours_tab_at_k{k}"] = int(gold <= g); r[f"ours_tab_at_k{k}_units"] = nu
    r["lc_tab_nchunks"] = lt[4]["n_chunks"]
    r["lc_full_k4"] = int(gold <= lf[4]["got"]); r["lc_full_k4_tok"] = lf[4]["tok"]; r["lc_full_nchunks"] = lf[4]["n_chunks"]; r["lc_full_k4_tablechunks"] = lf[4]["table_chunks"]
    r["lc_full_k8"] = int(gold <= lf[8]["got"])
    g, nu, ntx, used = ours(u, lf[4]["tok"], True)
    r["ours_full_k4"] = int(gold <= g); r["ours_full_units"] = nu; r["ours_full_textunits"] = ntx
    g2, nu2, _, _ = ours(u, lf[4]["tok"], False)
    r["ours_cellonly_at_fullbudget"] = int(gold <= g2)
    r["A_correct"] = A[u]["correct"]; r["C_correct"] = C[u]["correct"]; r["A_tok"] = A[u]["n_tok_ctx"]
    rows.append(r)
    if n % 100 == 0: print(f"  {n}/882 {time.time()-t0:.0f}s", flush=True)
with open(OUT / "feasibility_fulldoc_rows.jsonl", "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
def pct(key, qs=rows): return f"{sum(r[key] for r in qs)}/{len(qs)} ({100*sum(r[key] for r in qs)/len(qs):.1f}%)"
print("\n== 정답 셀 전부 포함(검색만), 882 ==")
print(f"표만 문서: 기존 레코드 LangChain {pct('A_correct')} 본 방법 {pct('C_correct')} | 재구현 LangChain k4 {pct('lc_tab_k4')} 본 방법(같은 토큰) {pct('ours_tab_at_k4')}")
for k in (1, 2):
    print(f"표만 문서 사다리 k={k}: LangChain {pct(f'lc_tab_k{k}')} (평균 {np.mean([r[f'lc_tab_k{k}_tok'] for r in rows]):.0f}토큰) 본 방법 같은 토큰 {pct(f'ours_tab_at_k{k}')}")
print(f"문서 원문(문단+표): 청크 수 평균 {np.mean([r['lc_full_nchunks'] for r in rows]):.1f}, LangChain k4 {pct('lc_full_k4')} (평균 {np.mean([r['lc_full_k4_tok'] for r in rows]):.0f}토큰, 상위4 중 표 청크 평균 {np.mean([r['lc_full_k4_tablechunks'] for r in rows]):.2f}), k8 {pct('lc_full_k8')}")
print(f"문서 원문 본 방법(셀+문장 색인, 같은 토큰) {pct('ours_full_k4')} (단위 평균 {np.mean([r['ours_full_units'] for r in rows]):.1f}, 그중 문장 {np.mean([r['ours_full_textunits'] for r in rows]):.1f}) | 셀만 색인(불공정 참고) {pct('ours_cellonly_at_fullbudget')}")
for L in ("lookup_m1", "lookup_m2+", "arith_m1", "arith_m2+"):
    qs = [r for r in rows if r["layer"] == L]
    print(f"  {L}: LangChain 원문 {pct('lc_full_k4', qs)} 본 방법 원문 {pct('ours_full_k4', qs)} | 표만 k1 LangChain {pct('lc_tab_k1', qs)} 본 방법 {pct('ours_tab_at_k1', qs)}")
json.dump({"n": 882, "summary_printed": True, "seconds": round(time.time()-t0)}, open(OUT / "feasibility_fulldoc_summary.json", "w"))
