"""HiTab 확인 표본(300 밖) 입력 생성 — build_hitab.py 와 같은 규칙(A = LangChain 4청크, C = 셀 문장 목록 T 까지, D = 채운 부분 표 T 까지)에
값 표기 보정 하나만 더함: 자료 행렬 값의 부동소수 표기 잡음(-0.20800000000000002)을 10 유효숫자로(C 문장·D 칸 모두; 소수 7자리 이상인 수만 건드림).
  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/hitab_newregime_20261019/build_hitab_pool.py <ids.json> <out-dir>
"""
import glob, json, re, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from rag_agent.bench.hitab_grid import load_table                   # noqa: E402
from rag_agent.serialization.chunks import markdown_source          # noqa: E402
from rag_agent.retrieve.encoders import default_encoder             # noqa: E402
from langchain_core.documents import Document                        # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402
from transformers import AutoTokenizer                               # noqa: E402

ids = json.load(open(sys.argv[1])); want = set(ids); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
rec_path = sorted(glob.glob(str(HERE / "retrieval" / "*b100*records.jsonl")))[0]
R = {r["query_id"]: r for r in map(json.loads, open(rec_path, encoding="utf-8")) if r["query_id"] in want}
assert len(R) == len(ids), (rec_path, len(R), len(ids))
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); n_tok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
enc = default_encoder(model_name="BAAI/bge-base-en-v1.5")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
FNOISE = re.compile(r"-?\d+\.\d{7,}")
clean = lambda s: FNOISE.sub(lambda m: format(float(m.group(0)), ".10g"), s)
esc = lambda v: clean(str(v)).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("\r", "")
out = {"A": [], "C": [], "D": []}; n_fixed = {"C": 0, "D": 0}
for q in ids:
    r = R[q]; tid = r["table_id"]; tab = load_table(tid, "data/hitab"); t = tab.table
    gold = {(int(c[1]), int(c[2])) for c in r["gold_cells"]}
    units = [(clean(u["text"]), (int(u["cells"][0][1]), int(u["cells"][0][2]))) for u in r["context_units"] if u.get("cells")]
    n_fixed["C"] += sum(FNOISE.search(u["text"]) is not None for u in r["context_units"] if u.get("cells"))
    text, spans = markdown_source(tab, t, tab.title)
    splits = splitter.split_documents([Document(page_content=text)])
    cells = []
    for d in splits:
        b, e = d.metadata["start_index"], d.metadata["start_index"] + len(d.page_content)
        cells.append({(i, j) for s, en, (i, j) in spans if b <= s and en <= e})
    ce = enc.encode([d.page_content for d in splits]).astype(np.float32); qv = enc.encode_query([r["question"]])[0].astype(np.float32)
    order = np.argsort(-(ce @ qv), kind="stable")[:4]
    a_ctx = "\n\n".join(splits[i].page_content for i in order); a_cells = set().union(*(cells[i] for i in order))
    T = n_tok(a_ctx)
    base = {"query_id": q, "table_id": tid, "kind": "lookup", "layer": "lookup_m1", "m": 1, "question": r["question"], "answer": r["answer"], "gold_ids": [f"{i}-{j}" for i, j in sorted(gold)], "budget_T": T}
    out["A"].append({**base, "context": a_ctx, "correct": int(gold <= a_cells), "cells_in_context": len(a_cells), "n_tok_ctx": T, "n_chunks_in_table": len(splits)})
    sel = []
    for txt, c in units:
        if n_tok("\n".join(s for s, _ in sel + [(txt, c)])) > T: break
        sel.append((txt, c))
    c_ctx = "\n".join(s for s, _ in sel); c_cells = {c for _, c in sel}
    out["C"].append({**base, "context": c_ctx, "correct": int(gold <= c_cells), "cells_in_context": len(c_cells), "n_tok_ctx": n_tok(c_ctx)})
    def render(sel_cells):
        rows = sorted({i for i, _ in sel_cells}); cols = sorted({j for _, j in sel_cells}); deliv = set()
        lines = [f"Table: {tab.title}", "| | " + " | ".join(esc(" > ".join(t.col_path(j))) for j in cols) + " |", "| --- | " + " | ".join("---" for _ in cols) + " |"]
        for i in rows:
            vals = []
            for j in cols:
                v = t.cell(i, j); v = "" if v is None else str(v)
                if v.strip(): deliv.add((i, j)); vals.append(esc(v))
                else: vals.append("")
            lines.append("| " + esc(" > ".join(t.row_path(i))) + " | " + " | ".join(vals) + " |")
        return "\n".join(lines), deliv
    dsel, best = [], ("", set(), 0)
    for txt, c in units:
        txt_, deliv = render(dsel + [c])
        if n_tok(txt_) > T: break
        dsel.append(c); best = (txt_, deliv, len(dsel))
    d_ctx, d_cells, nsel = best
    n_fixed["D"] += any(FNOISE.search(str(t.cell(i, j)) or "") for (i, j) in d_cells)
    out["D"].append({**base, "context": d_ctx, "correct": int(gold <= d_cells), "cells_in_context": len(d_cells), "n_tok_ctx": n_tok(d_ctx), "selected_units": nsel})
N = len(ids)
for arm, rs in out.items():
    with open(OUT / f"{arm}_records.jsonl", "x", encoding="utf-8") as f:
        for x in rs: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"{arm}: {N}건, 정답 셀 포함 {sum(x['correct'] for x in rs)}, 전달 셀 평균 {sum(x['cells_in_context'] for x in rs)/N:.1f}, 토큰 평균 {sum(x['n_tok_ctx'] for x in rs)/N:.0f}" + (f", 표당 청크 수 평균 {sum(x['n_chunks_in_table'] for x in rs)/N:.2f}, 4개 이하 {sum(x['n_chunks_in_table']<=4 for x in rs)}" if arm == "A" else ""))
print("부동소수 잡음 보정 건수(문장 수 / D 문항 수):", n_fixed)
