#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Cell vs row vs whole-table index on the multi-cell lookups.

Every gold set here lives inside one table and most inside one row, so an
all-or-nothing setEM@k punishes the cell index for a property of the queries,
not of the retriever. This ranks the same corpus at three granularities and
scores the same way: a query counts at k when the top-k chunks together cover
EVERY gold cell. No reader, no token budget.

Output: results/lookup_gap/multicell_units.{json,md}."""
import json
import sys
import types
from collections import Counter
from pathlib import Path

sys.path.insert(0, "."); sys.path.insert(0, "scripts"); sys.path.insert(0, "analysis")

import numpy as np                                                   # noqa: E402
import corpus_dump_vs_cell as cdv                                    # noqa: E402
from header_path_coverage import load_corpus                         # noqa: E402
from rag_agent.retrieve.encoders import default_encoder              # noqa: E402
from rag_agent.retrieve.hybrid_index import HybridIndex, _minmax     # noqa: E402
from rag_agent.serialization.base import Chunk                       # noqa: E402
from rag_agent.serialization.caption import caption_sentence         # noqa: E402
from rag_agent.serialization.templates import STRUCTURAL_COMPACT     # noqa: E402

KS = (1, 2, 3, 5, 10, 20, 50)
# Cost axis: how many corpus cells the retrieval hands over. Tokens were the
# obvious ruler and are the wrong one -- a token budget is a property of
# whichever reader is bolted on, so a comparison normalised by it stops being a
# statement about the index. Cells delivered is a property of the retrieval.
CELL_BUDGETS = (10, 50, 100, 500)
ALPHA = 0.7
POP = "hitab_dev_multicell_lookup"

a = types.SimpleNamespace(dataset="hitab", data_dir="data/hitab", split="dev",
                          population=POP, cell_scheme="S3c", rhb_question_types=[],
                          rhb_em_only=False, mh_queries=400, seed=42, max_queries=0)
C = load_corpus(a)
enc = cdv._CachedEncoder(default_encoder(model_name="BAAI/bge-small-en-v1.5"),
                         ".cache/corpus_dump_vs_cell",
                         "hitab_dev_BAAI/bge-small-en-v1.5")

cell_txt = [caption_sentence(C.title.get(t, ""), *C.cell_paths[n],
                             template=STRUCTURAL_COMPACT)
            for n, (t, _i, _j) in enumerate(C.cell_owner)]
# what each chunk covers, as a set of (table, row, col)
# (chunks to rank, cells each chunk covers, TEXT THE READER WOULD RECEIVE).
# The third differs from the first only for tables: a table is ranked by its
# index text (title, caption, header labels) but read as the whole grid, so
# pricing it by the index text would compare a lookup key against a document.
units = {
    "cell": ([Chunk(table_id=t, chunk_id=f"c::{t}::{i}:{j}", text=x, scheme="S3c",
                    kind="cell") for x, (t, i, j) in zip(cell_txt, C.cell_owner)],
             [{(t, i, j)} for (t, i, j) in C.cell_owner],
             list(cell_txt)),
    "row": ([Chunk(table_id=t, chunk_id=f"r::{t}::{i}", text=x, scheme="row",
                   kind="row") for x, (t, i) in zip(C.row_text, C.row_owner)],
            [{(t, i, j) for (t2, i2, j) in C.cell_owner if (t2, i2) == (t, i)}
             for (t, i) in C.row_owner],
            list(C.row_text)),
    "table": ([Chunk(table_id=t, chunk_id=f"t::{t}", text=C.table_text[t],
                     scheme="table", kind="table") for t in C.tids],
              [{(t, i, j) for (t2, i, j) in C.cell_owner if t2 == t} for t in C.tids],
              [C.md_lines[t] if isinstance(C.md_lines[t], str) else "\n".join(C.md_lines[t]) for t in C.tids]),
}

out = {}
for name, (chunks, covers, cost_text) in units.items():
    ix = HybridIndex(chunks, encoder=enc, alpha=0.5)
    mean_cells = sum(len(c) for c in covers) / len(covers)
    # a chunk of another size is not the same purchase: the index that wins
    # per-k can still lose per cell handed over
    eq = {B: max(1, int(B // mean_cells)) for B in CELL_BUDGETS}
    hits = {k: [] for k in tuple(KS) + tuple(eq.values())}
    for q in C.queries:
        bm, dn = ix._bm25_scores(q["question"]), ix._dense_scores(q["question"])
        order = np.argsort(-(ALPHA * _minmax(dn) + (1 - ALPHA) * _minmax(bm)))
        gold = set(q["gold_cells"])
        got, kept = set(), 0
        for k in sorted(hits):
            while kept < k:
                got |= covers[order[kept]]
                kept += 1
            hits[k].append(int(gold <= got))
    out[name] = {"n_chunks": len(chunks),
                 "mean_cells_per_chunk": round(mean_cells, 2),
                 "equivalent_k": eq,
                 **{f"setEM@{k}": round(sum(v) / len(v), 4) for k, v in hits.items()}}
    print(name, out[name], flush=True)

pop = {q["query_id"]: q for q in
       json.load(open("results/lookup_gap/multicell_pop.json"))["queries"]}
md = ["# 다중 셀 조회 — 색인 단위별 setEM (리더 없음)", "",
      f"모집단 `populations/{POP}.txt` (n={len(C.queries)}). 계측기 `analysis/multicell_unit_bench.py`.",
      "hybrid α=0.7, `BAAI/bge-small-en-v1.5`. 상위 k개 청크가 **합쳐서** gold 셀 전부를",
      "덮으면 1. 토큰 예산 없음 — 예산은 리더의 성질이라 색인 비교의 축으로 쓰지 않는다.", "",
      "| 단위 | 청크 수 | " + " | ".join(f"setEM@{k}" for k in KS) + " |",
      "|---|---:|" + "---:|" * len(KS)]
for name in ("cell", "row", "table"):
    r = out[name]
    md.append(f"| {name} | {r['n_chunks']} | " +
              " | ".join(f"{r[f'setEM@{k}']:.4f}" for k in KS) + " |")
md += ["", "## 같은 양을 넘길 때 (셀 개수 기준)", "",
       "청크마다 담는 셀 수가 다르므로 같은 k는 같은 양이 아니다. 등가 k = floor(셀 예산 /",
       "청크당 평균 셀 수). **토큰이 아니라 셀 개수로 맞춘다** — 토큰 예산은 붙이는 리더의",
       "성질이라 그것으로 정규화하면 색인에 대한 진술이 아니게 된다.", "",
       "| 단위 | 청크당 평균 셀 | " + " | ".join(f"셀 {B}개" for B in CELL_BUDGETS) + " |",
       "|---|---:|" + "---:|" * len(CELL_BUDGETS)]
for name in ("cell", "row", "table"):
    r = out[name]
    md.append(f"| {name} | {r['mean_cells_per_chunk']} | " + " | ".join(
        f"{r[f'setEM@{r['equivalent_k'][B]}']:.4f} (k={r['equivalent_k'][B]})"
        for B in CELL_BUDGETS) + " |")

md += ["", "gold 셀의 위치: 94/94가 한 표 안, 68/94가 한 행 안, 20/94가 한 열 안 "
       "(행 수 중앙값 1, 열 수 중앙값 2).",
       "즉 셀 색인은 한 청크로 못 덮는 집합을 여러 청크로 맞춰야 하고, 행·표 색인은",
       "한 청크가 집합을 통째로 덮는다."]
d = Path("results/lookup_gap")
(d / "multicell_units.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
(d / "multicell_units.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
