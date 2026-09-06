# rag-agent — RAG over hierarchical-header tables

Start at [`CLAUDE.md`](CLAUDE.md) (authoritative: problem, priorities, rejected
hypotheses, citation rules), then [`RESULTS.md`](RESULTS.md) (every current number with
its source file) and the latest `HANDOFF-<date>.md`. Pre-2026-08-31 numbers are in
[`RESULTS_ARCHIVE-2026-08-30.md`](RESULTS_ARCHIVE-2026-08-30.md).

**Rule:** every number carries the results file it came from.

## Pipeline

```
table  -> cell sentence  "In the table '<title>', among <row path>, the value of <col path> is <v>."   (S3c)
       -> embed (models/bge-base-cell-ft-p0, fine-tuned once offline) -> hybrid BM25+dense index, corpus-wide
query  -> rank every cell -> top-k sentences -> local Qwen2.5-7B-Instruct (4-bit) -> EM
```

There is **no separate table-retrieval stage**: the table rank is where the table's first
cell appears in the cell ranking. "Cell hit given the table" is read from the rank of the
gold cells *inside their own table* (`ranks_in_table`).

## Layout

```
rag_agent/
  serialization/   caption.py (cell sentence, title modes), templates.py, base.py (Chunk)
  retrieve/        hybrid_index.py (BM25 + FAISS exact IP), encoders.py (bge, query prefix)
  bench/           hitab.py (loader, gold cells), population.py (frozen query sets), schema.py
  reconstruct/     header_grid.py (row/col header paths from the flat grid)
  stores/          original_store.py (2-D table + header tree)
  llm/             local_qwen.py (reader), groq_llm.py / openai_llm.py (remote legs), factory.py
  eval/metrics.py  hitab_exact_match and friends
  data/loader.py   HiTab file access
scripts/
  corpus_dump_vs_cell.py    corpus builders (hitab_corpus / aitqa / realhitbench / multihiertt), cached encoder
  freeze_populations.py     derive + freeze every query population into populations/*.txt
  finetune_cell_encoder.py  contrastive fine-tuning of the cell encoder (a0 / p0 recipes)
  manual_sentence_ceiling.py, point3_reconstruction_cost.py, baseline_comparison_llm.py   shared helpers
  mt2net_retriever_baseline.py, operand_collision_within_doc.py, table_ranker_sweep.py, gold_label_audit.py
analysis/                   one instrument per file, each writing under results/<leg>/
  cell_rank_dump.py         corpus-wide rank of every gold cell  -> *_ranks.jsonl  (stage 1)
  stage1_board.py           query type x (table@k, cell|table@k, all-covered@k)  -> results/stage1/BOARD.md
  retrieved_answer_em.py    inject gold / top-k / oracle-rerank / CE-rerank cells, local reader, EM (stage 3)
  orc_ladder.py, rerank_verdict.py, p0_verdict.py, ...   verdict scripts named in results/*/VERDICT.md
  phase4_*.py, phase5_*.py, lookup_*.py, unaligned_*.py  the 2026-08-31 reader / audit legs (RESULTS.md §1-6)
populations/                frozen query id lists (n fixed; see freeze_populations.py)
results/                    committed measurements; every VERDICT.md names its instrument
models/                     fine-tuned encoders (git-ignored; recipes in PREREG-*.md)
tests/                      pytest, no network, fake embedders
```

## Running

```bash
.venv/bin/python -m pytest tests/ -q

# stage 1 -- rank every gold cell corpus-wide (no reader), current arm
PYTHONPATH=. .venv/bin/python analysis/cell_rank_dump.py --dataset hitab --split dev \
    --population hitab_dev_lookup_all --embed-model models/bge-base-cell-ft-p0 \
    --alpha 0.8 --title-mode page --out-dir results/stage1
PYTHONPATH=. .venv/bin/python analysis/stage1_board.py results/stage1/*_ranks.jsonl

# stage 3 -- inject retrieved cells, local reader, EM
PYTHONPATH=. .venv/bin/python analysis/retrieved_answer_em.py --split dev \
    --population hitab_dev_lookup_all --embed-model models/bge-base-cell-ft-p0 \
    --alpha 0.8 --title-mode page --ks 1 3 10 --oracle-ks 10 --out results/stage3/x.jsonl
```

`p0` must always run with `--title-mode page` (train/inference mismatch otherwise).
Reader legs append one record per (query, condition) and resume; re-runs go to a new `--out`.

## History

The 2026-08 line (Operand-Set Completeness, operand-targeted retrieval, structural
injection, completeness gate, header-path resolvers, FinQA/WikiSQL/IM-TQA loaders) was
closed on 2026-08-30 and removed from the tree on 2026-09-05. Its code is in history before
commit `b2fddb8`; its result files at `753fa2e`; its design document is
`RESEARCH_STRUCTURE.md` (superseded where it conflicts with `CLAUDE.md`).
