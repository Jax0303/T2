> **2026-09-11 평가 무결성 정정:** 새 실행과 비교군 해석은 [EVALUATION-INTEGRITY-2026-09-11.md](EVALUATION-INTEGRITY-2026-09-11.md)를 먼저 읽는다. 아래 과거 수치를 수정 코드의 결과로 인용하지 않는다. Gold 문맥은 수학적 상한이 아니며, 기존 청킹 행은 전체 TableRAG 시스템 재현이 아니다.

# rag-agent — RAG over hierarchical-header tables

Start at [`CLAUDE.md`](CLAUDE.md) (authoritative: problem, priorities, rejected
hypotheses, citation rules), then [`RESULTS.md`](RESULTS.md) (every current number with
its source file) and the latest status docs listed in `CLAUDE.md`. Pre-2026-08-31 numbers are in
`RESULTS_ARCHIVE-2026-08-30.md`, deleted 2026-09-08 and recoverable with
`git show 6e8302a:rag-agent/RESULTS_ARCHIVE-2026-08-30.md`.

**Rule:** every number carries the results file it came from.

## Pipeline

```
table  -> cell sentence  "In the table '<title>', among <row path>, the value of <col path> is <v>."   (S3c)
       -> embed (BAAI/bge-base-en-v1.5, off the shelf -- NO TRAINING) -> hybrid BM25+dense, corpus-wide
query  -> rank every cell -> top-20 cell sentences -> local Qwen2.5-7B-Instruct (4-bit) -> EM
```

There is **no separate table-retrieval stage** and **no training stage** (removed
2026-09-08). Experiments run on the HiTab **test** split only; the dev split and the
fine-tuned encoders are no longer part of this work.

**The metric is per-query retrieval accuracy** — for each question, does the retrieved
context contain the cells the dataset says the answer is read from, yes or no. No
`recall@k`, no `all-covered@k`, no k-ladder: one operating point, the 20 cells the reader
actually receives. See `CLAUDE.md` §0.

Current tables: [`results/retrieval_accuracy/TABLES.md`](results/retrieval_accuracy/TABLES.md)
(built by `analysis/accuracy_tables.py`; instruments `scripts/retrieval_accuracy.py` and
`scripts/answer_accuracy.py`).

## Layout

Everything below is what the current pipeline uses. 88 files belonging to the retired
metric line were removed on 2026-09-08; they are in git history, not here.

```
rag_agent/
  bench/           hitab_grid.py  raw grid <-> data matrix, gold cells from answer_formulas
                   hitab.py       loader on top of it, schema.py
  retrieve/        encoders.py (bge + query prefix), hybrid_index.py,
                   sparse_bm25.py (CSR BM25 -- the corpus-wide index needs it)
  serialization/   caption.py (the cell sentence), templates.py, base.py
  reconstruct/     header_grid.py, treethinker.py (header paths from a flat grid)
  stores/          original_store.py (parsed table + header trees)
  llm/             local_qwen.py (reader), factory.py, groq_llm.py, openai_llm.py
  eval/metrics.py  hitab_exact_match and friends
  data/loader.py   HiTab file access
scripts/
  retrieval_accuracy.py       TABLE 1 -- per-query retrieval accuracy at a fixed cell budget
  answer_accuracy.py          TABLE 2 -- reader EM on exactly the context retrieval delivered
  point3_reconstruction_cost.py, tree_reconstruct_hitab_raw.py
                              the retired table-alignment gate, kept ONLY so
                              analysis/pipeline_audit.py can measure what it used to drop
analysis/
  pipeline_audit.py           what the pipeline indexed and scored, before vs now
  accuracy_tables.py          builds results/retrieval_accuracy/TABLES.md from the JSONs
  unit_defect.py              the two dataset label defects, decided without looking at predictions
  ceiling_cases_full.py       every reader-ceiling error, classified with its recovery route
  failure_anatomy.py          retrieval failures + answer failures, side by side
  reader_ceiling_anatomy.py   splits scorer-caused from reader-caused failures
tests/                        14 files; test_hitab_grid.py checks the gold resolution
                              against the shipped test split, not a fixture
```

## Running

```bash
.venv/bin/python -m pytest tests/ -q

# Table 1 -- retrieval accuracy, per query, at the 20-cell operating point
PYTHONPATH=. .venv/bin/python scripts/retrieval_accuracy.py --split test \
    --corpus split --unit cell --template s3c --alpha 0.7 --budget 20 \
    --dump-context 20 --tag t_s3c_hybrid
# --corpus all searches every table HiTab ships (3,597 tables / 468k cells)
# --template mt2net | s2 | flat are the published-unit and ablation arms
# --unit row | table are the industry-practice and table-unit baselines

# Table 2 -- the reader, on exactly the cells retrieval delivered
PYTHONPATH=. .venv/bin/python scripts/answer_accuracy.py \
    --records results/retrieval_accuracy/t_s3c_hybrid_records.jsonl \
    --condition retrieved            # or: gold (the reader ceiling)

# what the pipeline indexed and scored, before vs now
PYTHONPATH=.:scripts .venv/bin/python analysis/pipeline_audit.py --split test
# every reader-ceiling error, classified with its recovery route
PYTHONPATH=. .venv/bin/python analysis/ceiling_cases_full.py --exclude-defect
# rebuild the report tables from the JSONs on disk
PYTHONPATH=. .venv/bin/python analysis/accuracy_tables.py \
    > results/retrieval_accuracy/TABLES.md
```

The encoder is off the shelf and there is no training step, so a run needs no checkpoint.

## History

The 2026-08 line (Operand-Set Completeness, operand-targeted retrieval, structural
injection, completeness gate, header-path resolvers, FinQA/WikiSQL/IM-TQA loaders) was
closed on 2026-08-30 and removed from the tree on 2026-09-05. The metric line that replaced
it (`all-covered@k`, the fine-tuned encoders, the dev split) was closed on 2026-09-08 and its
88 remaining files removed in the same commit. Both are in git history; nothing in the tree
belongs to either. A new session begins at `CLAUDE.md` §0.
