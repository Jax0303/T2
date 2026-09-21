> **2026-09-11 평가 무결성 정정:** 새 실행과 비교군 해석은 [EVALUATION-INTEGRITY-2026-09-11.md](EVALUATION-INTEGRITY-2026-09-11.md)를 먼저 읽는다. Gold 문맥은 수학적 상한이 아니며, 기존 청킹 행은 전체 TableRAG 시스템 재현이 아니다.

# rag-agent — RAG over hierarchical-header tables

Start at [`CLAUDE.md`](CLAUDE.md) §0 and the status documents it lists. Current results:

- HiTab, five representations under one retriever/reader: `results/five_representations_20260912/`
  (manifest `analysis/comparison_manifest.five.v2.json`, source runs `results/evaluation_v2/`)
- MultiHiertt: `REPORT-2026-09-13.md`, `VERIFICATION-2026-09-13.md`, `DATA-USE-2026-09-14.md`,
  header rules `PREREG-2026-09-14-header-v3.md`
- Strict Recall (budget-free set scoring): `results/STRICT_RECALL_TABLES-2026-09-15.md`

**Rule:** every number carries the results file it came from.

## Pipeline

```
table  -> cell sentence (unique label + row path + col path + value)   (S3c)
       -> embed (BAAI/bge-base-en-v1.5, off the shelf -- NO TRAINING) -> hybrid BM25+dense, alpha 0.7
query  -> rank cells -> selected cell sentences -> local reader (Qwen2.5-7B / Qwen3-8B, 4-bit) -> EM
```

## Layout

```
rag_agent/
  bench/           hitab_grid.py (raw grid <-> data matrix, gold cells), hitab.py, schema.py
  retrieve/        encoders.py, hybrid_index.py, sparse_bm25.py
  serialization/   caption.py, templates.py, chunks.py, tablerag_unit.py, base.py
  reconstruct/     header_grid.py (MultiHiertt header rules v2..v3.3), treethinker.py
  stores/          original_store.py
  llm/             local_qwen.py (reader), factory.py, groq_llm.py, openai_llm.py
  eval/            metrics.py, answer_em.py, multihiertt_em.py, artifacts.py, strict_recall.py
  data/loader.py
scripts/
  retrieval_accuracy.py       HiTab retrieval (fixed budget / --metric-mode strict_*)
  answer_accuracy.py          HiTab reader EM on the stored context
  mh_arms.py                  MultiHiertt retrieval arms (--header-rule, --metric-mode)
  retrieval_accuracy_mh.py    forwards to mh_arms
  answer_accuracy_mh.py       MultiHiertt reader EM (--resume)
  reader_cot_pilot.py         MultiHiertt reader pilot (validation)
  point3_reconstruction_cost.py, tree_reconstruct_hitab_raw.py   cell_text/hierarchy helpers pinned by tests
analysis/
  validated_tables.py, five_representation_audit.py, compose_oracle.py, unit_defect.py
  verify_upstream.py, upstream_parity.py, upstream_audit/
  mh_final_report.py, audit_2026_09_13.py, header_*.py
tests/
```

## Running

```bash
.venv/bin/python -m pytest tests/ -q
PYTHONPATH=. .venv/bin/python scripts/retrieval_accuracy.py --help
PYTHONPATH=.:scripts .venv/bin/python scripts/mh_arms.py --help
```

## History

Retired lines (Operand-Set Completeness, `all-covered@k`, fine-tuned encoders, dev split,
token-budget ladder, RealHiTBench/AIT-QA, and the rejected 2026-09-09..10 experiments) are
removed from the tree. The last cleanup (2026-09-15, 794 files including `RESULTS.md`) is
recoverable with `git show f4e6865:rag-agent/<path>`.
