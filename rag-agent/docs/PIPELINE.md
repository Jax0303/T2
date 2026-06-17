# Operand-targeted Table-RAG pipeline

The five components implementing the thesis pipeline, all under the
`rag_agent` package and driven by scripts in `scripts/`.

```
question + table
      │
  (1) serialization      rag_agent/serialization/   S1 flat / S2 header-path chunks
      │
  (2) HPIR decomposition rag_agent/query/            question -> header-path operands
      │
  (3) operand retrieval  rag_agent/retrieve/         hybrid BM25+dense over S2 cells,
      │                                              one query per operand -> union
      │                                              metric: operand_recall@k
  (4) coverage+fallback  rag_agent/fallback/         self-assessed coverage + HPIR
      │                                              confidence; whole-table fallback
      │                                              under a token budget when low
  (5) generation+eval    rag_agent/generation/       direct vs codegen answer,
      │                                              EM / numeric / exec accuracy
   answer + scores
```

## Setup

```bash
pip install rank_bm25            # required; numpy already present
python scripts/download_hitab.py # fetch the benchmark (see docs/DATA.md)
```

The dense encoder and LLM are optional/pluggable: without `torch` /
`sentence-transformers` the retriever uses a NumPy `HashingEncoder` fallback,
and `--llm mock` gives a deterministic baseline — so the whole pipeline runs and
tests pass on a fresh CPU container. On the GPU box pass `--encoder bge` and
`--llm groq:...` / `--llm local:Qwen/...` for the real numbers.

## Run each stage

```bash
# (3) operand_recall@{1,3,5,10}
python scripts/operand_recall_eval.py --split dev --max-samples 200

# (4) coverage / fallback distribution + no-fallback ablation
python scripts/fallback_ablation.py --split dev --max-samples 200

# (5) end-to-end answer accuracy (direct + codegen)
python scripts/answer_eval.py --llm mock --modes direct codegen --max-samples 100
```

## Ablation knobs

| Question | Flag |
|---|---|
| value of the fallback | `scripts/answer_eval.py --no-fallback` vs default |
| value of S2 over S1 | `serialize(..., scheme="S1")` vs `"S2"` in retrieval index |
| value of HPIR decomposition | `--alpha 0` (BM25-only) vs hybrid |
| codegen vs direct reader | `--modes direct` vs `--modes codegen` |
| retrieval ceiling (oracle) | gold operands via `gold_operands_from_hitab` |

## Key measurements (paper result tables)

1. `header_path_match` / decomposition confidence — `rag_agent.retrieve.decomposition_confidence`
2. `operand_recall@k` — `scripts/operand_recall_eval.py`
3. `coverage_rate` distribution + `fallback_rate` — `scripts/fallback_ablation.py`
4. `answer_accuracy` (EM / numeric / exec) — `scripts/answer_eval.py`

All scripts log per-sample JSON to `results/` and use `seed=42`.
