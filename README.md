# Table-RAG Structural Audit

Reproducible pipeline for two experiments in a master's thesis on
**table-structure preservation in Retrieval-Augmented Generation**:

| Thesis section | Experiment | Script |
|---|---|---|
| §3 Serialization damage diagnosis | 5 serializers × 4 structural metrics on HiTab | `experiments/exp01_serialization_audit.py` |
| §4 Embedder layer-wise probing | Linear/MLP probes across 12 transformer layers | `experiments/exp02_layer_probing.py` |
| §5 HART Table Retrieval | Table retrieval indexing, dense/sparse retrieval, and end-to-end evaluation | `hart-table-retrieval/` |
| §6 Adaptive Table-RAG Agent | Adaptive Table-RAG agent with structural verification, LLM reader, and symbolic execution for hard queries | `rag-agent/` |

## Requirements

- Python 3.11
- CPU only (no GPU required)
- 32 GB RAM recommended
- [uv](https://docs.astral.sh/uv/) package manager

## Quickstart

```bash
# 1. Clone
git clone <repo-url> && cd table-rag-structural-audit

# 2. Create venv & install (CPU-only PyTorch is resolved automatically)
uv venv --python 3.11
uv sync --all-extras

# 3. Download HiTab data
#    Clone microsoft/HiTab and copy JSON files into data/raw/
git clone https://github.com/microsoft/HiTab.git /tmp/HiTab
cp -r /tmp/HiTab/data/* data/raw/

# 4. Run experiments
uv run python experiments/exp01_serialization_audit.py
uv run python experiments/exp02_layer_probing.py

# 5. Run Adaptive Table-RAG Agent Evaluation
python rag-agent/scripts/run_eval.py --llm local:Qwen/Qwen2.5-7B-Instruct --per-class 8 --out rag-agent/results/local_qwen7b.json

# 6. Run tests
uv run pytest
```

## Directory structure

```
table-rag-structural-audit/
├── pyproject.toml
├── data/
│   ├── raw/              # HiTab original JSON (downloaded)
│   └── processed/        # Preprocessed Table objects
├── src/
│   ├── io/               # HiTab loader & Table schema
│   ├── serializers/      # 5 serializers (HTML, Markdown, CSV, JSON-tree, OTSL)
│   ├── metrics/          # TEDS, Header-Path Acc, Cell-Coord Preserve, Merged-Cell Recovery
│   ├── probing/          # Layer-wise hidden extraction, probe classifiers, probe tasks
│   └── utils/            # Seed, logging
├── experiments/          # Experiment entry points + Hydra configs
├── tests/                # Unit tests
├── notebooks/            # Exploratory notebooks
├── results/              # Output tables & figures (git-ignored)
├── hart-table-retrieval/ # End-to-end retrieval and evaluation pipeline
└── rag-agent/            # Adaptive multi-stage Table-RAG agent with symbolic execution
```

## Key datasets

- **HiTab** ([microsoft/HiTab](https://github.com/microsoft/HiTab)):
  Hierarchical table dataset with complex headers and merged cells.
  No new data is constructed; only ground-truth annotations are used.

## Reproducing

All experiments use Hydra for configuration.
Default configs are in `experiments/configs/`.
Every run is seeded (default: 42) for deterministic results.
A smoke-test flag (`smoke_test=true`) limits each experiment to finish
within 30 minutes on a 8-core CPU laptop.

## §6 Adaptive Table-RAG Agent (`rag-agent/`)

An adaptive multi-stage Table-RAG Agent designed for hard queries requiring complex logical and arithmetic operations (derived from the HiTab dataset appendix). The agent routes incoming queries through an intelligent planning flow, leveraging both the serialized vector index and the original 2D structural table store to maximize retrieval fidelity and reasoning accuracy.

### Architecture & Pipeline Flow
1. **Query Classification**: Intent routing based on semantic and rule-based signals (`REASONING_ONLY`, `SIMPLE_LOOKUP`, `ARITHMETIC_AGG`, etc.).
2. **Stage Planning**: Choosing the optimal pipeline sequence (e.g., retrieving, verifying, symbolic execution, or direct reading).
3. **Retrieval**: Fetches top-$K$ table candidates from a vector index (Chroma + `bge-large-en-v1.5`).
4. **Verification**: Reranks tables using query-aware keyword and numerical overlaps against the **original 2D structure**, recovering from serialization flaws.
5. **Extraction & Execution**:
   - **Symbolic Path**: For arithmetic/formulaic queries, an LLM cell-extractor identifies cell coordinates and expressions, followed by a secure python AST evaluation.
   - **LLM Reader Path**: A high-capacity reader processes verified table blocks to generate answers.

### Empirical Evaluation (40 Stratified Hard Queries)
The agent was benchmarked on 40 highly complex queries across five stratified classes (`multi_op_formula`, `arithmetic_agg`, `pair_or_topk_arg`, `single_arg`, `comparison_or_count`), comparing **Llama-3.1-8B-Instant (Groq)** and **Qwen-2.5-7B-Instruct (Local)**.

- **Retrieval Reranking Success**: Over the baseline Vector Retrieval, the **Structural Verifier Reranker** boosted **Recall@1** from **57.5% to 67.5%** (+10.0 percentage points) across both setups, validating the benefit of original 2D context checking.
- **End-to-End Reasoning Performance**:
  - **Llama-3.1-8B-Instant (Groq)**: Exact Match (EM) = **0.0%**, Numeric Match (NM) = **15.0%**
  - **Qwen-2.5-7B-Instruct (Local)**: Exact Match (EM) = **30.0%**, Numeric Match (NM) = **45.0%**

This confirms that the local Qwen-2.5-7B-Instruct model (even in 4-bit quantization) is significantly more capable of handling complex structured reasoning queries than Llama-3.1-8B.

## License

[MIT](https://spdx.org/licenses/MIT.html)
