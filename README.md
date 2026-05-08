# Table-RAG Structural Audit

Reproducible pipeline for two experiments in a master's thesis on
**table-structure preservation in Retrieval-Augmented Generation**:

| Thesis section | Experiment | Script |
|---|---|---|
| §3 Serialization damage diagnosis | 5 serializers × 4 structural metrics on HiTab | `experiments/exp01_serialization_audit.py` |
| §4 Embedder layer-wise probing | Linear/MLP probes across 12 transformer layers | `experiments/exp02_layer_probing.py` |

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

# 5. Run tests
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
└── results/              # Output tables & figures (git-ignored)
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

## License

[MIT](https://spdx.org/licenses/MIT.html)
