# Benchmark data — download & layout

The container is cloned fresh each session and does **not** ship the datasets,
so fetch them before running any experiment. The downloaders live in
`scripts/`; the data they write lands under `data/` (git-ignored).

## Quick start

Run from the `rag-agent/` root:

```bash
# Core target only (no extra deps): HiTab
python scripts/download_hitab.py

# All three benchmarks (FinQA / WikiSQL need `pip install datasets`)
python scripts/download_benchmarks.py --finqa --wikisql
```

Both scripts are idempotent — existing files are skipped unless `--force`.

## Benchmarks

| Benchmark | Role in the thesis | Source | Needs `datasets` |
|---|---|---|---|
| **HiTab** | core target — hierarchical tables | microsoft/HiTab (direct download) | no |
| **FinQA** | numeric reasoning (codegen vs direct) | `dreamerdeo/finqa` | yes |
| **WikiSQL** | flat-table control group | `Salesforce/wikisql` | yes |

## Resulting layout

`rag_agent.data.loader` expects HiTab at:

```
data/hitab/data/
├── train_samples.jsonl     # 7417 QA samples
├── dev_samples.jsonl       # 1671
├── test_samples.jsonl      # 1584
└── tables/
    ├── hmt/<table_id>.json  # 3597 hierarchical-matrix tables
    └── raw/<table_id>.json  # 3597 raw tables
```

FinQA / WikiSQL land under `data/finqa/<split>.jsonl` and
`data/wikisql/<split>.jsonl`.

## Loading

```python
from rag_agent.data.loader import load_hitab
from rag_agent.serialization import serialize, from_hitab_raw

samples = load_hitab(split="dev", max_samples=10)   # each has a `table` field
chunks = serialize(from_hitab_raw(samples[0]["table"]), scheme="S2")
```

See `rag_agent/serialization/` for the S1 (flat) / S2 (header-path) schemes.
