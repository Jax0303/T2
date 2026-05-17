# HART Table Retrieval + Sidecar Verifier Agent

Two related research efforts on table retrieval over HiTab:

1. **HART (Header-Aware Retrieval over Tables)** — main project. Serializer × embedder × header-alignment ablation.
2. **Sidecar Verifier Agent** — query-aware structural verifier that reranks/judges vector RAG hits using the original parsed 2-D table. See [`sidecar_verifier/README.md`](sidecar_verifier/README.md).

![Architecture](docs/sidecar_architecture.png)

## TL;DR results

| Project | Key result |
|---|---|
| HART | **Negative**: HART scorer (header-alignment α-blend) does *not* beat a plain markdown serialization on HiTab dev. `plain_markdown` wins R@1 / nDCG / MRR; HART (header_path) only wins R@5 / R@10. Increasing α monotonically *decreases* metrics. |
| Sidecar verifier | **Positive on HiTab/FeTaQA, negative on TabFact**: Query-aware verifier rerank lifts R@1 by **+12.3 pp on HiTab (plain_markdown)** and **+1.5 pp on FeTaQA (TARGET)**, but **-1.2 pp on TabFact** because uniformly-structured Wikipedia tables don't expose discriminative header keywords. |

Both findings are honest research conclusions and motivate the limitations / next-steps sections.

## Repository layout

```
hart-table-retrieval/
├── README.md                  this file
├── configs/
│   └── experiment.yaml        embedders, serializers, alpha grid, paths
├── src/
│   ├── data/                  HiTab loader, HeaderTree
│   ├── serializers/           plain_markdown / json_kv / header_path
│   ├── retrieval/             embedder, indexer, searcher, hart_scorer
│   ├── evaluation/            recall@k, ndcg, mrr, hit
│   └── utils/                 config
├── scripts/
│   ├── run_indexing.py        ChromaDB index for all serializer×embedder combos
│   ├── run_retrieval.py       dense retrieval with HART α-blend
│   ├── run_evaluation.py      → results/evaluation_summary.csv
│   ├── run_ablation.py        HART-full / no-align / no-depth / single-vec
│   ├── token_length_control.py  token confound analysis
│   └── sidecar_architecture.py  diagram generator (docs/sidecar_architecture.png)
├── sidecar_verifier/          query-aware verifier agent (see its README)
└── target_bench/              cloned TARGET benchmark, used by sidecar_verifier/eval/target_run.py
```

`data/` and `target_bench/` are git-ignored — re-create them on a new machine via:

```bash
# data setup
mkdir -p data && cd data
git clone --depth=1 https://github.com/microsoft/HiTab.git hitab
python3 -c "import zipfile; zipfile.ZipFile('hitab/data/tables.zip').extractall('hitab/data')"

# TARGET benchmark (for sidecar_verifier/eval/target_run.py)
cd ..
git clone --depth=1 https://github.com/target-benchmark/target.git target_bench
# Slim target_bench/target_benchmark/retrievers/__init__.py to only import the Abs* classes
# so we don't need pexpect / hnswlib / OTTQA DrQA dependencies.
```

## Environment (home PC reproducer)

- WSL2 Ubuntu, Python 3.12
- RTX 3060 Ti (8 GB) — GPU is required for the Qwen-3B answerer; everything else runs on CPU
- 8 GB RAM allocated to WSL. For stability set
  `C:\Users\<you>\.wslconfig` →
  ```ini
  [wsl2]
  memory=10GB
  swap=4GB
  vmIdleTimeout=-1
  ```
  then `wsl --shutdown` once. Without this, long-running tasks can be killed when the host reclaims VRAM/RAM.

```bash
python3 -m venv --without-pip .venv
curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python3

.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv/bin/pip install \
    chromadb sentence-transformers numpy pandas pyyaml tiktoken matplotlib scipy openai \
    accelerate bitsandbytes tabulate \
    datasets pydantic python-dotenv func-timeout qdrant-client evaluate rouge-score sacrebleu \
    langchain langchain-community langchain-core langchain-openai langchain-text-splitters
```

## Running

```bash
# HART pipeline
python scripts/run_indexing.py --config configs/experiment.yaml --data-dir data/hitab --chroma-dir data/chroma_db
python scripts/run_retrieval.py --config configs/experiment.yaml --data-dir data/hitab --chroma-dir data/chroma_db
python scripts/run_evaluation.py
python scripts/run_ablation.py --data-dir data/hitab --chroma-dir data/chroma_db
python scripts/token_length_control.py

# Sidecar verifier
python sidecar_verifier/demo.py --n-queries 5 --llm
python sidecar_verifier/eval/faithfulness.py --max-queries 300 --serializer plain_markdown
python sidecar_verifier/eval/answer_accuracy.py --max-queries 30 --also-gold

# TARGET benchmark (paper-grade comparison)
python sidecar_verifier/eval/target_run.py --dataset fetaqa --top-k 10
python sidecar_verifier/eval/target_run.py --dataset tabfact --top-k 10
# OTTQA pending — corpus is large, expect 30–60 min
```

## Output files

```
results/
├── evaluation_summary.csv             HART main eval
├── ablation_summary.csv               HART ablation
├── token_length_analysis.csv          token confound
├── token_length_scatter.png
├── verifier_eval.json                 sidecar retrieval eval (plain_markdown)
├── verifier_eval_json_kv.json
├── verifier_eval_header_path.json
├── answer_accuracy.json               sidecar end-to-end with LLM
└── target/
    ├── fetaqa_summary.json
    ├── tabfact_summary.json
    └── *.jsonl                         per-query retrieval hits
```
