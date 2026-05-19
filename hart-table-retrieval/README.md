# HART Table Retrieval + Sidecar Verifier Agent

Two related research efforts on table retrieval over HiTab:

1. **HART (Header-Aware Retrieval over Tables)** — main project. Serializer × embedder × header-alignment ablation.
2. **Sidecar Verifier Agent** — query-aware structural verifier that reranks/judges vector RAG hits using the original parsed 2-D table. See [`sidecar_verifier/README.md`](sidecar_verifier/README.md).

![Architecture](docs/sidecar_architecture.png)

## TL;DR results

| Project | Key result |
|---|---|
| HART | **Negative**: HART scorer (header-alignment α-blend) does *not* beat a plain markdown serialization on HiTab dev — now confirmed on **both** bge-large-en-v1.5 and multilingual-e5-large-instruct (GPU runs). `plain_markdown` wins R@1 / nDCG / MRR; HART (header_path) only wins R@5 / R@10. Increasing α monotonically *decreases* metrics on header_path. |
| Sidecar verifier (retrieval) | **Positive on HiTab/FeTaQA, negative on TabFact**: Query-aware verifier rerank lifts R@1 by **+12.3 pp on HiTab (plain_markdown)** and **+1.5 pp on FeTaQA (TARGET)**, but **-1.2 pp on TabFact** because uniformly-structured Wikipedia tables don't expose discriminative header keywords. |
| **Hard-query end-to-end (2026-05-20)** | On a stratified 40-query hard subset from HiTab dev (paper-appendix difficulty categories), routing R@1 = **0.700** (`pair_or_topk_arg` perfect 1.000), but answer accuracy is **0.250** overall with Qwen2.5-7B-Instruct 4-bit. **Multi-cell arithmetic stays at 0.000** even with CoT prompting and the gold table (oracle) — reader bottleneck, not routing. |

All three findings are honest research conclusions. See [Hard-query eval](#hard-query-end-to-end-eval) for the bug post-mortem and next steps.

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
│   ├── run_hard_query_eval.py  stratified hard-query end-to-end eval (Qwen-7B 4bit + CoT)
│   └── sidecar_architecture.py  diagram generator (docs/sidecar_architecture.png)
├── docs/
│   └── routing_explanation.md  4-stage routing (vector → query-aware verify → reconcile → read)
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
- RTX 3060 Ti (8 GB) — GPU is now used for indexing/retrieval (bge-large, multilingual-e5) AND the Qwen-7B answerer. The `SentenceTransformerEmbedder` auto-detects CUDA; pass `--retriever-device cpu` when the reader needs the full VRAM.
- **For lab PC (more VRAM)**: same code, no flags needed — embedder defaults to CUDA, LLM defaults to CUDA, both share happily on ≥16 GB. Pass `--llm-model Qwen/Qwen2.5-14B-Instruct` (or `--quantization 8bit`) for the answerer if VRAM allows — see `scripts/run_hard_query_eval.py`.
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

# Sidecar verifier (retrieval only)
python sidecar_verifier/demo.py --n-queries 5 --llm
python sidecar_verifier/eval/faithfulness.py --max-queries 300 --serializer plain_markdown
python sidecar_verifier/eval/answer_accuracy.py --max-queries 30 --also-gold

# Hard-query end-to-end eval (Qwen-7B 4bit, CoT prompt, stratified across HiTab difficulty classes)
python scripts/run_hard_query_eval.py --per-class 8 --max-new-tokens 384 --also-gold
# On home PC (8 GB VRAM) keep retriever on CPU so the 7B fits:
python scripts/run_hard_query_eval.py --per-class 8 --max-new-tokens 384 --also-gold --retriever-device cpu
# On lab PC (≥16 GB VRAM) you can leave both on GPU and try 14B:
python scripts/run_hard_query_eval.py --per-class 8 --max-new-tokens 384 --also-gold \
    --retriever-device cuda --llm-model Qwen/Qwen2.5-14B-Instruct

# TARGET benchmark (paper-grade comparison)
python sidecar_verifier/eval/target_run.py --dataset fetaqa --top-k 10
python sidecar_verifier/eval/target_run.py --dataset tabfact --top-k 10
# OTTQA pending — corpus is large, expect 30–60 min
```

## Output files

```
results/
├── evaluation_summary.csv             HART main eval (bge-large + multilingual-e5, 2026-05-20)
├── ablation_summary.csv               HART ablation
├── token_length_analysis.csv          token confound
├── token_length_scatter.png
├── verifier_eval.json                 sidecar retrieval eval (plain_markdown)
├── verifier_eval_json_kv.json
├── verifier_eval_header_path.json
├── answer_accuracy.json               sidecar end-to-end with LLM (Qwen-3B, 30 queries)
├── hard_query_eval_cot.json           ★ stratified hard-query eval w/ CoT (Qwen-7B, 40 queries)
└── target/
    ├── fetaqa_summary.json
    ├── tabfact_summary.json
    └── *.jsonl                         per-query retrieval hits
```

## Hard-query end-to-end eval

A 40-query stratified subset of HiTab dev, partitioned by the HiTab paper's appendix difficulty categories (derived from each sample's `aggregation` array and Excel-style `answer_formulas`):

| Class | Population | What it tests |
|---|---|---|
| `multi_op_formula` | 37 | Excel formulas with ≥2 operators, e.g. `=(B+C+D)/E` |
| `arithmetic_agg` | 139 | sum / div / diff / average / range |
| `pair_or_topk_arg` | 153 | "X or Y?" — pair-argmax/argmin, top-k pick |
| `single_arg` | 93 | argmax / argmin / max / min |
| `comparison_or_count` | 54 | greater_than / less_than / opposite (sign flip) / counta |

### Results (2026-05-20, Qwen2.5-7B-Instruct 4bit, CoT prompt)

| Class | n | R@1 (vec) | R@1 (verifier rerank) | answer_acc | oracle_acc |
|---|---:|---:|---:|---:|---:|
| multi_op_formula | 8 | 0.625 | 0.500 | **0.000** | **0.000** |
| arithmetic_agg | 8 | 0.375 | 0.375 | **0.000** | **0.000** |
| pair_or_topk_arg | 8 | 0.500 | **1.000** | 0.500 | 0.500 |
| single_arg | 8 | 0.625 | 0.750 | 0.125 | 0.250 |
| comparison_or_count | 8 | 0.750 | 0.875 | **0.625** | 0.625 |
| **OVERALL** | 40 | 0.575 | **0.700** | **0.250** | 0.275 |

Full per-query trace: `results/hard_query_eval_cot.json` (40 rows: query, gold table, gold answer + formula, vector top, final routed top, predicted answer, oracle answer).

### Bugs found and fixed before these numbers were trustworthy

1. **`_format_table_for_llm` lost the multi-level top header** — only the leaf segment was used as the column label, so e.g. a HiTab table with `(black male workers, immigrant)` + `(other male workers, immigrant)` rendered as **eight columns all called "immigrant"**. The reader could not identify which column to read. Fix: join the full top-header path with ` :: ` (separator chosen to not collide with markdown pipes).
2. **Left-header was not in the markdown** — rows were rendered with index `0..N` instead of the row label (`spouse`, `dating partner`, …). Fix: prepend left-header leaf as a `row_header` column.
3. **`_numeric_match` could not match list-style gold answers** — `gold=['quebec']` vs `pred='quebec'` compared `"['quebec']"` to `"quebec"` and returned False. Most string-answer classes were therefore reported as 0% even when the LLM was right. Fix: case-insensitive substring match against each gold list element.
4. **Prompt told the reader "output just the number"**, so it never attempted multi-cell arithmetic. Fix: CoT prompt — `Reasoning: …` then `Final answer: …`, with a regex parser that pulls the final line.

Before these fixes, every class read as 0% answer accuracy — which **was a measurement artifact, not the reader's true ceiling**.

### What the numbers say after the fixes

- **Routing works**. Verifier rerank (w_verify=0.2) lifts overall R@1 from 0.575 → 0.700. `pair_or_topk_arg` becomes perfect.
- **Single-cell lookup classes work** at 25–62.5% answer accuracy with Qwen-7B + CoT.
- **Multi-cell arithmetic is the genuine ceiling.** Inspecting CoT traces in `results/hard_query_eval_cot.json`:
  - For `=E5+E7+E8+E9` (gold 30, "family homicides %"), Qwen sums the *wrong* set of rows and outputs `201.99`.
  - For `=INT(G7/G8)` (gold 2), Qwen computes `0.343` (the right two cells, **flipped denominator**) and skips the INT.
  - For `=G10/D10` (gold 7.32), Qwen does the right operation on adjacent — but *wrong* — rows.
  Qwen-7B 4bit can attempt arithmetic, but it picks cells unreliably and inverts division frequently.

### Next steps (TODO for lab PC)

1. **TableRAG-style cell pre-extraction + symbolic compute** — strongest fix for the 0% arithmetic classes. Make the LLM emit `(cell_refs, expression)` (e.g. `cells=["E5","E7","E8","E9"]; expr="E5+E7+E8+E9"`), then evaluate the expression in pandas. Routing already locates the correct table; this isolates the arithmetic to a deterministic step.
2. **Try Qwen-14B-Instruct 4-bit or 7B in 8-bit** on the lab PC (≥16 GB VRAM). Expected gain on arithmetic ≥ 10 pp based on the 3B → 7B jump pattern.
3. **Try OpenAI `gpt-4o-mini`** as the reader for comparison — sets a non-local upper bound. Needs `OPENAI_API_KEY`.
4. **Tolerance-tightening pass on `_numeric_match`** — currently 2% rel-tol; close-but-not-exact cases like `pred=34 vs gold=32.1` are scored as miss. Consider widening to 5% with a unit-aware check.
5. **Stratified expansion** — run with `--per-class 25` (≈125 queries) once the reader is stronger, to get tighter per-class confidence intervals.
6. **Document the routing path on a real case** — pick one query, dump every stage (vector hits → keyword/numeric overlaps → reranked scores → reader prompt → answer) into a single appendix file. Useful for the thesis.
