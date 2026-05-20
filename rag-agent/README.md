# rag-agent

Adaptive table-RAG agent for hard HiTab queries (formulas / functions from
the paper appendix). Original parsed-2D store and Chroma vector store are
kept **separate** — the agent compares the two at verify time.

📊 **Detailed experiment report**: [`EXPERIMENTS.md`](EXPERIMENTS.md) — full
methodology, every bug found in audit, and v1→v2→v3 result progression.

Final headline (HiTab dev, 40 hard queries, Qwen-2.5-7B-Instruct 4-bit reader):

| Metric | Vector only | **After verifier** |
|---|---:|---:|
| R@1 | 0.575 | **0.675** (+10 pp) |
| R@5 | — | **0.875** |
| MRR | — | 0.759 |
| nDCG@10 | — | 0.789 |
| Answer NM | — | **0.450** (existing bench: 0.250) |
| Symbolic exec acc (comparison class) | — | **0.375** |


## Stage flow

```
                    ┌────────────────────────────────┐
       query  ───▶  │ 1. classify_query (rule-based) │
                    └─────────────┬──────────────────┘
                                  ▼
                    ┌────────────────────────────────┐
                    │ 2. plan_stages (route policy)  │
                    └─────────────┬──────────────────┘
                                  ▼
       ┌───────────────┐    ┌────────────────────────────┐
       │ VectorStore   │◀───│ 3. retrieve (top-K vectors)│
       │ Chroma + GPU  │    └─────────────┬──────────────┘
       └───────────────┘                  ▼
                            ┌──────────────────────────────────┐
                            │ 4. verify (keyword + number      │
                            │    overlap vs OriginalStore)     │
                            │    → rerank top-K tables         │
                            └─────────────┬────────────────────┘
                                          ▼
                        ┌────────────────────────────────────────┐
                        │ 5a. SYMBOLIC (arithmetic queries only) │
                        │   LLM emits  {cells, expression}       │
                        │   header-path resolve → safe AST eval  │
                        └─────────────┬──────────────────────────┘
                                      ▼
                        ┌─────────────────────────────────────┐
                        │ 5b. LLM reader (fallback / lookup)  │
                        └─────────────────────────────────────┘
```

- **REASONING_ONLY** intent → skip retrieve+verify+symbolic → LLM only.
- **SIMPLE_LOOKUP / SINGLE_ARG / COMPARISON_OR_COUNT** → retrieve+verify+reader.
- **ARITHMETIC_AGG / MULTI_OP_FORMULA** → +symbolic compute; reader is a fallback.

All skipped stages are recorded in the trace JSON so per-stage metrics can be
computed offline.

## Free LLM options

| Spec | Notes |
|---|---|
| `groq:llama-3.3-70b-versatile` | Free tier ~14k req/day. Recommended reader. |
| `groq:llama-3.1-8b-instant`   | Faster, weaker. Useful as the cell-extractor. |
| `local:Qwen/Qwen2.5-7B-Instruct` | 4-bit on a 3060 Ti. Default fallback. |

Set `GROQ_API_KEY` in env for Groq. The reader and the cell-extractor can be
different models (`--llm` and `--symbolic-llm`).

## Evaluation metrics (paper-aligned)

Retrieval — same as HART/HiTab paper:
- Recall@1, Recall@5
- MRR
- nDCG@10 (binary relevance)

Answer — HiTab §5 + matching the existing hard-query bench's tolerance:
- Exact Match (EM)
- Numeric Match (NM) with ±2% rel-tol, accepting ×100 / ÷100 / abs() variants.

Plus **symbolic execution accuracy** for arithmetic classes (was symbolic eval
correct?) — a custom metric mirroring HiTab Table 9's formula-supervised
execution-accuracy idea.

Per-difficulty-class breakdown matches the existing `run_hard_query_eval.py`:
`multi_op_formula`, `arithmetic_agg`, `pair_or_topk_arg`, `single_arg`,
`comparison_or_count`, `single_op_formula`, `simple_lookup`.

## Quickstart

Assumes the data and Chroma index already exist (re-using
`hart-table-retrieval/data/{hitab,chroma_db}`).

```bash
# from /home/user/T2-1 with the existing T2 venv:
GROQ_API_KEY=... /home/user/T2/hart-table-retrieval/.venv/bin/python \
  rag-agent/scripts/run_eval.py \
  --llm groq:llama-3.3-70b-versatile \
  --per-class 8 --out rag-agent/results/groq_70b.json

# Local-only (no API):
/home/user/T2/hart-table-retrieval/.venv/bin/python \
  rag-agent/scripts/run_eval.py \
  --llm local:Qwen/Qwen2.5-7B-Instruct \
  --retriever-device cpu \
  --per-class 8 --out rag-agent/results/local_qwen.json
```

Output (per-class table + overall) is printed to stdout and saved as JSON.

## Directory layout

```
rag-agent/
├── README.md
├── rag_agent/
│   ├── stores/        # OriginalStore (parsed 2D) + VectorStore (Chroma + GPU embedder)
│   ├── router/        # query_classifier (rule-based) + policy (stage planning)
│   ├── retrieve/      # verifier (keyword/number overlap vs OriginalStore) + rerank
│   ├── extract/       # cell_extractor (LLM JSON output) + symbolic_eval (safe AST)
│   ├── llm/           # BaseLLM, GroqLLM, LocalQwenLLM, build_llm factory
│   ├── agent.py       # orchestrator
│   └── eval/          # paper metrics (R@k, MRR, nDCG, EM, NM) + difficulty class
├── scripts/
│   └── run_eval.py    # benchmark entry point
└── results/           # output JSONs (gitignored)
```

## Why "not HART"

HART's α-blend (cosine on serialized text + header-alignment) did not beat
`plain_markdown` on HiTab dev. This package replaces that scorer with:

1. plain vector retrieval as the first stage (no header alignment),
2. cross-verification against the **original 2D structure** rather than
   the serialized text the retriever already saw,
3. symbolic compute for arithmetic — bypassing the reader entirely on the
   class HART can't fix.
