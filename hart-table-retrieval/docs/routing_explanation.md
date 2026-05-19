# Routing: how the agent picks which table to read

The agent keeps **two stores separated** and uses both for routing.

| Store | Purpose | Implementation |
|---|---|---|
| Vector DB (ChromaDB) | Embedding of serialized table text | `src/retrieval/indexer.py`, `data/chroma_db/` |
| Original 2D table | Raw cell values, header paths, title | `sidecar_verifier/store/table_store.py` (pandas DataFrame) |

Routing for a single query runs as a 3-stage cascade.

## Stage 1 — Vector retrieval (`sidecar_verifier/agent/retriever.py`)

1. Embed the query with the same encoder used at indexing time (`bge-large` by default).
2. ChromaDB cosine-NN over `top_k_vectors=20` sub-document vectors.
3. Group hits by `table_id`, keep the *best-scoring* vector per table.
4. Return `top_k_tables` candidates ranked by max sub-doc similarity.

Output per candidate: `{table_id, score, vector_id, chunk_text, meta}`.

This stage's signal is **semantic similarity between query and table text**. It does NOT touch the original 2D table.

## Stage 2 — Query-aware verification (`sidecar_verifier/agent/verifier.py`)

For each candidate, look up the **original** table from the pandas store and compute:

- `keyword_overlap = |query_keywords ∩ (title ∪ header_paths)| / |query_keywords|`
- `numeric_overlap = |query_numbers ∩ table_cells| / |query_numbers|` (1.0 if query has no numbers)
- `confidence = 0.6·keyword_overlap + 0.4·numeric_overlap`

Why two directions: v1 of this code only verified that retrieved chunk text reappeared in the candidate table — which was tautological because the chunk *came from* that table. v2 checks the *query* against the *table*, which is the asymmetric direction we actually care about.

Crucially, this stage is the only one that reads the **original** 2D structure (cell values, full header). The vector DB is intentionally insufficient for verification, by design.

## Stage 3 — Reconciliation (`sidecar_verifier/agent/reconciler.py`)

Three modes:

- `rerank` (default, best on HiTab): `final_score = w_vector·vector_score + w_verify·confidence`. Sweet spot: `w_verify=0.1~0.2`.
- `filter`: drop candidates with `confidence < threshold`, keep vector order.
- `filter+rerank`: both.

Stronger filtering hurt `R@5/R@10` in prior experiments → light reranking is the right call.

## Stage 4 — Read (`sidecar_verifier/agent/answerer.py`)

Top-1 reconciled table is rendered as Markdown + a header-paths block, then passed to a local LLM:

```
You are a precise table QA assistant. Answer ONLY from the table below.
Table:
Title: ...
Header paths (top): col[0]: ... > ...
Data: | r | c0 | c1 | ...
Question: <user query>
Answer:
```

Generation is greedy (`do_sample=False`).

## Trace (`sidecar_verifier/agent/tracer.py`)

After the LLM responds, the tracer maps each number in the answer back to a `(row, col)` cell of the chosen table. Numbers that don't resolve are flagged as hallucinations.

## Sanity-check example

```
query: "how many percentage points did medium-sized companies (50-249 employees) account for?"
gold table = 0_1_nsf21326-tab003
gold formula = =(I12+I13)/I5    (multi-op aggregation)
gold answer  = some %

stage 1 vector top-5:  [tab003, tab001, tab005, tab007, tab009]   # ranked by bge similarity
stage 2 verification (per candidate):
   tab003: kw_overlap=0.83, num_overlap=1.0 → conf=0.90
   tab001: kw_overlap=0.50, num_overlap=0.0 → conf=0.30
   ...
stage 3 rerank (w_v=0.2): tab003 wins
stage 4 read: LLM emits a number → tracer locates (12,8)+(13,8) → confirms grounding
```

The bottleneck is stage 4, not 1–3. With a 3B reader, single-cell lookups work, multi-op formulas (sum, diff, div) fail and the LLM emits "N/A" or the first cell. This is precisely the difficulty class the hard-query eval targets.
