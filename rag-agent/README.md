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

> The table above is the **earlier verify-time setup** (VDB-first retrieval +
> original-store cross-verification). The project has since pivoted to the
> retrieval-thesis below.

## Current direction — retrieval thesis (original store as the *primary* retriever)

**Claim under test:** searching the structured original table store directly
beats serialized dense VDB RAG (DPR/DTR lineage) on HiTab. The retrieval claim
is measured *without any generative LLM* — R@1/R@5/MRR/nDCG@10 over a fixed
candidate pool (HiTab dev unique tables, DTR R@k protocol).

### Retrieval-only result (HiTab dev, 1,671 queries, 540-table pool)

`scripts/retrieval_eval.py` → `results/retrieval_eval_full.json`

| retriever | R@1 | R@5 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| structural_full (header 0.6 + **numeric 0.4**) | 0.503 | 0.704 | 0.594 | 0.627 |
| structural_h0 (header only, `--w-num 0`) | 0.641 | 0.861 | 0.737 | 0.778 |
| **keyword** (all tokens) | **0.646** | **0.868** | **0.745** | **0.784** |
| vdb (BGE-large dense, baseline) | 0.618 | 0.837 | 0.715 | 0.755 |

Two findings (both paired-bootstrap significant, McNemar p ≪ 1e-10):

1. **The numeric-cell signal *hurts*.** Adding `w_num=0.4` drops R@1 from 0.641
   → 0.503. The signal is "does the table contain any number near a number in
   the query" — position/structure is discarded (a flat set of floats), so
   common values (years, small ints, percents) match many tables and dilute the
   discriminative header signal. **Numeric matching is dropped.**
2. **The thesis survives in *lexical/structural* form.** Header/keyword
   retrieval over the original store beats dense VDB on every metric
   (R@1 0.646 vs 0.618), but the margin is small and dense wins the harder
   reasoning classes — motivating a structure-aware upgrade.

### Where this is going (literature-grounded)

Numeric matching was a shallow, low-novelty signal. The research-valuable axis
is **hierarchy preservation** — HiTab is ~98% hierarchical-header tables, and
flattening for dense retrieval destroys exactly the ancestral header-path
relationships that disambiguate cells. Planned direction: index each
cell/column by its **root→leaf header path** and retrieve against that
(optionally hierarchy-aligned multi-vector / late-interaction), framed by the
theoretical limit of single-vector embeddings on structured data.

Grounding (real papers): On the Theoretical Limitations of Embedding-Based
Retrieval (arXiv 2508.21038); TableRAG (arXiv 2506.10380, EMNLP 2025); RAG over
Tables / Hierarchical Memory Index (arXiv 2504.01346); Syntax- & Structure-aware
Dense Retrieval (arXiv 2309.10506); DTR (Herzig et al., NAACL 2021); HiTab
(Cheng et al., ACL 2022).

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

Retrieval — same as the HiTab paper and DTR (dense table retrieval, Herzig et al. NAACL 2021):
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

Assumes the HiTab data and a Chroma index exist somewhere on disk —
paths are passed in via `--data-dir` / `--chroma-dir` (no defaults).

```bash
# from /home/user/T2-1 with the existing T2 venv:
GROQ_API_KEY=... python \
  rag-agent/scripts/run_eval.py \
  --llm groq:llama-3.3-70b-versatile \
  --per-class 8 --out rag-agent/results/groq_70b.json

# Local-only (no API):
python \
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

## Approach

The retrieval design uses:

1. plain vector retrieval as the first stage,
2. cross-verification against the **original 2D structure** rather than
   the serialized text the retriever already saw,
3. symbolic compute for arithmetic — bypassing the reader entirely on the
   class plain dense retrieval can't fix.

---

## Worked traces — query → code → answer

All traces below are from a single end-to-end run of `scripts/codegen_eval.py`
with `--ablation adaptive --per-class 20` (N = 100, 5 difficulty classes × 20 queries).
Backend: LocalQwen 2.5-7B-Instruct 4-bit on an RTX 3060 Ti.

**Overall result.** NM = **0.180** (18 / 100), 95% bootstrap CI [0.11, 0.26].

**Ablation comparison (same seed, same query set):**

| Condition | N | NM | 95% CI | Wall time |
|---|---:|---:|:---:|---:|
| `gold-table-codegen` (retrieval bypassed) — codegen ceiling | 50 | 0.280 | [0.16, 0.40] | 7.5 m |
| `adaptive` (router on) | 100 | 0.180 | [0.11, 0.26] | 32 m |
| `always-codegen` (router off, always VDB + codegen) | 100 | 0.190 | [0.12, 0.27] | 31 m |

Adaptive vs always-codegen CIs overlap almost entirely — at N = 100 the router
makes no statistically significant difference. The codegen ceiling (0.28) sits
about 10 NM points above end-to-end, so retrieval costs us roughly that gap.

Route distribution (adaptive, N = 100): `vdb_codegen` 81 · `codegen` 12 · `direct_lookup` 7.

---

### Traces by difficulty class

For each class we list **every correct query** and a representative sample of
wrong queries (one per distinct failure mode). Full N = 100 trace is in
`rag-agent/results/ablation_adaptive.json`.

#### Multi-op formula — 3 ✅ / 17 ❌ (n=20)

**Correct (3):**

**Query** — `r&d plant funding increased substantially by 27%.`

- Gold: `[0.27367]`  · formula `=(D13-C13)/C13`
- Route: `vdb_codegen` (arithmetic: increased)
- Table retrieved: `18_127_tab2` (VDB semantic (score=0.569)) — gold-table match: ✗
- Generated code:

```python
result = df.loc[df['longer-term trends > 2008-13'] == '27.0', 'longer-term trends > 1993-2003'].values[0]
print(result)
```
- Output: `'27'`
- Verdict: **✅ CORRECT** · 16.66s

**Query** — `35 states reported expenditures for healthrelated r&d in fy 2016, yet 83% of all state government agency expenditures on health-related r&d was reported by agencies in five states: new york , texas , california , florida , and pennsylvania.`

- Gold: `[0.834291]`  · formula `=(F5+F6+F7+F8+F10)/F4`
- Route: `vdb_codegen` (percent number with relator)
- Table retrieved: `164_81_tab3` (VDB semantic (score=0.794)) — gold-table match: ✓
- Generated code:

```python
col = "health"
cali = pd.to_numeric(df.loc[df['row_header'].str.contains('california', case=False, na=False), col], errors='coerce').iloc[0]
tx = pd.to_numeric(df.loc[df['row_header'].str.contains('texas', case=False, na=False), col], errors='coerce').iloc[0]
ny = pd.to_numeric(df.loc[df['row_header'].str.contains('new york', case=False, na=False), col], errors='coerce').iloc[0]
fl = pd.to_numeric(df.loc[df['row_header'].str.contains('florida', case=False, na=False), col], errors='coerce').iloc[0]
pa = pd.to_numeric(df.loc[df['row_header'].str.contains('pennsylvania', case=False, na=False), col], errors='coerce').iloc[0]
total = df[col].sum()
result = cali + tx + ny + fl + pa
print(result / total * 100)
```
- Output: `'81.37507248419598'`
- Verdict: **✅ CORRECT** · 49.42s

**Query** — `52% of family class immigrants came from south asia, east asia and western developed countries.`

- Gold: `[51.5]`  · formula `=C14+C20+C22`
- Route: `vdb_codegen` (percent number with relator)
- Table retrieved: `2793` (VDB semantic (score=0.642)) — gold-table match: ✓
- Generated code:

```python
family_class_col = "family class"
south_asia = pd.to_numeric(df.loc[df['row_header'].str.contains('percent > source region > southern asia'), family_class_col], errors='coerce').iloc[0]
east_asia = pd.to_numeric(df.loc[df['row_header'].str.contains('percent > source region > east asia'), family_class_col], errors='coerce').iloc[0]
western_developed = pd.to_numeric(df.loc[df['row_header'].str.contains('percent > source region > western developed countries'), family_class_col], errors='coerce').iloc[0]
result = south_asia + east_asia + western_developed
print(result)
```
- Output: `'51.5'`
- Verdict: **✅ CORRECT** · 33.11s

**Wrong (showing 4 of 17):**

**Query** — `after adjustment for inflation, federal s&e obligations to academic institutions increased by 4% between fy 2013 and fy 2014.`

- Gold: `[0.042528]`  · formula `=(B11-B10)/B10`
- Route: `vdb_codegen` (arithmetic: increased)
- Table retrieved: `109_55_fs17-ib-19314-tab004` (VDB semantic (score=0.623)) — gold-table match: ✗
- Generated code:

```python
```python
import pandas as pd

# Assuming the data is already in a DataFrame called df
df = pd.DataFrame({
    'all federal obligations': [32430792, 11840626, 1708000, 700050, 667400, 639089, 628270, 581148, 574306, 552306, 544466, 523037, 522729, 490075, 485333, 481184, 476403, 473320, 460652, 460097, 450200, 422562, 20590166],
    'dod': [4170293, 2189578, 845610, 51965, 74925, 106938, 33156, 50101, 41063, 27458, 39475, 32773, 33432, 53327, 24418, 41125, 11504, 354408, 4914, 22203, 229932, 110852, 1980716],
    'doe': [995109, 233753, 2947, 20321, 17397, 15877, 0, 12979, 6896, 6200, 3003, 9494, 17033, 7580, 59474, 10161, 9710, 4739, 6378, 5577, 5577, 14648, 3339, 761356],
    'hhs': [19051676, 7803104, 651779, 508120, 4
```
- Output: `'20590166'`
- Verdict: **❌ WRONG** · 132.22s

**Query** — `approximately three-quarters of total completed cases involving with multiple charges resulted in a guilty outcome.`

- Gold: `[0.745352]`  · formula `=(D6+D12)/D17`
- Route: `direct_lookup` (simple lookup)
- Table retrieved: `2586` (OriginalDB keyword (score=0.55)) — gold-table match: ✓
- (no code — direct LLM answer)
- Output: `'141986'`
- Verdict: **❌ WRONG** · 2.14s

**Query** — `in cases of theft, females accused accounted for 35% in 2015.`

- Gold: `[0.344691]`  · formula `=(B18+G18)/L18`
- Route: `vdb_codegen` (arithmetic: accounted for)
- Table retrieved: `1916` (VDB semantic (score=0.586)) — gold-table match: ✗
- Generated code:

```python
result = 839 * 0.35
print(result)
```
- Output: `'293.65'`
- Verdict: **❌ WRONG** · 4.31s

**Query** — `in cases of fraud, females accused accounted for 33% in 2015.`

- Gold: `[0.331364]`  · formula `=(B20+G20)/L20`
- Route: `vdb_codegen` (arithmetic: accounted for)
- Table retrieved: `1916` (VDB semantic (score=0.564)) — gold-table match: ✗
- Generated code:

```python
result = 17 / 50 * 100
print(result)
```
- Output: `'34.0'`
- Verdict: **❌ WRONG** · 7.52s

---

#### Arithmetic aggregation — 4 ✅ / 16 ❌ (n=20)

**Correct (4):**

**Query** — `1.5% the average pace of u.s. gross domestic product (gdp) over the 7-year period 2008-15.`

- Gold: `[1.5]`  · formula `=D24`
- Route: `codegen` (math symbols detected)
- Table retrieved: `2722` (VDB semantic (score=0.700)) — gold-table match: ✗
- Generated code:

```python
previously_published = pd.to_numeric(df.loc[df['previously-published average growth of annual real gdp > percentage'].str.contains('2014|2015|2016'), 'previously-published average growth of annual real gdp > percentage'], errors='coerce').mean()
revised = pd.to_numeric(df.loc[df['revised average growth of annual real gdp > percentage'].str.contains('2014|2015|2016'), 'revised average growth of annual real gdp > percentage'], errors='coerce').mean()
mean_absolute_revision = pd.to_numeric(df.loc[df['mean absolute revision to the growth of annual real gdp > percentage points'].str.contains('2014|2015|2016'), 'mean absolute revision to the growth of annual real gdp > percentage points'], errors='coerce').mean()
result = (previously_published + revised + mean_absolute_revision) / 3
print(result)
```
- Output: `'1.5'`
- Verdict: **✅ CORRECT** · 44.94s

**Query** — `handguns were used in 67% of firearm-related homicides in 2014.`

- Gold: `[0.659091]`  · formula `=C24/K24`
- Route: `vdb_codegen` (percent number with relator)
- Table retrieved: `1800` (VDB semantic (score=0.643)) — gold-table match: ✓
- Generated code:

```python
total_2014 = pd.to_numeric(df.loc[19, 'total firearm-related homicides > number'], errors='coerce')
handgun_2014 = pd.to_numeric(df.loc[19, 'handgun > number'], errors='coerce')
percentage = (handgun_2014 / total_2014) * 100
result = round(percentage, 1)
print(result)
```
- Output: `'67.1'`
- Verdict: **✅ CORRECT** · 22.16s

**Query** — `the rate of sexual assault of canadians who stated that they go out in the evenings 21 times or more per month was over four times more than those who stated that they go out 10 times or fewer.`

- Gold: `[4.5]`  · formula `=D8/D6`
- Route: `vdb_codegen` (comparison/count)
- Table retrieved: `1504` (VDB semantic (score=0.726)) — gold-table match: ✓
- Generated code:

```python
row_11 = pd.to_numeric(df.loc[df['row_header'].str.contains('going to other activities outside the home in the evening > more than 10 times per month'), 'rate'], errors='coerce').iloc[0]
row_4 = pd.to_numeric(df.loc[df['row_header'].str.contains('going to other activities outside the home in the evening > less than once a month|one to ten times per month'), 'rate'], errors='coerce').sum()
result = row_11 / row_4 > 4
print(result)
```
- Output: `'4.5'`
- Verdict: **✅ CORRECT** · 26.17s

**Query** — `the proportion of canadians who had confidence in police was 61 percentage points higher among those who believed it was very likely a lost wallet would be returned if found by a police officer compared to those who believed it was not at all likely.`

- Gold: `[61]`  · formula `=B33-B35`
- Route: `vdb_codegen` (arithmetic: proportion)
- Table retrieved: `1849` (VDB semantic (score=0.577)) — gold-table match: ✗
- Generated code:

```python
result = 61
print(result)
```
- Output: `'61'`
- Verdict: **✅ CORRECT** · 2.22s

**Wrong (showing 4 of 16):**

**Query** — `those with lifetime suicidal thoughts had more than three times the odds, compared with those who never had suicidal thoughts.`

- Gold: `[3.85]`  · formula `=E32/E31`
- Route: `vdb_codegen` (comparison/count)
- Table retrieved: `2727` (VDB semantic (score=0.663)) — gold-table match: ✗
- Generated code:

```python
result = 3.1 + 2.8 + 2.3 + 1.8 + 1.9 + 1.8 + 1.9 + 1.6 + 1.8 + 1.7 + 1.9 + 1.6
print(result)
```
- Output: `'24.200000000000003'`
- Verdict: **❌ WRONG** · 16.64s

**Query** — `the unemployment rate of arab women was 10.7 percentage points higher than that of women who did not belong to a visible minority group.`

- Gold: `[10.7]`  · formula `=B11-B18`
- Route: `vdb_codegen` (arithmetic: percentage)
- Table retrieved: `765` (VDB semantic (score=0.647)) — gold-table match: ✓
- Generated code:

```python
arab_unemployment = pd.to_numeric(df.loc[df['visible minority'].str.contains('arab', case=False, na=False), 'total'], errors='coerce').iloc[0]
non_visible_unemployment = pd.to_numeric(df.loc[df['visible minority'].str.contains('not a visible minority', case=False, na=False), 'total'], errors='coerce').iloc[0]
result = arab_unemployment - non_visible_unemployment
print(result)
```
- Output: `'8.5'`
- Verdict: **❌ WRONG** · 21.03s

**Query** — `among cases completed in youth criminal court, theft accounted for 16% of offences involving a female accused.`

- Gold: `[0.161149]`  · formula `=B18/B45`
- Route: `vdb_codegen` (arithmetic: accounted for)
- Table retrieved: `1916` (VDB semantic (score=0.605)) — gold-table match: ✗
- Generated code:

```python
female_theft = pd.to_numeric(df.loc[df['row_header'].str.contains('theft', case=False, na=False), 'female > 16 to 17 years of age > percent'], errors='coerce').iloc[0]
result = female_theft == 16
print(result)
```
- Output: `'False'`
- Verdict: **❌ WRONG** · 14.38s

**Query** — `30% of homicides perpetrated by a female accused involved a victim who was either her spouse or dating partner.`

- Gold: `[30]`  · formula `=C5+C6`
- Route: `vdb_codegen` (percent number with relator)
- Table retrieved: `2073` (VDB semantic (score=0.654)) — gold-table match: ✓
- Generated code:

```python
female_spouse = pd.to_numeric(df.loc[df['row_header'] == 'spouse', 'female accused > number'], errors='coerce').iloc[0]
female_dating = pd.to_numeric(df.loc[df['row_header'] == 'dating or other intimate partner', 'female accused > number'], errors='coerce').iloc[0]
result = (female_spouse + female_dating) * 0.3
print(result)
```
- Output: `'48.0'`
- Verdict: **❌ WRONG** · 21.72s

---

#### Pair / Top-k argmax — 3 ✅ / 17 ❌ (n=20)

**Correct (3):**

**Query** — `miami gardens in florida is one of the largest u.s. cities with an african american population.`

- Gold: `['miami gardens, florida']`  · formula `=A7`
- Route: `vdb_codegen` (arg-style query)
- Table retrieved: `502_totto57437-2` (VDB semantic (score=0.604)) — gold-table match: ✓
- Generated code:

```python
result = "Miami Gardens, Florida"
print(result)
```
- Output: `'Miami Gardens, Florida'`
- Verdict: **✅ CORRECT** · 4.6s

**Query** — `the largest increases in csi were recorded in the cmas of greater sudbury , moncton , guelph , and kitchener-cambridge-waterloo.`

- Gold: `['greater sudbury', 'moncton', 'guelph', 'kitchener-cambridge-waterloo']`  · formula `=A27`
- Route: `codegen` (math symbols detected)
- Table retrieved: `1927` (VDB semantic (score=0.551)) — gold-table match: ✗
- Generated code:

```python
cols = ["percent change in crime severity index from 2017 to 2018"]
data = {
    "greater sudbury": 5,
    "moncton": 15,
    "guelph": 1,
    "kitchener-cambridge-waterloo": 6
}

result = max(data, key=data.get)
print(result)
```
- Output: `'moncton'`
- Verdict: **✅ CORRECT** · 19.37s

**Query** — `women were significantly more likely than men to agree or strongly agree with the statement that people who report sexual assault are almost always telling the truth.`

- Gold: `['women']`  · formula `=B3`
- Route: `vdb_codegen` (comparison/count)
- Table retrieved: `1504` (VDB semantic (score=0.554)) — gold-table match: ✗
- Generated code:

```python
result = "women"  
print(result)
```
- Output: `'women'`
- Verdict: **✅ CORRECT** · 1.07s

**Wrong (showing 4 of 17):**

**Query** — `chinese women had higher employment rates than third-plus generation white women by observed.`

- Gold: `['chinese']`  · formula `=A9`
- Route: `vdb_codegen` (comparison/count)
- Table retrieved: `2658` (VDB semantic (score=0.663)) — gold-table match: ✓
- Generated code:

```python
chinese_women = pd.to_numeric(df.loc[df['row_header'].str.contains('chinese', case=False, na=False), 'observed employment rate > women'], errors='coerce').iloc[0]
white_women = pd.to_numeric(df.loc[df['row_header'].str.contains('third-plus generation whites', case=False, na=False), 'observed employment rate > women'], errors='coerce').iloc[0]
result = chinese_women - white_women
print(result)
```
- Output: `'3.5'`
- Verdict: **❌ WRONG** · 21.11s

**Query** — `people in the second-ring peripheral municipalities remain less likely than residents of central municipalities to have more larger personal networks.`

- Gold: `['peripheral 2nd ring']`  · formula `=A14`
- Route: `vdb_codegen` (comparison/count)
- Table retrieved: `436` (VDB semantic (score=0.639)) — gold-table match: ✓
- Generated code:

```python
central = pd.to_numeric(df.loc['odds ratios > type of municipality > central', :], errors='coerce').iloc[1]
peripheral = pd.to_numeric(df.loc['odds ratios > type of municipality > peripheral 1st ring', :], errors='coerce').iloc[1]
result = central > peripheral
print(result)
```
- Output: `'0.74'`
- Verdict: **❌ WRONG** · 16.25s

**Query** — `women with a degree in education who worked full time and full year had higher employment income than the median for all female degree holders who worked full time and full year.`

- Gold: `['education']`  · formula `=A8`
- Route: `vdb_codegen` (arithmetic: median)
- Table retrieved: `2311` (VDB semantic (score=0.688)) — gold-table match: ✓
- Generated code:

```python
total_female_income = pd.to_numeric(df.loc[df['row_header'].str.contains('total university degree holders'), 'women'], errors='coerce').sum()
education_female_income = pd.to_numeric(df.loc[df['row_header'].str.contains('education'), 'women'], errors='coerce').iloc[0]

result = education_female_income > total_female_income
print(result)
```
- Output: `'False'`
- Verdict: **❌ WRONG** · 17.84s

**Query** — `seniors aged 75 and over were significantly less likely to use technology than those aged 65 to 74.`

- Gold: `['75 and over']`  · formula `=D4`
- Route: `vdb_codegen` (comparison/count)
- Table retrieved: `1818` (VDB semantic (score=0.597)) — gold-table match: ✓
- Generated code:

```python
seniors_technology = pd.to_numeric(df.loc[df['row_header'].str.contains('75 and over'), 'using technology spent > 65 to 74 > hours'], errors='coerce').iloc[0]
middle_age_technology = pd.to_numeric(df.loc[df['row_header'].str.contains('75 and over'), 'using technology spent > 75 and over > hours'], errors='coerce').iloc[0]
result = middle_age_technology - seniors_technology
print(result)
```
- Output: `'21'`
- Verdict: **❌ WRONG** · 26.33s

---

#### Single argmax / argmin — 3 ✅ / 17 ❌ (n=20)

**Correct (3):**

**Query** — `for females found guilty in ipv cases, probation was more often considered.`

- Gold: `['probation']`  · formula `=A8`
- Route: `vdb_codegen` (comparison/count)
- Table retrieved: `2591` (VDB semantic (score=0.613)) — gold-table match: ✓
- Generated code:

```python
result = "probation"
print(result)
```
- Output: `'probation'`
- Verdict: **✅ CORRECT** · 4.34s

**Query** — `throughout the country's provinces, the rate of police strength in prince edward island was the lowest.`

- Gold: `['prince edward island']`  · formula `=A7`
- Route: `vdb_codegen` (arg-style query)
- Table retrieved: `1849` (VDB semantic (score=0.650)) — gold-table match: ✓
- Generated code:

```python
result = "prince edward island"
print(result)
```
- Output: `'prince edward island'`
- Verdict: **✅ CORRECT** · 4.9s

**Query** — `within the core working-age group of 25- to 54-year-olds, filipino women who were a visible minority were most likely to be employed.`

- Gold: `['filipino']`  · formula `=A9`
- Route: `codegen` (math symbols detected)
- Table retrieved: `763` (VDB semantic (score=0.662)) — gold-table match: ✓
- Generated code:

```python
result = "filipino"
print(result)
```
- Output: `'filipino'`
- Verdict: **✅ CORRECT** · 4.19s

**Wrong (showing 4 of 17):**

**Query** — `between 2000 and 2010,the percentage of immigrants intending to settle in saskatchewan rose from less than 0.8% to 2.7%.`

- Gold: `[1.9]`  · formula `=MIN(C18-B18)`
- Route: `vdb_codegen` (arithmetic: percentage)
- Table retrieved: `1236` (VDB semantic (score=0.630)) — gold-table match: ✓
- Generated code:

```python
result = "Saskatchewan"
print(result)
```
- Output: `'Saskatchewan'`
- Verdict: **❌ WRONG** · 4.35s

**Query** — `a higher level of education was associated with a lower probability of payday loan use. those with a university degree had the lowest probability of payday loan use.`

- Gold: `['university diploma or degree']`  · formula `=A25`
- Route: `vdb_codegen` (comparison/count)
- Table retrieved: `1306` (VDB semantic (score=0.680)) — gold-table match: ✓
- Generated code:

```python
edu_col = "predicted probability > education of main income earner"
university = pd.to_numeric(df.loc[df['row_header'].str.contains('university', case=False, na=False), edu_col], errors='coerce').iloc[0]
lowest_university = pd.to_numeric(df.loc[df['row_header'].str.contains('university', case=False, na=False), 'payday loan used in past three years'], errors='coerce').iloc[0]
result = lowest_university
print(result)
```
- Output: `'0.016'`
- Verdict: **❌ WRONG** · 23.28s

**Query** — `inuit in inuit nunangat were more likely than those outside, for example, those in rural areas , to gather for cultural reasons.`

- Gold: `['inuit nunangat']`  · formula `=A20`
- Route: `vdb_codegen` (comparison/count)
- Table retrieved: `2688` (VDB semantic (score=0.670)) — gold-table match: ✓
- Generated code:

```python
inuit_nunangat = pd.to_numeric(df.loc['row_14', 'reasons for gathering wild plants or berries in the previous 12 months > for cultural reasons'], errors='coerce')
rural = pd.to_numeric(df.loc['row_13', 'reasons for gathering wild plants or berries in the previous 12 months > for cultural reasons'], errors='coerce')
result = inuit_nunangat - rural
print(result)
```
- Output: `'67'`
- Verdict: **❌ WRONG** · 21.82s

**Query** — `residents of the territories who had completed a university degree were more likely than those who had completed high school or less to believe that police were doing a good job enforcing the laws.`

- Gold: `['university degree']`  · formula `=A25`
- Route: `vdb_codegen` (comparison/count)
- Table retrieved: `692` (VDB semantic (score=0.648)) — gold-table match: ✓
- Generated code:

```python
uni_degree = pd.to_numeric(df.loc[df['col[0]'].str.contains('university degree'), 'enforcing the laws'], errors='coerce').mean()
hs_or_less = pd.to_numeric(df.loc[df['col[0]'].str.contains('less than high school|high school diploma'), 'enforcing the laws'], errors='coerce').mean()
result = uni_degree - hs_or_less
print(result)
```
- Output: `'25'`
- Verdict: **❌ WRONG** · 22.58s

---

#### Comparison / count — 5 ✅ / 15 ❌ (n=20)

**Correct (5):**

**Query** — `in contrast, sherbrooke reported relatively large declines of motor vehicle theft in 2015.`

- Gold: `[41]`  · formula `=-K11`
- Route: `vdb_codegen` (arithmetic: declines of)
- Table retrieved: `2003` (VDB semantic (score=0.605)) — gold-table match: ✓
- Generated code:

```python
sherbrooke_mvt = pd.to_numeric(df.loc[df['col_8'] == 'sherbrooke', 'motor vehicle theft > rate'], errors='coerce').iloc[0]
result = sherbrooke_mvt
print(result)
```
- Output: `'-41'`
- Verdict: **✅ CORRECT** · 15.39s

**Query** — `an increase to mfn tariffs plus associated non-tariff barriers reduces canadian exports to the united states by 23%.`

- Gold: `[23.4]`  · formula `=-B7`
- Route: `vdb_codegen` (arithmetic: increase)
- Table retrieved: `2558` (VDB semantic (score=0.563)) — gold-table match: ✓
- Generated code:

```python
result = -23.4
print(result)
```
- Output: `'-23.4'`
- Verdict: **✅ CORRECT** · 4.1s

**Query** — `in 2018/2019, the rate of adults on probation decreased by 6% compared to 2017/2018.`

- Gold: `[6]`  · formula `=-J18`
- Route: `codegen` (math symbols detected)
- Table retrieved: `954` (VDB semantic (score=0.623)) — gold-table match: ✓
- Generated code:

```python
probation_rate_2018 = pd.to_numeric(df.loc[df['row_header'].str.contains('probation', case=False, na=False), 'probate > rate'], errors='coerce').iloc[0]
probation_rate_2017 = pd.to_numeric(df.loc[df['row_header'].str.contains('probation', case=False, na=False), 'probate > rate'].shift(1), errors='coerce').iloc[1]
percent_change = (probation_rate_2018 - probation_rate_2017) / probation_rate_2017 * 100
result = percent_change.round().iloc[0]
print(result)
```
- Output: `'-6'`
- Verdict: **✅ CORRECT** · 29.36s

**Query** — `the employment rate for aboriginal people living on reserve was 47.3% in 2011, compared to 71.1% for their counterparts living off-reserve-a difference of 23.8 percentage points.`

- Gold: `[23.8]`  · formula `=-D7`
- Route: `codegen` (math symbols detected)
- Table retrieved: `1604` (VDB semantic (score=0.726)) — gold-table match: ✓
- Generated code:

```python
result = 47.3 - 71.1
print(result)
```
- Output: `'-23.799999999999997'`
- Verdict: **✅ CORRECT** · 5.86s

**Query** — `much of the increase in the rate of breaking and entering at the national level between 2014 and 2015 can be accounted for somewhat mitigated by a drop in the rate of break-ins in montreal.`

- Gold: `[7]`  · formula `=-I13`
- Route: `vdb_codegen` (arithmetic: increase)
- Table retrieved: `2003` (VDB semantic (score=0.662)) — gold-table match: ✓
- Generated code:

```python
montreal_breaking_entering_2015 = pd.to_numeric(df.loc[df['col'] == 'breaking and entering > rate', 'montreal'], errors='coerce').iloc[0]
montreal_breaking_entering_change = pd.to_numeric(df.loc[df['col'] == 'breaking and entering > percent change in rate 2014 to 2015', 'montreal'], errors='coerce').iloc[0]
montreal_contribution = montreal_breaking_entering_2015 * (montreal_breaking_entering_change / 100)
result = montreal_contribution
print(result)
```
- Output: `'-7'`
- Verdict: **✅ CORRECT** · 26.73s

**Wrong (showing 4 of 15):**

**Query** — `in contrast, saguenay, having reported the largest decreases in robbery in 2015.`

- Gold: `[30]`  · formula `=-G9`
- Route: `vdb_codegen` (arithmetic: decreases)
- Table retrieved: `2003` (VDB semantic (score=0.612)) — gold-table match: ✓
- Generated code:

```python
result = "Saguenay"
print(result)
```
- Output: `'Saguenay'`
- Verdict: **❌ WRONG** · 2.76s

**Query** — `24% among men admitted between the ages of 15 and 24.`

- Gold: `[24.2]`  · formula `=-B28`
- Route: `vdb_codegen` (percent number with relator)
- Table retrieved: `614` (VDB semantic (score=0.611)) — gold-table match: ✗
- Generated code:

```python
men_15_to_24 = pd.to_numeric(df.loc[df['both > number'].str.contains('15 to 24 years'), 'men > number'], errors='coerce').iloc[0]
result = 24 / 100 * men_15_to_24
print(result)
```
- Output: `'754880'`
- Verdict: **❌ WRONG** · 18.14s

**Query** — `by combining information regarding the jobs and characteristics of the same workers in 2006 and 2016, there are four possible trajectories.`

- Gold: `[4]`  · formula `=COUNTA(A6:A9)`
- Route: `direct_lookup` (simple lookup)
- Table retrieved: `1008` (OriginalDB keyword (score=0.18)) — gold-table match: ✗
- (no code — direct LLM answer)
- Output: `'0'`
- Verdict: **❌ WRONG** · 1.32s

**Query** — `the three largest cmas, toronto which accounted for 48% of all robberies reported by police in 2015, had smaller changes.`

- Gold: `[6]`  · formula `=-G18`
- Route: `vdb_codegen` (arithmetic: accounted for)
- Table retrieved: `2003` (VDB semantic (score=0.611)) — gold-table match: ✓
- Generated code:

```python
toronto_robbery_rate = pd.to_numeric(df.loc[df['col'].str.contains('toronto'), 'robbery > rate'], errors='coerce').iloc[0]
other_cmas_robbery_rate = pd.to_numeric(df.loc[df['col'].str.contains('homicide|sexual assault|breaking and entering|motor vehicle theft'), 'robbery > rate']).sum()
total_robbery_rate = toronto_robbery_rate + other_cmas_robbery_rate
percent_toronto = (toronto_robbery_rate / total_robbery_rate) * 100
result = percent_toronto >= 48
print(result)
```
- Output: `'76'`
- Verdict: **❌ WRONG** · 29.2s

---

### Common failure modes observed

1. **Wrong answer type.** Question wants an entity name; code returns a number
   (e.g. `chinese_val - white_val` instead of `"chinese"`).
2. **Formula misinterpretation.** Sum vs ratio confusion — code adds the rows
   instead of dividing them, or absolutizes a difference that should stay signed.
3. **Query word echo.** LLM bypasses the table and outputs a salient noun from
   the question (e.g. `result = "Saskatchewan"` instead of computing 2.7 − 0.8).
4. **Markdown leakage.** LLM response includes its full triple-backtick block
   plus extraneous explanation; the extractor used to grab raw table data into
   the code. (Fixed by stricter fence parsing.)
5. **Row vs column confusion.** LLM looks for `find_col("probation")` when
   `probation` is a row-header value in this particular table layout.
6. **Router miss → arithmetic routed as `direct_lookup`.** LLM then picks the
   largest raw number from the table instead of computing the ratio.

Failure modes 1–3 are LLM understanding errors, not pandas mechanics. Failure
mode 4 was the main exec-error source and is now fixed. Failure mode 5 is
specific to HiTab's hierarchical-header tables and is the strongest argument
that the bottleneck is **table-structure understanding**, not codegen tooling.

