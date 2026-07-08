# Retrieval algorithms — study + what fits our problem

Re-studied the retrieval toolbox to fix the column-axis precision bottleneck (and
the "small+complete" goal). Key realization: **our resolver uses the two *weakest*
matching methods, and our column problem is exactly "schema linking", whose SOTA
uses a method we aren't using (cross-encoder reranking).**

## The retrieval-architecture ladder (weak → strong matching)

| family | how it matches | accuracy | cost | we use? |
|---|---|---|---|---|
| sparse lexical (BM25/TF-IDF) | term overlap | low (no vocab bridge) | tiny | ~yes (lexical col) |
| dense **bi-encoder** (BGE/e5) | pooled vectors, cosine, **no query↔cand interaction** | medium | tiny | yes (embed resolver) |
| late interaction (ColBERT) | token-level MaxSim | high | low | no |
| **cross-encoder reranker** | query×candidate **joint attention** | **highest (+~10 nDCG over bi-encoder)** | high *per pair* | **no** |
| learned sparse (SPLADE) | learned term expansion (%↔percent) | high | low | no |

Production pattern = **cascade**: a cheap front-end (BM25/bi-encoder) shortlists ~100,
then a **cross-encoder reranks** the shortlist (cross-encoders are too slow to score
everything, only a shortlist).

## Why this nails our column problem

Our failed column fixes (last-col, bi-encoder embed, keyword) all use weak matching.
But **a table has only 3–8 columns** — the "shortlist" is already tiny. So we can
afford the *most accurate* method on **every** column with negligible cost:

→ **cross-encoder reranking of (query, column-header) for each column.**

A cross-encoder jointly attends query↔header, so it resolves the exact mismatches
that broke us: **"percentage" ↔ "%"**, **"per man / times more likely" ↔ "prevalence
per 100,000"**, **"multiple relationship" ↔ "odds ratio"** — vocabulary gaps a
bi-encoder's pooled cosine and lexical overlap both miss.

### This is literally "schema linking" (text-to-SQL), and its SOTA agrees
Mapping a question to the relevant **columns** is the schema-linking task. Recent
schema-linking work selects columns with exactly this:
- **CE-SL / RESDSQL** — a **cross-encoder (BGE-reranker / RoBERTa)** scores the query
  against each column individually. (the method we should adopt)
- **CHESS** — an **LLM** scores query×column (heavier; our W4b showed LLM helps the
  row axis too).
- **LitE-SQL** — vector/bi-encoder retrieval = *what we currently do* (the weaker one).
- **Extractive schema linking** — recall-oriented per-column probabilities (fits a
  completeness objective: keep every column above a recall threshold).

So our column axis is behind schema-linking SOTA by exactly one step: bi-encoder →
cross-encoder.

## Secondary levers (for implicit operands / recall)

- **Query expansion / HyDE** (generate a hypothetical answer, match on it) bridges the
  vocabulary gap and lifts recall (+~4% R@20 in studies). Could surface the implicit
  **total/denominator row** and maybe a **year** the query never names — our other
  diagnosed gaps. Cheap, optional.
- **Learned sparse (SPLADE)** also expands terms (%↔percent) but a cross-encoder is
  simpler for our tiny candidate set.

## Recommendation (grounded next step)

1. **Primary: add a cross-encoder reranker resolver** (e.g. `BAAI/bge-reranker-base`)
   for the **column axis** — score (query, each column header path), keep the top
   column(s) above a recall threshold. Replaces the bi-encoder/lexical column step.
   Tiny cost (≤8 columns), highest-accuracy method, and it directly targets the
   diagnosed "%↔percentage" metric-column failures. Backed by schema-linking SOTA.
   - Likely also helps the **row axis** (same vocabulary-gap mechanism); test both.
2. **Measure**: col-axis coverage + OSC + mean cells (LLM-free first), then answer
   accuracy. Expectation: col-cov up *without* the whole-axis dump → smaller AND more
   complete (unlike the heuristics, which traded one for the other).
3. **Secondary (if needed): HyDE-style query expansion** for the implicit total row /
   unnamed year.

Why this is different from the failed heuristics: those *guessed* a column (last /
keyword) or used weak matching (bi-encoder). A cross-encoder *reads* the query and
each header together — the accuracy jump (+~10 nDCG in IR; SOTA in schema linking) is
exactly what the precise-column-selection problem needs.

## Sources
- IR architectures / cross-encoder vs bi-encoder vs ColBERT/SPLADE:
  https://arxiv.org/pdf/2502.14822 , https://arxiv.org/html/2404.13950v1
- Schema linking (column selection) SOTA: https://arxiv.org/html/2501.17174v1 (Extractive),
  https://arxiv.org/html/2510.14296v1 (bidirectional retrieval),
  https://arxiv.org/html/2510.09014v1 (LitE-SQL, vector-based)
- Query expansion / HyDE: https://arxiv.org/pdf/2305.03653 ,
  https://www.emergentmind.com/topics/hypothetical-document-embeddings-hyde
