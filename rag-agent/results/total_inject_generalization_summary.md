# Does the total-row injection win transfer beyond HiTab?

Tests whether the HiTab OSC win (structural total-row injection beats BM25/dense/hybrid)
generalizes to FinQA and WikiSQL. Population: gold-operand count m>=2 per dataset.
LLM-free. Scripts: `scripts/finqa_total_inject.py`, diagnostic in `/tmp/diag_*`.

## Applicability diagnostic — is there a total row to inject at all?

| dataset | m>=2 n | gold tables w/ NAMED total row | degenerate (empty-path flags >50% rows) | median gold-table rows |
|---------|-------:|-------------------------------:|----------------------------------------:|-----------------------:|
| hitab   |    167 | **60.5%**                      | 13.2%                                   | (large, hierarchical)  |
| finqa   |    645 | 39.4%                          | 0.8%                                    | **5** (min 1, max 17)  |
| wikisql |  7537  | **0.0%**                       | **75.1%**                               | (flat relational)      |

## FinQA — injection adds ~nothing (tables too small to have a blind spot)

OSC plain vs +total-row injection, per retriever:

| method | k  | OSC plain | OSC +tot |   Δ    | cells_p | cells_a |
|--------|----|-----------|----------|--------|---------|---------|
| dense  | 1  | 0.1643    | 0.1984   | +0.034 | 1.6     | 2.0     |
| dense  | 3  | 0.2605    | 0.2729   | +0.012 | 5.4     | 5.5     |
| dense  | 10 | 0.3147    | 0.3147   | +0.000 | 10.2    | 10.2    |
| dense  | 20 | 0.3147    | 0.3147   | +0.000 | 10.3    | 10.3    |

(bm25/hybrid identical pattern.) mean total-cells injected/query = **0.1**.

## Read

**The win does NOT transfer — for two distinct structural reasons, and that sharpens
the contribution's scope rather than weakening it.**

1. **WikiSQL (flat relational tables): mechanism inapplicable.** There is no stored
   aggregate/total cell — sum/max/count are computed over a column on the fly. 0% of
   gold tables have a named total row; `is_total_row`'s empty-path fallback degenerates
   (flags 75% of rows). "Injecting the total row" would mean dumping most of the table.

2. **FinQA (small financial tables, median 5 rows): mechanism unnecessary.** Total rows
   do exist (39%), but tables are so small that any budget k>=10 already retrieves the
   whole table — the total row is already in context, so injection adds 0 cells / 0 OSC.
   There is no retrieval blind spot to patch. FinQA's ceiling (OSC **0.315** even with
   the full table in context) is **operand decomposition**, not total-row reachability.

**Why HiTab is the regime where injection wins:** it is the only one with *both*
(a) tables large enough that a realistic budget cannot cover the whole table, *and*
(b) structurally-dissimilar **named** total rows that similarity retrieval misses.
The contribution is therefore: *retriever-agnostic structural total-row injection for
large hierarchical tables with stored aggregates* — not a universal table-RAG fix.
