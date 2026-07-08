# All-datasets retrieval sweep — BM25 vs dense vs hybrid (no LLM)

Table-retrieval only (rank gold table in full per-dataset pool). Embedder=BAAI/bge-large-en-v1.5 (GPU), BM25Okapi k1=1.5/b=0.75, hybrid=RRF(k=60). Metrics: R@1/R@5/R@10/MRR/nDCG@10, binary single-gold.

Script: `scripts/multidataset_retrieval.py` · JSON: `results/multidataset_retrieval.json`


## hitab  (pool=540, n=1671)

|                      | bm25 | dense | hybrid |
|----------------------|------|-------|--------|
| R@1                  | 0.574 | 0.703 | 0.665 |
| R@5                  | 0.780 | 0.904 | 0.879 |
| R@10                 | 0.844 | 0.939 | 0.924 |
| MRR                  | 0.668 | 0.793 | 0.765 |
| nDCG@10              | 0.706 | 0.827 | 0.801 |

**R@1 by query type (aggregation, n>=15):**

|                      | bm25 | dense | hybrid |
|----------------------|------|-------|--------|
| none (n=1195)        | 0.562 | 0.711 | 0.659 |
| pair-argmax (n=107)  | 0.645 | 0.757 | 0.757 |
| div (n=91)           | 0.527 | 0.549 | 0.571 |
| argmax (n=64)        | 0.594 | 0.672 | 0.672 |
| sum (n=48)           | 0.458 | 0.521 | 0.542 |
| opposite (n=44)      | 0.636 | 0.818 | 0.614 |
| pair-argmin (n=36)   | 0.639 | 0.639 | 0.611 |
| diff (n=23)          | 0.739 | 0.913 | 0.913 |
| argmin (n=15)        | 0.600 | 0.867 | 0.933 |

**R@1 by gold-operand count:**

|                      | bm25 | dense | hybrid |
|----------------------|------|-------|--------|
| 1_operand (n=1398)   | 0.583 | 0.723 | 0.673 |
| 2_operands (n=189)   | 0.582 | 0.646 | 0.677 |
| 3+_operands (n=84)   | 0.405 | 0.488 | 0.512 |

## finqa  (pool=883, n=883)

|                      | bm25 | dense | hybrid |
|----------------------|------|-------|--------|
| R@1                  | 0.117 | 0.156 | 0.168 |
| R@5                  | 0.419 | 0.518 | 0.559 |
| R@10                 | 0.532 | 0.615 | 0.670 |
| MRR                  | 0.251 | 0.310 | 0.333 |
| nDCG@10              | 0.309 | 0.376 | 0.408 |

**R@1 by query type (aggregation, n>=15):**

|                      | bm25 | dense | hybrid |
|----------------------|------|-------|--------|
| finqa_program (n=883) | 0.117 | 0.156 | 0.168 |

**R@1 by gold-operand count:**

|                      | bm25 | dense | hybrid |
|----------------------|------|-------|--------|
| 1_operand (n=238)    | 0.067 | 0.063 | 0.097 |
| 2_operands (n=174)   | 0.144 | 0.144 | 0.149 |
| 3+_operands (n=471)  | 0.132 | 0.208 | 0.210 |

## wikisql  (pool=2630, n=8421)

|                      | bm25 | dense | hybrid |
|----------------------|------|-------|--------|
| R@1                  | 0.482 | 0.357 | 0.534 |
| R@5                  | 0.665 | 0.552 | 0.750 |
| R@10                 | 0.734 | 0.628 | 0.819 |
| MRR                  | 0.570 | 0.452 | 0.632 |
| nDCG@10              | 0.602 | 0.486 | 0.672 |

**R@1 by query type (aggregation, n>=15):**

|                      | bm25 | dense | hybrid |
|----------------------|------|-------|--------|
| none (n=6017)        | 0.511 | 0.378 | 0.554 |
| count (n=779)        | 0.457 | 0.330 | 0.501 |
| max (n=507)          | 0.398 | 0.312 | 0.481 |
| min (n=468)          | 0.372 | 0.267 | 0.483 |
| avg (n=329)          | 0.380 | 0.289 | 0.435 |
| sum (n=321)          | 0.396 | 0.305 | 0.489 |

**R@1 by gold-operand count:**

|                      | bm25 | dense | hybrid |
|----------------------|------|-------|--------|
| 1_operand (n=884)    | 0.385 | 0.307 | 0.471 |
| 2_operands (n=5502)  | 0.447 | 0.330 | 0.501 |
| 3+_operands (n=2035) | 0.619 | 0.452 | 0.650 |
