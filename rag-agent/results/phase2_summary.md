# Phase 2 — 검색 베이스라인 요약 (라우팅 없음)

풀=3597표 · 평가=1671 dev 질의 · 임베더=BAAI/bge-large-en-v1.5 · seed=42
best dense 직렬화: `header_path` · 프로토콜: full-corpus DTR R@k; binary single-gold nDCG; max-pool chunks

| method | R@1 | R@5 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| dense_header_path | 0.4901 | 0.7750 | 0.8318 | 0.6158 | 0.6644 |
| dense_plain_markdown | 0.3896 | 0.6685 | 0.7361 | 0.5145 | 0.5629 |
| hybrid_rrf | 0.3758 | 0.6715 | 0.7750 | 0.5136 | 0.5706 |
| dense_json_kv | 0.3214 | 0.6110 | 0.6774 | 0.4520 | 0.5009 |
| bm25 | 0.2406 | 0.4716 | 0.5500 | 0.3469 | 0.3890 |

BM25 튜닝(grid k1×b): 채택 k1=0.9, b=0.4 (전체 grid는 results/phase2_retrieval.json::grids.bm25).

## R@1 paired bootstrap vs best-dense (1000 resample, 95% CI)
| method | Δ(method−best_dense) | CI95 | sig |
|---|---|---|---|
| bm25 | -0.2496 | [-0.2759, -0.2232] | True |
| dense_plain_markdown | -0.1005 | [-0.1269, -0.0754] | True |
| dense_json_kv | -0.1688 | [-0.1939, -0.1436] | True |
| hybrid_rrf | -0.1143 | [-0.1394, -0.0892] | True |

해석(수치): 최고 MRR = `dense_header_path` (0.6158); BM25(튜닝) MRR=0.3469, dense_plain_markdown MRR=0.5145, hybrid_rrf MRR=0.5136.

## 답변 측 end-to-end (top-1 표 → LLM)
LLM=local:Qwen2.5-7B-Instruct-4bit · n=1671 · context=top-1 table, plain_markdown

| baseline | R@1 | EM | NM | F1 |
|---|---|---|---|---|
| oracle (상한) | 1.000 | 0.268 | 0.447 | 0.330 |
| dense_header_path | 0.490 | 0.177 | 0.290 | 0.219 |
| bm25 | 0.241 | 0.101 | 0.157 | 0.124 |
| nocontext (하한) | 0.000 | 0.031 | 0.087 | 0.064 |

Gate 2 상한(oracle≥검색): True · 하한(nocontext≤검색): True

## retrieval–answer gap
across-baseline Spearman ρ(R@1, NM) = 1.000 (p=0.000)
- bm25: P(정답|top1 적중)=0.483 (n=402) vs P(정답|불일치)=0.054 (n=1269)
- dense_header_path: P(정답|top1 적중)=0.494 (n=818) vs P(정답|불일치)=0.094 (n=853)