# 병목 진단 (2026-09-16)

K_LADDER_TABLES-2026-09-16.md 의 k-사다리 현상(recall 상승 / 검색성공시 정확도 하락 / 전체 정확도 완만 상승)을 더 세분화한 진단. 기존 ESM/Recall/Precision/F1 정의·명칭은 바꾸지 않는다.

## 1. gold-cell rank 버킷별 N / 답변 정확도

모집단: hitab test primary (mode=all, m=1, aggregation=none) (query count=991, gold_rank 미확인 6건 → '>20' 버킷에 포함)

| rank | N | QA k=1 | QA k=5 | QA k=10 | QA k=20 |
|---|---|---|---|---|---|
| 1 | 570 | 0.9877 | 0.8632 | 0.8544 | 0.8491 |
| 2 | 112 | 0.0804 | 0.6339 | 0.5714 | 0.5179 |
| 3 | 58 | 0.0172 | 0.4655 | 0.3621 | 0.3966 |
| 4-5 | 51 | 0.0588 | 0.6863 | 0.5686 | 0.5294 |
| 6-10 | 71 | 0.0704 | 0.0704 | 0.507 | 0.4648 |
| 11-20 | 44 | 0.0909 | 0.0455 | 0.0682 | 0.3864 |
| >20 | 85 | 0.0235 | 0.0235 | 0.0235 | 0.0353 |

## 2. top-1 검색 오류 분류

모집단: hitab test primary (mode=all, m=1, aggregation=none) (query count=991, top-1 정답 570건, top-1 오답 421건)

| class | N | 오답 중 비율 |
|---|---|---|
| same_value | 22 | 0.0523 |
| wrong_row | 113 | 0.2684 |
| wrong_column | 143 | 0.3397 |
| same_leaf_header | 15 | 0.0356 |
| nearby_cell | 11 | 0.0261 |
| wrong_table | 102 | 0.2423 |
| other | 15 | 0.0356 |

## 3. Controlled QA (gold + 통제된 distractor)

| condition | query count | dropped | 답변 정확도 |
|---|---|---|---|
| gold_1_hard_negative_first | 150 | 0 | 0.84 |
| gold_1_hard_negative_last | 150 | 0 | 0.76 |
| gold_1_random_first | 150 | 0 | 0.94 |
| gold_1_random_last | 150 | 0 | 0.94 |
| gold_1_same_value_first | 124 | 26 | 0.9839 |
| gold_1_same_value_last | 124 | 26 | 0.9919 |
| gold_4_hard_negative_first | 150 | 0 | 0.7867 |
| gold_4_hard_negative_last | 150 | 0 | 0.78 |
| gold_4_random_first | 150 | 0 | 0.92 |
| gold_4_random_last | 150 | 0 | 0.94 |
| gold_4_same_value_first | 107 | 43 | 0.9907 |
| gold_4_same_value_last | 107 | 43 | 0.9907 |
| gold_9_hard_negative_first | 150 | 0 | 0.8 |
| gold_9_hard_negative_last | 150 | 0 | 0.78 |
| gold_9_random_first | 148 | 2 | 0.8986 |
| gold_9_random_last | 148 | 2 | 0.9122 |
| gold_9_same_value_first | 82 | 68 | 1.0 |
| gold_9_same_value_last | 82 | 68 | 0.9878 |
| gold_only | 150 | 0 | 0.9933 |

## 4. predicted-vs-gold structure retrieval

skip — no predicted-structure retrieval output in the current tree (the table-first retrieval line, STAGE 5, was discarded and its files deleted 2026-09-15/16; recoverable only from git show 9bf1586, not restored here without a user instruction)
