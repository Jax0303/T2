# 830건 조회 쿼리 — gold 셀을 앞지른 셀 (코퍼스 전체 랭킹)

출처 `results/lookup_gap/rank830/*_above.jsonl` (`analysis/cell_rank_dump.py --dump-above 20`).
S3c, hybrid α=0.7, 예산 없음. 랭킹 재현본은 `results/rank/`의 커밋본과 **바이트 동일**.

gold 셀이 1위가 아닌 쿼리 **402/830**. gold 위 셀 총 54418개
(쿼리당 중앙값 4, 최대 2000, rank 2000에서 절단된 쿼리 15건).

| 관계 | 셀 수 | 비율 |
|---|---:|---:|
| other_table | 50748 | 0.9326 |
| sibling_elsewhere | 1246 | 0.0229 |
| same_table_far | 1012 | 0.0186 |
| same_col | 995 | 0.0183 |
| same_row | 408 | 0.0075 |
| same_address | 9 | 0.0002 |

1위 셀(gold 대신 최상위에 온 셀)의 관계:

| 관계 | 쿼리 수 | 비율 |
|---|---:|---:|
| other_table | 128 | 0.3184 |
| same_col | 123 | 0.3060 |
| same_row | 112 | 0.2786 |
| sibling_elsewhere | 25 | 0.0622 |
| same_table_far | 11 | 0.0274 |
| same_address | 3 | 0.0075 |

gold 표 안의 셀이 하나라도 gold를 앞지른 쿼리: 362/402 (0.9005).
gold_rank 분포: 1위 밖이지만 2위 안 113, 10위 안 267, 200위 밖 41.

189건 부분집합(예산 top-200 후보 기준)의 같은 표는 `results/lookup_gap/retrieval.md`.
