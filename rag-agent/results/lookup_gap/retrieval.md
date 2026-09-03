# 조회 쿼리 검색 정확도 (Recall 아님)

출처 `results/phase4/retrieval_{294,ext129}/*_hitab_lookup.jsonl` (후보 top-200).
계측기 `analysis/lookup_retrieval.py`. 모델 실행 없음.
830건 코퍼스 전체 랭킹 보드는 `results/rank/STAGE1_BOARD.md`.

## P4_path_cell

| 지표 | 값 |
|---|---:|
| n | 189 |
| R@1 | 0.6085 |
| R@5 | 0.7831 |
| R@10 | 0.8466 |
| R@50 | 0.9153 |
| R@200 | 0.9471 |
| MRR | 0.6918 |
| gold_in_top200 | 0.9471 |
| median_rank_when_found | 0 |
| chunks_entering_context_mean | 70.58 |
| n_queries_gold_not_rank0 | 64 |
| n_cells_above_gold | 1067 |
| n_cells_above_gold_that_entered_context | 878 |

gold 셀보다 위에 있던 셀의 관계 (전 쿼리 합산):

| 관계 | 셀 수 | 비율 |
|---|---:|---:|
| other_table | 710 | 0.6654 |
| same_col | 128 | 0.1200 |
| sibling_elsewhere | 122 | 0.1143 |
| same_row | 67 | 0.0628 |
| unparsed | 29 | 0.0272 |
| same_table_far | 11 | 0.0103 |

## P1_fixed_512

| 지표 | 값 |
|---|---:|
| n | 189 |
| R@1 | 0.5714 |
| R@5 | 0.7619 |
| R@10 | 0.8307 |
| R@50 | 0.9259 |
| R@200 | 0.9788 |
| MRR | 0.6536 |
| gold_in_top200 | 0.9788 |
| median_rank_when_found | 0 |
| chunks_entering_context_mean | 7.56 |

`R@k`의 분모는 쿼리 189건, 분자는 gold 셀이 후보 상위 k 안에 든 쿼리 수다.
greedy fill 컨텍스트 안 포함률(= 지금까지 보고한 Recall)은 P4 0.9312 / P1 0.7989로
따로이며, 후보 순위와 예산 절단은 다른 축이다.
P1은 청크가 512토큰 블록이라 gold 셀 위 청크의 헤더 관계를 정의할 수 없다.
