# hitab_lookup bucket B, 세부 (P4_path_cell)

출처 `results/lookup_gap/distractor.json`. 계측기 `analysis/lookup_b_detail.py`.
모델 재실행 없음.

## 1. 이웃 셀에서 값을 가져온 28건 — 틀린 헤더의 축

| 틀린 헤더 | n |
|---|---:|
| col_header | 12 |
| row_header | 11 |
| both | 3 |
| gold_cell_itself | 1 |
| duplicate_address | 1 |

축별 경로 차이 (`equal` = 그 축은 정답과 동일):

| 축 | equal | sibling | prefix | other |
|---|---:|---:|---:|---:|
| row | 14 | 6 | 1 | 7 |
| col | 13 | 8 | 0 | 7 |

origin × 틀린 헤더:

| origin | 틀린 헤더 | n |
|---|---|---:|
| same_row_wrong_col | col_header | 12 |
| same_col_wrong_row | row_header | 9 |
| other_table | row_header | 2 |
| other_table | both | 2 |
| same_table_elsewhere | both | 1 |
| same_row_wrong_col | gold_cell_itself | 1 |
| other_table | duplicate_address | 1 |

## 2. 컨텍스트에 없는 값 10건

| gold | pred | pred/gold | gold의 배율 | 질문 본문의 수 | 컨텍스트 헤더·제목의 수 |
|---|---|---:|---|---|---:|
| [17718556.0] | 17718.556 | 0.001 | /1000 | 아니오 | 0 |
| [51.5] | 32.7 | 0.634951 | 아니오 | 아니오 | 0 |
| [2.8] | 16.3 | 5.821429 | 아니오 | 아니오 | 0 |
| [13.4] | 17.5 | 1.30597 | 아니오 | 아니오 | 0 |
| [7] | 2.5% | 0.357143 | 아니오 | 아니오 | 0 |
| [13.5] | 46.1 | 3.414815 | 아니오 | 아니오 | 0 |
| [25.0] | 3.0 | 0.12 | 아니오 | 아니오 | 0 |
| [70.0] | 33 | 0.471429 | 아니오 | 아니오 | 0 |
| [101123.0] | 96500 | 0.954283 | 아니오 | 아니오 | 0 |
| [12632.0] | 325.719 | 0.025785 | 아니오 | 아니오 | 14 |

배율로 설명되는 건: 1/10. 질문 본문의 수를 그대로 답한 건: 0/10. 컨텍스트의 헤더·제목에 그 수가 있던 건: 1/10. (셀 값으로는 정의상 0건 — 그래서 이 버킷이다.)
