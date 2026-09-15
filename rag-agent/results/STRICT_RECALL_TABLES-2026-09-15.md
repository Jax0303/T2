# Strict Recall 표 — 2026-09-15 세션

이 세션에서 만든 표만 모았다. 모든 수치는 각 표의 출처 요약 JSON 에서 옮겼다(반올림만).
공통 조건: cell 단위, s3c 문장, BM25+BGE-base hybrid α=0.7. HiTab 은 질문의 정답 표 안에서 검색, MultiHiertt 는 질문의 문서 안에서 검색(헤더 규칙 v3.3, 라벨 없음). 토큰 = 리더 토크나이저로 센 문맥 블록 (HiTab Qwen2.5-7B-Instruct, MultiHiertt Qwen3-8B). DTC = Derived Table Coverage(선택 셀에서 역산한 표).

## A. Strict Recall under a 20-cell context (fixed-budget baseline)

`budget_select` 가 리더에게 넘기는 20셀 문맥을 채점한다. 예산 없는 결과가 아니다.

### A1. HiTab test

- 출처: `results/strict_fixed_budget/hitab_test_gold_cell_s3c_a0.7_fixed_budget20_summary.json`
- 지표: Strict Recall under a 20-cell context (fixed-budget baseline)
- 분할: test · 선택기: `{"function": "budget_select", "stop_after_distinct_cells": 20, "last_unit_kept_whole": true, "max_units": 0, "search_scope": "gold"}`
- 제외: 1건 {"gold_unmappable": 1}

| 그룹 | N | Strict Recall | Exact Set Match | macro P | macro R | macro F1 | micro P | micro R | micro F1 | 셀 평균 | 셀 중앙 | 셀 p95 | gold 평균 | 토큰 평균 | 토큰 중앙 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| overall | 1583 | 0.9406 | 0.0032 | 0.1153 | 0.9534 | 0.1789 | 0.1146 | 0.9365 | 0.2043 | 19.55 | 20.00 | 20.00 | 1.22 | 1147.54 | 1100.00 |
| single_cell | 1082 | 0.9649 | 0.0018 | 0.0859 | 0.9649 | 0.1384 | 0.0856 | 0.9649 | 0.1573 | 19.50 | 20.00 | 20.00 | 1.00 | 1118.38 | 1060.50 |
| multi_cell | 50 | 0.8400 | 0.0200 | 0.2603 | 0.9310 | 0.3260 | 0.2617 | 0.8800 | 0.4034 | 19.64 | 20.00 | 20.00 | 2.50 | 1104.30 | 1072.00 |
| arithmetic | 451 | 0.8936 | 0.0044 | 0.1698 | 0.9284 | 0.2598 | 0.1674 | 0.9040 | 0.2825 | 19.65 | 20.00 | 20.00 | 1.62 | 1222.29 | 1174.00 |
| header_gold | 338 | 0.9231 | 0.0148 | 0.3201 | 0.9432 | 0.4308 | 0.3199 | 0.9069 | 0.4730 | 19.55 | 20.00 | 20.00 | 1.24 | 1093.70 | 1029.50 |
| data_cell_gold_only | 1245 | 0.9454 | 0.0000 | 0.0597 | 0.9562 | 0.1105 | 0.0589 | 0.9446 | 0.1109 | 19.54 | 20.00 | 20.00 | 1.22 | 1162.16 | 1110.00 |

### A2. HiTab dev

- 출처: `results/strict_fixed_budget/hitab_dev_gold_cell_s3c_a0.7_fixed_budget20_summary.json`
- 지표: Strict Recall under a 20-cell context (fixed-budget baseline)
- 분할: dev · 선택기: `{"function": "budget_select", "stop_after_distinct_cells": 20, "last_unit_kept_whole": true, "max_units": 0, "search_scope": "gold"}`
- 제외: 0건 {}

| 그룹 | N | Strict Recall | Exact Set Match | macro P | macro R | macro F1 | micro P | micro R | micro F1 | 셀 평균 | 셀 중앙 | 셀 p95 | gold 평균 | 토큰 평균 | 토큰 중앙 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| overall | 1671 | 0.9060 | 0.0030 | 0.1145 | 0.9290 | 0.1793 | 0.1128 | 0.8926 | 0.2002 | 19.42 | 20.00 | 20.00 | 1.28 | 1114.54 | 1122.00 |
| single_cell | 1140 | 0.9500 | 0.0018 | 0.0819 | 0.9500 | 0.1333 | 0.0808 | 0.9500 | 0.1489 | 19.41 | 20.00 | 20.00 | 1.00 | 1076.26 | 1077.00 |
| multi_cell | 55 | 0.8182 | 0.0364 | 0.2068 | 0.9000 | 0.2860 | 0.2057 | 0.8921 | 0.3343 | 19.18 | 20.00 | 20.00 | 2.53 | 1161.45 | 1197.00 |
| arithmetic | 476 | 0.8109 | 0.0021 | 0.1819 | 0.8821 | 0.2770 | 0.1786 | 0.8162 | 0.2930 | 19.47 | 20.00 | 20.00 | 1.79 | 1200.81 | 1238.00 |
| header_gold | 345 | 0.9246 | 0.0145 | 0.3161 | 0.9418 | 0.4346 | 0.3142 | 0.9198 | 0.4684 | 19.57 | 20.00 | 20.00 | 1.16 | 1156.20 | 1212.00 |
| data_cell_gold_only | 1326 | 0.9012 | 0.0000 | 0.0620 | 0.9257 | 0.1129 | 0.0598 | 0.8864 | 0.1121 | 19.38 | 20.00 | 20.00 | 1.31 | 1103.71 | 1105.50 |

### A3. MultiHiertt validation

- 출처: `results/strict_fixed_budget/mh_validation_cell_hv3.3_none_doc_fixed_budget20_summary.json`
- 지표: Strict Recall under a 20-cell context (fixed-budget baseline)
- 분할: validation · 선택기: `{"function": "budget_select", "stop_after_distinct_cells": 20, "last_unit_kept_whole": true, "search_scope": "doc: tables of the question's own document"}`
- 제외: 18건 {"gold_in_header": 13, "gold_table_unparsed": 5} · 모집단 {"all_table_questions": 929, "table_only_questions": 338, "hybrid_questions": 591, "text_only_questions_excluded": 115, "no_evidence_excluded": 0, "bad_evidence_coord_excluded": 0}

| 그룹 | N | Strict Recall | Exact Set Match | macro P | macro R | macro F1 | micro P | micro R | micro F1 | 셀 평균 | 셀 중앙 | 셀 p95 | gold 평균 | 토큰 평균 | 토큰 중앙 | DTC strict | DTC P | DTC R | DTC F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| overall | 911 | 0.8496 | 0.0000 | 0.1244 | 0.9187 | 0.2119 | 0.1244 | 0.8880 | 0.2183 | 20.00 | 20.00 | 20.00 | 2.80 | 557.93 | 544.00 | 0.9737 | 0.7058 | 0.9841 | 0.7852 |
| single_cell | 75 | 0.9333 | 0.0000 | 0.0467 | 0.9333 | 0.0889 | 0.0467 | 0.9333 | 0.0889 | 20.00 | 20.00 | 20.00 | 1.00 | 546.80 | 533.00 | 1.0000 | 0.6933 | 1.0000 | 0.7862 |
| multi_cell | 123 | 0.8699 | 0.0000 | 0.1585 | 0.9302 | 0.2586 | 0.1585 | 0.8590 | 0.2677 | 20.00 | 20.00 | 20.00 | 3.69 | 585.10 | 575.00 | 1.0000 | 0.7566 | 1.0000 | 0.8290 |
| arithmetic | 713 | 0.8373 | 0.0000 | 0.1267 | 0.9152 | 0.2167 | 0.1267 | 0.8928 | 0.2219 | 20.00 | 20.00 | 20.00 | 2.84 | 554.41 | 544.00 | 0.9663 | 0.6983 | 0.9797 | 0.7775 |
| all_table_questions | 911 | 0.8496 | 0.0000 | 0.1244 | 0.9187 | 0.2119 | 0.1244 | 0.8880 | 0.2183 | 20.00 | 20.00 | 20.00 | 2.80 | 557.93 | 544.00 | 0.9737 | 0.7058 | 0.9841 | 0.7852 |
| table_only_questions | 332 | 0.8735 | 0.0000 | 0.1261 | 0.9344 | 0.2150 | 0.1261 | 0.9108 | 0.2215 | 20.00 | 20.00 | 20.00 | 2.77 | 552.90 | 526.00 | 0.9940 | 0.6635 | 0.9955 | 0.7587 |
| hybrid_questions | 579 | 0.8359 | 0.0000 | 0.1235 | 0.9097 | 0.2100 | 0.1235 | 0.8752 | 0.2164 | 20.00 | 20.00 | 20.00 | 2.82 | 560.81 | 556.00 | 0.9620 | 0.7301 | 0.9775 | 0.8003 |

## B. Strict Recall, budget-free (`strict_no_k`)

R_q = 관련성 점수(질문의 표/문서 안 min-max hybrid 점수) >= threshold 인 셀. `budget_select` 호출 0회. threshold 는 선택 분할에서 한 번 고르고 적용 분할에서 다시 조정하지 않았다.

### B1. threshold 선택 탐색 (선택 분할, 발췌)

- HiTab dev 출처: `results/strict_no_k/hitab_dev_gold_cell_s3c_a0.7_strict_no_k_summary.json` → `threshold.sweep` (101행 전체), 선택 0.94
- MH validation 출처: `results/strict_no_k/mh_validation_cell_hv3.3_none_doc_strict_no_k_summary.json` → `threshold.sweep` (101행 전체), 선택 0.91
- 규칙: grid 0.00..1.00 step 0.01 on the selection split; the threshold with the highest macro F1 wins, ties go to the higher threshold; a threshold that returns every candidate unit is refused; never selected on test

| τ | HiTab dev Strict Recall | macro F1 | 셀 평균 | 후보 셀 평균 | MH validation Strict Recall | macro F1 | 셀 평균 | 후보 셀 평균 |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 1.0000 | 0.0994 | 127.32 | 127.32 | 1.0000 | 0.0450 | 146.28 | 146.28 |
| 0.50 | 0.9503 | 0.2068 | 33.94 | 127.32 | 0.9265 | 0.2091 | 36.78 | 146.28 |
| 0.80 | 0.7947 | 0.4105 | 6.99 | 127.32 | 0.5939 | 0.4430 | 8.06 | 146.28 |
| 0.90 | 0.6709 | 0.4909 | 2.89 | 127.32 | 0.3403 | 0.4641 | 3.50 | 146.28 |
| 0.91 | — | — | — | — | 0.3216 | 0.4647 | 3.15 | 146.28 |
| 0.94 | 0.5847 | 0.5031 | 1.75 | 127.32 | — | — | — | — |

### B2. HiTab test (dev 에서 고른 τ 적용)

- 출처: `results/strict_no_k/hitab_test_gold_cell_s3c_a0.7_strict_no_k_summary.json`
- 지표: Strict Recall, budget-free: R_q = units with relevance score >= threshold
- 분할: test · 선택기: `{"function": "threshold_select", "threshold": 0.94, "budget_select_calls": 0, "search_scope": "gold"}`
- 제외: 1건 {"gold_unmappable": 1}
- threshold: 선택 분할 dev, 선택값 0.94, 적용값 0.94 (applied_from_file)

| 그룹 | N | Strict Recall | Exact Set Match | macro P | macro R | macro F1 | micro P | micro R | micro F1 | 셀 평균 | 셀 중앙 | 셀 p95 | gold 평균 | 토큰 평균 | 토큰 중앙 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| overall | 1583 | 0.5622 | 0.3626 | 0.4965 | 0.5920 | 0.5137 | 0.4421 | 0.5310 | 0.4825 | 1.65 | 1.00 | 4.00 | 1.22 | 95.79 | 68.00 |
| single_cell | 1082 | 0.7033 | 0.4769 | 0.5716 | 0.7033 | 0.6078 | 0.4904 | 0.7033 | 0.5779 | 1.59 | 1.00 | 4.00 | 1.00 | 88.77 | 66.00 |
| multi_cell | 50 | 0.1800 | 0.1200 | 0.4737 | 0.3650 | 0.3787 | 0.5368 | 0.3360 | 0.4133 | 1.90 | 1.00 | 8.00 | 2.50 | 113.56 | 65.00 |
| arithmetic | 451 | 0.2661 | 0.1153 | 0.3189 | 0.3502 | 0.3027 | 0.3258 | 0.3086 | 0.3170 | 1.76 | 1.00 | 4.00 | 1.62 | 110.65 | 82.00 |
| header_gold | 338 | 0.4586 | 0.3018 | 0.3947 | 0.4741 | 0.4109 | 0.4565 | 0.4129 | 0.4336 | 1.94 | 2.00 | 5.00 | 1.24 | 109.69 | 88.50 |
| data_cell_gold_only | 1245 | 0.5904 | 0.3791 | 0.5241 | 0.6241 | 0.5416 | 0.4373 | 0.5636 | 0.4925 | 1.57 | 1.00 | 4.00 | 1.22 | 92.01 | 67.00 |

### B3. MultiHiertt train (validation 에서 고른 τ 적용)

- 출처: `results/strict_no_k/mh_train_cell_hv3.3_none_doc_strict_no_k_summary.json`
- 지표: Strict Recall, budget-free: R_q = units with relevance score >= threshold
- 분할: train · 선택기: `{"function": "threshold_select", "threshold": 0.91, "budget_select_calls": 0, "search_scope": "doc: tables of the question's own document"}`
- 제외: 81건 {"gold_in_header": 72, "gold_table_unparsed": 6, "gold_cell_missing": 3} · 모집단 {"all_table_questions": 7117, "table_only_questions": 2908, "hybrid_questions": 4209, "text_only_questions_excluded": 713, "no_evidence_excluded": 0, "bad_evidence_coord_excluded": 0}
- threshold: 선택 분할 validation, 선택값 0.91, 적용값 0.91 (applied_from_file)

| 그룹 | N | Strict Recall | Exact Set Match | macro P | macro R | macro F1 | micro P | micro R | micro F1 | 셀 평균 | 셀 중앙 | 셀 p95 | gold 평균 | 토큰 평균 | 토큰 중앙 | DTC strict | DTC P | DTC R | DTC F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| overall | 7036 | 0.3110 | 0.1310 | 0.5153 | 0.5259 | 0.4703 | 0.4198 | 0.4762 | 0.4462 | 3.07 | 2.00 | 8.00 | 2.70 | 88.60 | 61.00 | 0.7652 | 0.8892 | 0.8478 | 0.8478 |
| single_cell | 558 | 0.6613 | 0.1344 | 0.2973 | 0.6613 | 0.3743 | 0.1992 | 0.6613 | 0.3062 | 3.32 | 3.00 | 9.00 | 1.00 | 101.75 | 71.00 | 0.9516 | 0.9023 | 0.9516 | 0.9185 |
| multi_cell | 954 | 0.5147 | 0.2516 | 0.6022 | 0.6759 | 0.5919 | 0.5169 | 0.6048 | 0.5574 | 3.90 | 3.00 | 10.00 | 3.33 | 118.16 | 85.00 | 0.9497 | 0.8981 | 0.9502 | 0.9151 |
| arithmetic | 5524 | 0.2404 | 0.1099 | 0.5224 | 0.4864 | 0.4590 | 0.4227 | 0.4428 | 0.4325 | 2.90 | 2.00 | 8.00 | 2.77 | 82.17 | 56.00 | 0.7145 | 0.8863 | 0.8196 | 0.8291 |
| all_table_questions | 7036 | 0.3110 | 0.1310 | 0.5153 | 0.5259 | 0.4703 | 0.4198 | 0.4762 | 0.4462 | 3.07 | 2.00 | 8.00 | 2.70 | 88.60 | 61.00 | 0.7652 | 0.8892 | 0.8478 | 0.8478 |
| table_only_questions | 2885 | 0.3383 | 0.1490 | 0.5156 | 0.5437 | 0.4810 | 0.4180 | 0.4986 | 0.4548 | 3.16 | 2.00 | 9.00 | 2.65 | 90.01 | 59.00 | 0.7847 | 0.8823 | 0.8582 | 0.8502 |
| hybrid_questions | 4151 | 0.2920 | 0.1185 | 0.5152 | 0.5136 | 0.4629 | 0.4211 | 0.4612 | 0.4402 | 3.00 | 2.00 | 8.00 | 2.74 | 87.62 | 61.00 | 0.7516 | 0.8940 | 0.8405 | 0.8462 |

### B4. HiTab dev (τ 선택 분할 — in-sample)

- 출처: `results/strict_no_k/hitab_dev_gold_cell_s3c_a0.7_strict_no_k_summary.json`
- 지표: Strict Recall, budget-free: R_q = units with relevance score >= threshold
- 분할: dev · 선택기: `{"function": "threshold_select", "threshold": 0.94, "budget_select_calls": 0, "search_scope": "gold"}`
- 제외: 0건 {}
- threshold: 선택 분할 dev, 선택값 0.94, 적용값 0.94 (selected_here)

| 그룹 | N | Strict Recall | Exact Set Match | macro P | macro R | macro F1 | micro P | micro R | micro F1 | 셀 평균 | 셀 중앙 | 셀 p95 | gold 평균 | 토큰 평균 | 토큰 중앙 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| overall | 1671 | 0.5847 | 0.3411 | 0.4759 | 0.6029 | 0.5031 | 0.4173 | 0.5148 | 0.4610 | 1.75 | 1.00 | 5.00 | 1.28 | 99.45 | 67.00 |
| single_cell | 1140 | 0.7018 | 0.4202 | 0.5337 | 0.7018 | 0.5785 | 0.4255 | 0.7018 | 0.5298 | 1.81 | 1.00 | 5.00 | 1.00 | 97.98 | 66.00 |
| multi_cell | 55 | 0.2727 | 0.2000 | 0.4337 | 0.3727 | 0.3760 | 0.5765 | 0.3525 | 0.4375 | 1.55 | 1.00 | 4.00 | 2.53 | 94.67 | 79.00 |
| arithmetic | 476 | 0.3403 | 0.1681 | 0.3424 | 0.3928 | 0.3372 | 0.3785 | 0.2916 | 0.3294 | 1.64 | 1.00 | 4.00 | 1.79 | 103.55 | 68.50 |
| header_gold | 345 | 0.4870 | 0.2870 | 0.3861 | 0.4918 | 0.4132 | 0.4717 | 0.4411 | 0.4559 | 1.84 | 1.00 | 5.00 | 1.16 | 108.81 | 76.00 |
| data_cell_gold_only | 1326 | 0.6101 | 0.3552 | 0.4992 | 0.6318 | 0.5264 | 0.4023 | 0.5317 | 0.4580 | 1.73 | 1.00 | 5.00 | 1.31 | 97.02 | 65.00 |

### B5. MultiHiertt validation (τ 선택 분할 — in-sample)

- 출처: `results/strict_no_k/mh_validation_cell_hv3.3_none_doc_strict_no_k_summary.json`
- 지표: Strict Recall, budget-free: R_q = units with relevance score >= threshold
- 분할: validation · 선택기: `{"function": "threshold_select", "threshold": 0.91, "budget_select_calls": 0, "search_scope": "doc: tables of the question's own document"}`
- 제외: 18건 {"gold_in_header": 13, "gold_table_unparsed": 5} · 모집단 {"all_table_questions": 929, "table_only_questions": 338, "hybrid_questions": 591, "text_only_questions_excluded": 115, "no_evidence_excluded": 0, "bad_evidence_coord_excluded": 0}
- threshold: 선택 분할 validation, 선택값 0.91, 적용값 0.91 (selected_here)

| 그룹 | N | Strict Recall | Exact Set Match | macro P | macro R | macro F1 | micro P | micro R | micro F1 | 셀 평균 | 셀 중앙 | 셀 p95 | gold 평균 | 토큰 평균 | 토큰 중앙 | DTC strict | DTC P | DTC R | DTC F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| overall | 911 | 0.3216 | 0.1164 | 0.5129 | 0.5275 | 0.4647 | 0.4135 | 0.4653 | 0.4379 | 3.15 | 2.00 | 8.00 | 2.80 | 90.48 | 61.00 | 0.8255 | 0.9071 | 0.8840 | 0.8793 |
| single_cell | 75 | 0.7467 | 0.1333 | 0.3230 | 0.7467 | 0.4109 | 0.2007 | 0.7467 | 0.3164 | 3.72 | 3.00 | 10.00 | 1.00 | 104.15 | 71.00 | 0.9333 | 0.8822 | 0.9333 | 0.8978 |
| multi_cell | 123 | 0.5203 | 0.2033 | 0.5887 | 0.6825 | 0.5849 | 0.5041 | 0.5463 | 0.5243 | 4.00 | 4.00 | 9.00 | 3.69 | 123.16 | 92.00 | 0.9593 | 0.9065 | 0.9593 | 0.9241 |
| arithmetic | 713 | 0.2426 | 0.0996 | 0.5198 | 0.4777 | 0.4497 | 0.4206 | 0.4368 | 0.4285 | 2.95 | 2.00 | 8.00 | 2.84 | 83.41 | 56.00 | 0.7910 | 0.9098 | 0.8658 | 0.8696 |
| all_table_questions | 911 | 0.3216 | 0.1164 | 0.5129 | 0.5275 | 0.4647 | 0.4135 | 0.4653 | 0.4379 | 3.15 | 2.00 | 8.00 | 2.80 | 90.48 | 61.00 | 0.8255 | 0.9071 | 0.8840 | 0.8793 |
| table_only_questions | 332 | 0.3765 | 0.1657 | 0.5344 | 0.5526 | 0.4961 | 0.4455 | 0.4940 | 0.4685 | 3.07 | 2.00 | 7.00 | 2.77 | 86.98 | 55.50 | 0.8825 | 0.8946 | 0.9096 | 0.8911 |
| hybrid_questions | 579 | 0.2902 | 0.0881 | 0.5005 | 0.5131 | 0.4467 | 0.3959 | 0.4492 | 0.4209 | 3.20 | 2.00 | 9.00 | 2.82 | 92.49 | 63.00 | 0.7927 | 0.9142 | 0.8693 | 0.8725 |
