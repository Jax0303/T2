# Phase 3 — HPC 층화 검색 성능

생성 2026-08-31. 계측기 `analysis/stratified_recall.py`. 리더 호출 없음.

## 측정 정의

- 검색기 `hybrid` α=0.7, 인코더 `BAAI/bge-small-en-v1.5` (CLAUDE.md §7 현행 기준).
- **Recall@k는 gold 셀 단위**다. 그 셀을 담고 있는 청크가 해당 쿼리 상위 k개 청크에
  들어갔으면 1. 건초더미는 코퍼스 전체를 그 정책으로 자른 청크 집합이다.
- 층 구분은 Phase 2의 `full_coverage`와 동일한 코드로 재계산했고, 전 조건에서
  Phase 2 CSV와 셀 단위 대조해 **불일치 0**을 확인했다 (아래 정합성 절).
- `NOT_IN_ANY_CHUNK`(토큰 창이 셀 필드를 반으로 자른 경우)는 어느 층에도 넣지 않고
  자기 행으로 따로 보고한다. 검색될 수 없으므로 Recall은 정의상 0이다.
- `P3_whole_table`은 HPC_full = 1.0 (4/4)이라 `full_coverage=False` 군이 비어
  층화가 성립하지 않는다. 사양대로 제외.

## 통계 방법 — 사양과 다른 점 (명시)

`full` 군과 `partial` 군은 **서로 다른 셀들의 서로소 집합**이라 paired bootstrap이
정의되지 않는다. 여기서는 **two-sample percentile bootstrap**을 썼다: 두 군을 각각
독립적으로 복원추출하고 차이(mean_full − mean_partial)의 2.5/97.5 분위수를 CI로 삼는다.
B=10000, seed=42는 사양 그대로다.

p-value는 (군 × 적중) 2×2 표의 **Fisher 정확검정** 양측값이다.
**모든 p는 다중비교 보정 전 값이다. 보정하지 않았다.**
이 문서에는 4 데이터셋 × 5 정책 × 2개 k = 40개의 검정이 들어 있다.

어느 한 군이라도 n < 30이면 **n<30, 해석 불가**로 표기한다.

## Recall@10

| dataset | policy | full n | full R | partial n | partial R | NOT_IN_ANY n | Δ(full−partial) | 95% CI | p (보정 전) | 판정 |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| hitab | `P1_fixed_256` | 438 | 0.9338 | 384 | 0.4167 | 8 | +0.5171 | [+0.4628, +0.5715] | 2.229e-62 |  |
| hitab | `P1_fixed_512` | 669 | 0.9297 | 157 | 0.3758 | 4 | +0.5539 | [+0.4768, +0.6318] | 2.635e-49 |  |
| hitab | `P1_fixed_1024` | 805 | 0.9255 | 25 ⚠n<30 | 0.2000 | 0 | +0.7255 | [+0.5617, +0.8731] | 1.657e-17 | **n<30, 해석 불가** |
| hitab | `P2_row` | 186 | 0.8011 | 644 | 0.7717 | 0 | +0.0293 | [-0.0370, +0.0939] | 0.4241 |  |
| hitab | `P4_path_cell` | 830 | 0.8373 | 0 ⚠n<30 | MEASURED: NO | 0 | MEASURED: NO | MEASURED: NO | MEASURED: NO | **n<30, 해석 불가** |
| multihiertt | `P1_fixed_256` | 591 | 0.7868 | 139 | 0.4029 | 7 | +0.3839 | [+0.2963, +0.4711] | 7.899e-18 |  |
| multihiertt | `P1_fixed_512` | 715 | 0.7259 | 22 ⚠n<30 | 0.3636 | 0 | +0.3622 | [+0.1517, +0.5566] | 0.000526 | **n<30, 해석 불가** |
| multihiertt | `P1_fixed_1024` | 733 | 0.7190 | 4 ⚠n<30 | 0.0000 | 0 | +0.7190 | [+0.6862, +0.7503] | 0.006458 | **n<30, 해석 불가** |
| multihiertt | `P2_row` | 391 | 0.6343 | 346 | 0.4855 | 0 | +0.1487 | [+0.0775, +0.2215] | 5.707e-05 |  |
| multihiertt | `P4_path_cell` | 737 | 0.4803 | 0 ⚠n<30 | MEASURED: NO | 0 | MEASURED: NO | MEASURED: NO | MEASURED: NO | **n<30, 해석 불가** |
| aitqa | `P1_fixed_256` | 305 | 0.7738 | 145 | 0.3793 | 1 | +0.3945 | [+0.3048, +0.4843] | 6.357e-16 |  |
| aitqa | `P1_fixed_512` | 396 | 0.7601 | 55 | 0.3455 | 0 | +0.4146 | [+0.2823, +0.5419] | 3.467e-09 |  |
| aitqa | `P1_fixed_1024` | 438 | 0.7306 | 13 ⚠n<30 | 0.3846 | 0 | +0.3460 | [+0.0672, +0.6019] | 0.01057 | **n<30, 해석 불가** |
| aitqa | `P2_row` | 189 | 0.7672 | 262 | 0.5115 | 0 | +0.2557 | [+0.1693, +0.3415] | 3.072e-08 |  |
| aitqa | `P4_path_cell` | 451 | 0.5322 | 0 ⚠n<30 | MEASURED: NO | 0 | MEASURED: NO | MEASURED: NO | MEASURED: NO | **n<30, 해석 불가** |
| realhitbench | `P1_fixed_256` | 148 | 0.7635 | 134 | 0.2687 | 5 | +0.4949 | [+0.3932, +0.5944] | 4.083e-17 |  |
| realhitbench | `P1_fixed_512` | 213 | 0.8404 | 70 | 0.2571 | 4 | +0.5832 | [+0.4645, +0.6926] | 4.887e-19 |  |
| realhitbench | `P1_fixed_1024` | 243 | 0.8889 | 42 | 0.2619 | 2 | +0.6270 | [+0.4809, +0.7625] | 1.178e-16 |  |
| realhitbench | `P2_row` | 177 | 0.3842 | 110 | 0.3636 | 0 | +0.0205 | [-0.0961, +0.1363] | 0.8023 |  |
| realhitbench | `P4_path_cell` | 287 | 0.3310 | 0 ⚠n<30 | MEASURED: NO | 0 | MEASURED: NO | MEASURED: NO | MEASURED: NO | **n<30, 해석 불가** |
| realhitbench_nomulti | `P1_fixed_256` | 92 | 0.7174 | 101 | 0.2772 | 5 | +0.4402 | [+0.3117, +0.5658] | 8.795e-10 |  |
| realhitbench_nomulti | `P1_fixed_512` | 135 | 0.8222 | 59 | 0.3051 | 4 | +0.5171 | [+0.3773, +0.6463] | 1.003e-11 |  |
| realhitbench_nomulti | `P1_fixed_1024` | 164 | 0.9024 | 32 | 0.3438 | 2 | +0.5587 | [+0.3841, +0.7271] | 6.686e-11 |  |
| realhitbench_nomulti | `P2_row` | 120 | 0.4833 | 78 | 0.4231 | 0 | +0.0603 | [-0.0833, +0.2006] | 0.4662 |  |
| realhitbench_nomulti | `P4_path_cell` | 198 | 0.3788 | 0 ⚠n<30 | MEASURED: NO | 0 | MEASURED: NO | MEASURED: NO | MEASURED: NO | **n<30, 해석 불가** |

## Recall@50

| dataset | policy | full n | full R | partial n | partial R | NOT_IN_ANY n | Δ(full−partial) | 95% CI | p (보정 전) | 판정 |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| hitab | `P1_fixed_256` | 438 | 0.9886 | 384 | 0.7240 | 8 | +0.2646 | [+0.2197, +0.3112] | 1.348e-32 |  |
| hitab | `P1_fixed_512` | 669 | 0.9836 | 157 | 0.7006 | 4 | +0.2829 | [+0.2121, +0.3579] | 9.807e-27 |  |
| hitab | `P1_fixed_1024` | 805 | 0.9789 | 25 ⚠n<30 | 0.6400 | 0 | +0.3389 | [+0.1689, +0.5339] | 9.558e-09 | **n<30, 해석 불가** |
| hitab | `P2_row` | 186 | 0.8763 | 644 | 0.8960 | 0 | -0.0196 | [-0.0736, +0.0313] | 0.4256 |  |
| hitab | `P4_path_cell` | 830 | 0.9120 | 0 ⚠n<30 | MEASURED: NO | 0 | MEASURED: NO | MEASURED: NO | MEASURED: NO | **n<30, 해석 불가** |
| multihiertt | `P1_fixed_256` | 591 | 0.9002 | 139 | 0.6331 | 7 | +0.2671 | [+0.1833, +0.3509] | 4.883e-13 |  |
| multihiertt | `P1_fixed_512` | 715 | 0.8951 | 22 ⚠n<30 | 0.5455 | 0 | +0.3497 | [+0.1378, +0.5643] | 5.054e-05 | **n<30, 해석 불가** |
| multihiertt | `P1_fixed_1024` | 733 | 0.8868 | 4 ⚠n<30 | 0.5000 | 0 | +0.3868 | [-0.1160, +0.8909] | 0.06758 | **n<30, 해석 불가** |
| multihiertt | `P2_row` | 391 | 0.7903 | 346 | 0.7428 | 0 | +0.0475 | [-0.0136, +0.1084] | 0.1375 |  |
| multihiertt | `P4_path_cell` | 737 | 0.6906 | 0 ⚠n<30 | MEASURED: NO | 0 | MEASURED: NO | MEASURED: NO | MEASURED: NO | **n<30, 해석 불가** |
| aitqa | `P1_fixed_256` | 305 | 0.9770 | 145 | 0.7655 | 1 | +0.2115 | [+0.1429, +0.2838] | 2.747e-12 |  |
| aitqa | `P1_fixed_512` | 396 | 0.9747 | 55 | 0.8000 | 0 | +0.1747 | [+0.0758, +0.2869] | 4.61e-06 |  |
| aitqa | `P1_fixed_1024` | 438 | 0.9361 | 13 ⚠n<30 | 1.0000 | 0 | -0.0639 | [-0.0868, -0.0411] | 1 | **n<30, 해석 불가** |
| aitqa | `P2_row` | 189 | 0.9048 | 262 | 0.7672 | 0 | +0.1376 | [+0.0717, +0.2037] | 0.0001437 |  |
| aitqa | `P4_path_cell` | 451 | 0.7118 | 0 ⚠n<30 | MEASURED: NO | 0 | MEASURED: NO | MEASURED: NO | MEASURED: NO | **n<30, 해석 불가** |
| realhitbench | `P1_fixed_256` | 148 | 0.8446 | 134 | 0.4328 | 5 | +0.4118 | [+0.3094, +0.5141] | 3.401e-13 |  |
| realhitbench | `P1_fixed_512` | 213 | 0.9014 | 70 | 0.4714 | 4 | +0.4300 | [+0.3063, +0.5535] | 4.837e-13 |  |
| realhitbench | `P1_fixed_1024` | 243 | 0.9465 | 42 | 0.5476 | 2 | +0.3989 | [+0.2446, +0.5509] | 2.547e-10 |  |
| realhitbench | `P2_row` | 177 | 0.6045 | 110 | 0.5545 | 0 | +0.0500 | [-0.0702, +0.1691] | 0.4599 |  |
| realhitbench | `P4_path_cell` | 287 | 0.4530 | 0 ⚠n<30 | MEASURED: NO | 0 | MEASURED: NO | MEASURED: NO | MEASURED: NO | **n<30, 해석 불가** |
| realhitbench_nomulti | `P1_fixed_256` | 92 | 0.8478 | 101 | 0.4950 | 5 | +0.3528 | [+0.2301, +0.4745] | 1.7e-07 |  |
| realhitbench_nomulti | `P1_fixed_512` | 135 | 0.9037 | 59 | 0.5085 | 4 | +0.3952 | [+0.2586, +0.5330] | 4.002e-09 |  |
| realhitbench_nomulti | `P1_fixed_1024` | 164 | 0.9634 | 32 | 0.5312 | 2 | +0.4322 | [+0.2637, +0.6075] | 1.641e-09 |  |
| realhitbench_nomulti | `P2_row` | 120 | 0.7083 | 78 | 0.6667 | 0 | +0.0417 | [-0.0897, +0.1731] | 0.5338 |  |
| realhitbench_nomulti | `P4_path_cell` | 198 | 0.5051 | 0 ⚠n<30 | MEASURED: NO | 0 | MEASURED: NO | MEASURED: NO | MEASURED: NO | **n<30, 해석 불가** |

## 청크 수 (건초더미 크기)

| dataset | `P1_fixed_256` | `P1_fixed_512` | `P1_fixed_1024` | `P2_row` | `P4_path_cell` |
|---|---|---|---|---|---|
| hitab | 1160 | 688 | 472 | 7196 | 58759 |
| multihiertt | 2007 | 1490 | 1341 | 12574 | 56745 |
| aitqa | 208 | 138 | 118 | 1315 | 5320 |
| realhitbench | 4327 | 2287 | 1280 | 20469 | 143377 |
| realhitbench_nomulti | 4327 | 2287 | 1280 | 20469 | 143377 |

## Phase 2 정합성 대조

| dataset | policy | 대조 결과 |
|---|---|---|
| hitab | `P1_fixed_256` | 830/830 대조, 불일치 0 |
| hitab | `P1_fixed_512` | 830/830 대조, 불일치 0 |
| hitab | `P1_fixed_1024` | 830/830 대조, 불일치 0 |
| hitab | `P2_row` | 830/830 대조, 불일치 0 |
| hitab | `P4_path_cell` | 830/830 대조, 불일치 0 |
| multihiertt | `P1_fixed_256` | 737/737 대조, 불일치 0 |
| multihiertt | `P1_fixed_512` | 737/737 대조, 불일치 0 |
| multihiertt | `P1_fixed_1024` | 737/737 대조, 불일치 0 |
| multihiertt | `P2_row` | 737/737 대조, 불일치 0 |
| multihiertt | `P4_path_cell` | 737/737 대조, 불일치 0 |
| aitqa | `P1_fixed_256` | 451/451 대조, 불일치 0 |
| aitqa | `P1_fixed_512` | 451/451 대조, 불일치 0 |
| aitqa | `P1_fixed_1024` | 451/451 대조, 불일치 0 |
| aitqa | `P2_row` | 451/451 대조, 불일치 0 |
| aitqa | `P4_path_cell` | 451/451 대조, 불일치 0 |
| realhitbench | `P1_fixed_256` | 287/287 대조, 불일치 0 |
| realhitbench | `P1_fixed_512` | 287/287 대조, 불일치 0 |
| realhitbench | `P1_fixed_1024` | 287/287 대조, 불일치 0 |
| realhitbench | `P2_row` | 287/287 대조, 불일치 0 |
| realhitbench | `P4_path_cell` | 287/287 대조, 불일치 0 |
| realhitbench_nomulti | `P1_fixed_256` | 생략 (multimatch 제외 조건) |
| realhitbench_nomulti | `P1_fixed_512` | 생략 (multimatch 제외 조건) |
| realhitbench_nomulti | `P1_fixed_1024` | 생략 (multimatch 제외 조건) |
| realhitbench_nomulti | `P2_row` | 생략 (multimatch 제외 조건) |
| realhitbench_nomulti | `P4_path_cell` | 생략 (multimatch 제외 조건) |

## 모집단

| dataset | 모집단 | 채점된 쿼리 | 채점된 gold 셀 | 제외 쿼리 |
|---|---|---:|---:|---:|
| hitab | `hitab_dev_lookup_all` | 830 | 830 | 0 |
| multihiertt | `multihiertt` | 399 | 737 | 0 |
| aitqa | `aitqa` | 451 | 451 | 0 |
| realhitbench | `realhitbench` | 231 | 287 | 0 |
| realhitbench_nomulti | `realhitbench` | 198 | 198 | 33 |

`realhitbench_nomulti`는 답 문자열이 gold 표 안 2개 이상 셀에 일치하는 33개 쿼리를
뺀 조건이다 (Task B). 이 조건에서는 Phase 2 CSV와 모집단이 달라 정합성 대조를 생략했다.

## 산출물

- `results/hpc/stratified/{dataset}_{policy}_stratified.json` 25개
- `results/hpc/stratified/{dataset}_{policy}_cells.csv` 25개 (셀 단위 원본)
