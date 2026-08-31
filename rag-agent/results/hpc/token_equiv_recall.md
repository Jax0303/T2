# Phase 3b — 토큰 등가 Recall

생성 2026-08-31. 계측기 `analysis/token_equiv_recall.py`. 리더 호출 없음.
검색기 `hybrid` α=0.7, 인코더 `BAAI/bge-small-en-v1.5`, 셀 스킴 `S3c`.

## 측정 정의

- **평균 청크 토큰**: 코퍼스 전체 청크 집합에 대해 `Budget(BAAI/bge-small-en-v1.5)`로
  실제로 센 값이다. 추정하지 않았다. 중앙값과 총 토큰도 같이 싣는다.
- **등가 k** = `floor(B / 평균청크토큰)`, 최소 1. 실비용 = k × 평균청크토큰.
- **Recall**은 gold 셀 단위, **층화 없음**. 그 셀을 담은 청크가 상위 k에 들면 1.
  어느 청크에도 안 들어간 셀(`NOT_IN_ANY_CHUNK`)도 분모에 포함되며 정의상 0이다.
- 헤더 자체인 셀(header_path 빈 셀)은 Phase 2와 같은 규칙으로 제외한다.

## 1. 평균 청크 토큰 수 (측정값)

| dataset | policy | 청크 수 | 평균 토큰 | 중앙값 | 총 토큰 |
|---|---|---:|---:|---:|---:|
| hitab | `P1_fixed_256` | 1160 | 208.23 | 256.0 | 241551 |
| hitab | `P1_fixed_512` | 688 | 351.09 | 387.5 | 241548 |
| hitab | `P1_fixed_1024` | 472 | 511.75 | 471.5 | 241548 |
| hitab | `P2_row` | 7196 | 59.48 | 56.0 | 428004 |
| hitab | `P4_path_cell` | 58759 | 38.19 | 30.0 | 2243732 |
| multihiertt | `P1_fixed_256` | 2007 | 175.50 | 189.0 | 352233 |
| multihiertt | `P1_fixed_512` | 1490 | 236.40 | 196.0 | 352232 |
| multihiertt | `P1_fixed_1024` | 1341 | 262.66 | 192.0 | 352231 |
| multihiertt | `P2_row` | 12574 | 45.81 | 42.0 | 575986 |
| multihiertt | `P4_path_cell` | 56745 | 19.19 | 18.0 | 1088742 |
| aitqa | `P1_fixed_256` | 208 | 198.10 | 231.0 | 41204 |
| aitqa | `P1_fixed_512` | 138 | 298.58 | 266.5 | 41204 |
| aitqa | `P1_fixed_1024` | 118 | 349.16 | 266.5 | 41201 |
| aitqa | `P2_row` | 1315 | 45.28 | 40.0 | 59542 |
| aitqa | `P4_path_cell` | 5320 | 19.36 | 19.0 | 103017 |
| realhitbench | `P1_fixed_256` | 4327 | 240.08 | 256.0 | 1038845 |
| realhitbench | `P1_fixed_512` | 2287 | 454.26 | 512.0 | 1038890 |
| realhitbench | `P1_fixed_1024` | 1280 | 811.65 | 1024.0 | 1038908 |
| realhitbench | `P2_row` | 20469 | 65.44 | 51.0 | 1339545 |
| realhitbench | `P4_path_cell` | 143377 | 30.83 | 25.0 | 4419840 |

## 2. 등가 k = floor(B / 평균청크토큰)

| dataset | policy | 평균 토큰 | k(B=2560) | 실비용 | k(B=5120) | 실비용 | k(B=10240) | 실비용 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| hitab | `P1_fixed_256` | 208.23 | 12 | 2499 | 24 | 4998 | 49 | 10203 |
| hitab | `P1_fixed_512` | 351.09 | 7 | 2458 | 14 | 4915 | 29 | 10182 |
| hitab | `P1_fixed_1024` | 511.75 | 5 | 2559 | 10 | 5118 | 20 | 10235 |
| hitab | `P2_row` | 59.48 | 43 | 2558 | 86 | 5115 | 172 | 10230 |
| hitab | `P4_path_cell` | 38.19 | 67 | 2558 | 134 | 5117 | 268 | 10234 |
| multihiertt | `P1_fixed_256` | 175.50 | 14 | 2457 | 29 | 5090 | 58 | 10179 |
| multihiertt | `P1_fixed_512` | 236.40 | 10 | 2364 | 21 | 4964 | 43 | 10165 |
| multihiertt | `P1_fixed_1024` | 262.66 | 9 | 2364 | 19 | 4991 | 38 | 9981 |
| multihiertt | `P2_row` | 45.81 | 55 | 2519 | 111 | 5085 | 223 | 10215 |
| multihiertt | `P4_path_cell` | 19.19 | 133 | 2552 | 266 | 5104 | 533 | 10226 |
| aitqa | `P1_fixed_256` | 198.10 | 12 | 2377 | 25 | 4952 | 51 | 10103 |
| aitqa | `P1_fixed_512` | 298.58 | 8 | 2389 | 17 | 5076 | 34 | 10152 |
| aitqa | `P1_fixed_1024` | 349.16 | 7 | 2444 | 14 | 4888 | 29 | 10126 |
| aitqa | `P2_row` | 45.28 | 56 | 2536 | 113 | 5117 | 226 | 10233 |
| aitqa | `P4_path_cell` | 19.36 | 132 | 2556 | 264 | 5112 | 528 | 10224 |
| realhitbench | `P1_fixed_256` | 240.08 | 10 | 2401 | 21 | 5042 | 42 | 10084 |
| realhitbench | `P1_fixed_512` | 454.26 | 5 | 2271 | 11 | 4997 | 22 | 9994 |
| realhitbench | `P1_fixed_1024` | 811.65 | 3 | 2435 | 6 | 4870 | 12 | 9740 |
| realhitbench | `P2_row` | 65.44 | 39 | 2552 | 78 | 5105 | 156 | 10209 |
| realhitbench | `P4_path_cell` | 30.83 | 83 | 2559 | 166 | 5117 | 332 | 10234 |

## 3. 등가 k에서의 Recall (층화 없음, 전체 gold 셀)

### B = 2560 토큰

| dataset | policy | k | Recall | gold 셀 n | NOT_IN_ANY_CHUNK |
|---|---|---:|---:|---:|---:|
| hitab | `P1_fixed_256` | 12 | 0.6976 | 830 | 8 |
| hitab | `P1_fixed_512` | 7 | 0.7819 | 830 | 4 |
| hitab | `P1_fixed_1024` | 5 | 0.8506 | 830 | 0 |
| hitab | `P2_row` | 43 | 0.8880 | 830 | 0 |
| hitab | `P4_path_cell` | 67 | 0.9205 | 830 | 0 |
| multihiertt | `P1_fixed_256` | 14 | 0.7408 | 737 | 7 |
| multihiertt | `P1_fixed_512` | 10 | 0.7151 | 737 | 0 |
| multihiertt | `P1_fixed_1024` | 9 | 0.7042 | 737 | 0 |
| multihiertt | `P2_row` | 55 | 0.7707 | 737 | 0 |
| multihiertt | `P4_path_cell` | 133 | 0.8046 | 737 | 0 |
| aitqa | `P1_fixed_256` | 12 | 0.6696 | 451 | 1 |
| aitqa | `P1_fixed_512` | 8 | 0.6475 | 451 | 0 |
| aitqa | `P1_fixed_1024` | 7 | 0.6364 | 451 | 0 |
| aitqa | `P2_row` | 56 | 0.8337 | 451 | 0 |
| aitqa | `P4_path_cell` | 132 | 0.8315 | 451 | 0 |
| realhitbench | `P1_fixed_256` | 10 | 0.5192 | 287 | 5 |
| realhitbench | `P1_fixed_512` | 5 | 0.6307 | 287 | 4 |
| realhitbench | `P1_fixed_1024` | 3 | 0.6272 | 287 | 2 |
| realhitbench | `P2_row` | 39 | 0.5401 | 287 | 0 |
| realhitbench | `P4_path_cell` | 83 | 0.4983 | 287 | 0 |

### B = 5120 토큰

| dataset | policy | k | Recall | gold 셀 n | NOT_IN_ANY_CHUNK |
|---|---|---:|---:|---:|---:|
| hitab | `P1_fixed_256` | 24 | 0.7843 | 830 | 8 |
| hitab | `P1_fixed_512` | 14 | 0.8337 | 830 | 4 |
| hitab | `P1_fixed_1024` | 10 | 0.9036 | 830 | 0 |
| hitab | `P2_row` | 86 | 0.9157 | 830 | 0 |
| hitab | `P4_path_cell` | 134 | 0.9434 | 830 | 0 |
| multihiertt | `P1_fixed_256` | 29 | 0.7965 | 737 | 7 |
| multihiertt | `P1_fixed_512` | 21 | 0.7897 | 737 | 0 |
| multihiertt | `P1_fixed_1024` | 19 | 0.7870 | 737 | 0 |
| multihiertt | `P2_row` | 111 | 0.8182 | 737 | 0 |
| multihiertt | `P4_path_cell` | 266 | 0.8535 | 737 | 0 |
| aitqa | `P1_fixed_256` | 25 | 0.8160 | 451 | 1 |
| aitqa | `P1_fixed_512` | 17 | 0.8204 | 451 | 0 |
| aitqa | `P1_fixed_1024` | 14 | 0.7894 | 451 | 0 |
| aitqa | `P2_row` | 113 | 0.9024 | 451 | 0 |
| aitqa | `P4_path_cell` | 264 | 0.8914 | 451 | 0 |
| realhitbench | `P1_fixed_256` | 21 | 0.5889 | 287 | 5 |
| realhitbench | `P1_fixed_512` | 11 | 0.6899 | 287 | 4 |
| realhitbench | `P1_fixed_1024` | 6 | 0.7422 | 287 | 2 |
| realhitbench | `P2_row` | 78 | 0.6341 | 287 | 0 |
| realhitbench | `P4_path_cell` | 166 | 0.5645 | 287 | 0 |

### B = 10240 토큰

| dataset | policy | k | Recall | gold 셀 n | NOT_IN_ANY_CHUNK |
|---|---|---:|---:|---:|---:|
| hitab | `P1_fixed_256` | 49 | 0.8542 | 830 | 8 |
| hitab | `P1_fixed_512` | 29 | 0.8819 | 830 | 4 |
| hitab | `P1_fixed_1024` | 20 | 0.9301 | 830 | 0 |
| hitab | `P2_row` | 172 | 0.9578 | 830 | 0 |
| hitab | `P4_path_cell` | 268 | 0.9542 | 830 | 0 |
| multihiertt | `P1_fixed_256` | 58 | 0.8480 | 737 | 7 |
| multihiertt | `P1_fixed_512` | 43 | 0.8697 | 737 | 0 |
| multihiertt | `P1_fixed_1024` | 38 | 0.8562 | 737 | 0 |
| multihiertt | `P2_row` | 223 | 0.8643 | 737 | 0 |
| multihiertt | `P4_path_cell` | 533 | 0.8806 | 737 | 0 |
| aitqa | `P1_fixed_256` | 51 | 0.9091 | 451 | 1 |
| aitqa | `P1_fixed_512` | 34 | 0.9180 | 451 | 0 |
| aitqa | `P1_fixed_1024` | 29 | 0.8825 | 451 | 0 |
| aitqa | `P2_row` | 226 | 0.9424 | 451 | 0 |
| aitqa | `P4_path_cell` | 528 | 0.9357 | 451 | 0 |
| realhitbench | `P1_fixed_256` | 42 | 0.6167 | 287 | 5 |
| realhitbench | `P1_fixed_512` | 22 | 0.7387 | 287 | 4 |
| realhitbench | `P1_fixed_1024` | 12 | 0.8014 | 287 | 2 |
| realhitbench | `P2_row` | 156 | 0.7247 | 287 | 0 |
| realhitbench | `P4_path_cell` | 332 | 0.6481 | 287 | 0 |

### 정책별 가로 비교 (Recall)

| dataset | B | `P1_fixed_256` | `P1_fixed_512` | `P1_fixed_1024` | `P2_row` | `P4_path_cell` | 최고 |
|---|---:|---:|---:|---:|---:|---:|---|
| hitab | 2560 | 0.6976 | 0.7819 | 0.8506 | 0.8880 | 0.9205 | `P4_path_cell` |
| hitab | 5120 | 0.7843 | 0.8337 | 0.9036 | 0.9157 | 0.9434 | `P4_path_cell` |
| hitab | 10240 | 0.8542 | 0.8819 | 0.9301 | 0.9578 | 0.9542 | `P2_row` |
| multihiertt | 2560 | 0.7408 | 0.7151 | 0.7042 | 0.7707 | 0.8046 | `P4_path_cell` |
| multihiertt | 5120 | 0.7965 | 0.7897 | 0.7870 | 0.8182 | 0.8535 | `P4_path_cell` |
| multihiertt | 10240 | 0.8480 | 0.8697 | 0.8562 | 0.8643 | 0.8806 | `P4_path_cell` |
| aitqa | 2560 | 0.6696 | 0.6475 | 0.6364 | 0.8337 | 0.8315 | `P2_row` |
| aitqa | 5120 | 0.8160 | 0.8204 | 0.7894 | 0.9024 | 0.8914 | `P2_row` |
| aitqa | 10240 | 0.9091 | 0.9180 | 0.8825 | 0.9424 | 0.9357 | `P2_row` |
| realhitbench | 2560 | 0.5192 | 0.6307 | 0.6272 | 0.5401 | 0.4983 | `P1_fixed_512` |
| realhitbench | 5120 | 0.5889 | 0.6899 | 0.7422 | 0.6341 | 0.5645 | `P1_fixed_1024` |
| realhitbench | 10240 | 0.6167 | 0.7387 | 0.8014 | 0.7247 | 0.6481 | `P1_fixed_1024` |

## 4. RealHiTBench 표당 셀 수 분포

`P4_path_cell` 청크 143,377개의 출처 확인.

| 항목 | 값 |
|---|---:|
| 표 | 537 |
| 셀이 하나라도 색인된 표 | 532 |
| 색인 셀 합계 | 143377 |
| min | 6 |
| p25 | 72 |
| median | 179 |
| p75 | 333 |
| p90 | 648 |
| p95 | 897 |
| p99 | 1500 |
| max | 2156 |
| mean | 269.5 |

| 셀 수 구간 | 표 수 | 셀 합 | 코퍼스 셀 비중 |
|---|---:|---:|---:|
| 0–99 | 186 | 10271 | 0.0716 |
| 100–499 | 263 | 64274 | 0.4483 |
| 500–999 | 70 | 48971 | 0.3416 |
| 1000–1999 | 10 | 13526 | 0.0943 |
| 2000–4999 | 3 | 6335 | 0.0442 |
| 5000–9999 | 0 | 0 | 0.0000 |
| 10000+ | 0 | 0 | 0.0000 |

**셀 1만개 이상인 표: 0개.** 비정상 표 없음.
최대 표 `business-table15` 2,156셀 (87x28 격자의 88.5%).
143,377은 532개 표 × 평균 269.5셀이다.
gold 셀을 가진 표 179개 중 1만 셀 이상은 0개, 해당 gold 셀 0개.

| 상위 표 | 셀 | 격자 | 비어있지 않은 비율 |
|---|---:|---|---:|
| `business-table15` | 2156 | 87x28 | 0.885 |
| `employment-table03` | 2112 | 279x8 | 0.946 |
| `transport-table08` | 2067 | 145x17 | 0.839 |
| `employment-table19` | 1908 | 331x6 | 0.961 |
| `religion-table05` | 1570 | 96x24 | 0.681 |
| `education-table20` | 1500 | 50x31 | 0.968 |
| `military-table07` | 1412 | 441x22 | 0.146 |
| `employment-table18` | 1368 | 127x12 | 0.898 |
| `health-table48` | 1245 | 91x33 | 0.415 |
| `education-table17` | 1196 | 52x24 | 0.958 |

## 산출물

- `results/hpc/token_equiv/{dataset}_{policy}_tokeq.json` 20개
- `results/hpc/token_equiv/{dataset}_{policy}_cells.csv` 20개 (셀 단위 rank 원본)
