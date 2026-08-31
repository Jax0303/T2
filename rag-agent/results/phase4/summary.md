# Phase 4 Task C -- 리더 EM (sample_294)

행 769 = 검색 2조건 x 294 쿼리 + gold_cell 181. `Qwen/Qwen2.5-7B-Instruct` 4-bit, temperature=0, seed=42, max_new_tokens=32, B_reader=4096.

EM 정규화: 통화기호/퍼센트 제거 -> 천단위 콤마(`숫자,숫자`)만 제거 -> 공백 축약 + 소문자화 -> 순수 소수(`-?\d+\.\d+`)일 때만 후행 0 제거. 그 외 정규화 없음. 부호 정규화 없음.

`is_correct`는 이 스크립트에서 **재채점**한 값이다. 리더 실행 시점의 `is_correct`는 gold를 리스트 repr(`"[47.0]"`) 그대로 비교해 전건 0이었다. 모델 출력은 정상이었고 gold 파싱만 틀렸다.

## gold_answer 원소 수 분포 (쿼리 단위, pool별)

| pool | 1개 | 2개 이상 | 최대 |
|---|---:|---:|---:|
| hitab_lookup | 60 | 0 | 1 |
| hitab_arith | 57 | 3 | 2 |
| aitqa | 43 | 17 | 2 |
| rhb_fact | 57 | 3 | 2 |
| rhb_num | 53 | 1 | 2 |
| **합계** | 270 | 24 | 2 |

다중 정답 판정: 원소 **전부 일치**를 요구한다. 예측을 콤마로 나눠 정규화한 뒤 gold 원소 집합과 다중집합으로 비교하며, 순서는 무시하고 부분 일치는 인정하지 않는다.

`gold_cell` 조건은 PREREGISTER 개정 4에 따라 계산 자원 제약으로 pool당 30건으로 축소했다(`seed=42` 부분집합). `hitab_lookup`은 축소 결정 전에 60건 전량이 이미 실행돼 60을 그대로 쓴다. `hitab_arith`의 31은 재배치 이전 실행에서 1건이 먼저 기록돼 있었기 때문이다. pool별 실제 n은 아래 표에 병기했다.

## pool x 조건별 EM

| pool | dataset | P1_fixed_512 | P4_path_cell | gold_cell |
|---|---|---|---|---|
| hitab_lookup | hitab | 0.4500 (n=60) | 0.5833 (n=60) | 0.9167 (n=60) |
| hitab_arith | hitab | 0.0167 (n=60) | 0.0167 (n=60) | 0.2258 (n=31) |
| aitqa | aitqa | 0.3333 (n=60) | 0.3000 (n=60) | 0.7000 (n=30) |
| rhb_fact | realhitbench | 0.3167 (n=60) | 0.3333 (n=60) | 0.7000 (n=30) |
| rhb_num | realhitbench | 0.1296 (n=54) | 0.1296 (n=54) | 0.4000 (n=30) |

## 유형별 EM (조회 180 / 산술 114)

| query_type | P1_fixed_512 | P4_path_cell | gold_cell |
|---|---|---|---|
| lookup | 0.3667 (n=180) | 0.4056 (n=180) | 0.8083 (n=120) |
| arith | 0.0702 (n=114) | 0.0702 (n=114) | 0.3115 (n=61) |

## HiTab 내부 lookup(60) vs arith(60)

| pool | P1_fixed_512 | P4_path_cell | gold_cell |
|---|---|---|---|
| hitab_lookup | 0.4500 (n=60) | 0.5833 (n=60) | 0.9167 (n=60) |
| hitab_arith | 0.0167 (n=60) | 0.0167 (n=60) | 0.2258 (n=31) |

## gold_in_topk 군별 EM (검색 조건만)

| policy | pool | gold_in_topk=True | gold_in_topk=False |
|---|---|---|---|
| P1_fixed_512 | hitab_lookup | 0.5192 (n=52) | 0.0000 (n=8) |
| P1_fixed_512 | hitab_arith | 0.0208 (n=48) | 0.0000 (n=12) |
| P1_fixed_512 | aitqa | 0.4091 (n=44) | 0.1250 (n=16) |
| P1_fixed_512 | rhb_fact | 0.5000 (n=36) | 0.0417 (n=24) |
| P1_fixed_512 | rhb_num | 0.1739 (n=23) | 0.0968 (n=31) |
| P4_path_cell | hitab_lookup | 0.6364 (n=55) | 0.0000 (n=5) |
| P4_path_cell | hitab_arith | 0.0200 (n=50) | 0.0000 (n=10) |
| P4_path_cell | aitqa | 0.3400 (n=50) | 0.1000 (n=10) |
| P4_path_cell | rhb_fact | 0.4865 (n=37) | 0.0870 (n=23) |
| P4_path_cell | rhb_num | 0.2000 (n=20) | 0.0882 (n=34) |

## gold_cell 조건 = 리더 상한

| pool | EM |
|---|---|
| hitab_lookup | 0.9167 (n=60) |
| hitab_arith | 0.2258 (n=31) |
| aitqa | 0.7000 (n=30) |
| rhb_fact | 0.7000 (n=30) |
| rhb_num | 0.4000 (n=30) |
| **전체** | 0.6409 (n=181) |

## hit_chunk_cap (P4 상위 200 상한 도달)

| pool | 건수 |
|---|---|
| hitab_lookup | 0 |
| hitab_arith | 0 |
| aitqa | 3 |
| rhb_fact | 3 |
| rhb_num | 10 |
| **합계** | 16 |

## prompt_tokens / latency / n_chunks_used 분포

| policy | ptok mean | median | max | lat mean | median | max | n_chunks mean | median | max |
|---|---|---|---|---|---|---|---|---|---|
| P1_fixed_512 | 3751 | 3774 | 4096 | 4.66 | 2.38 | 71.40 | 7.1 | 7 | 13 |
| P4_path_cell | 4046 | 4074 | 4096 | 9.09 | 2.49 | 77.77 | 107.4 | 100 | 200 |
| gold_cell | 117 | 96 | 744 | 0.40 | 0.36 | 1.43 | 1.4 | 1 | 12 |

## 파싱 실패 (pred_parsed 비어 있음)

| policy | 건수 |
|---|---|
| P1_fixed_512 | 0 |
| P4_path_cell | 0 |
| gold_cell | 0 |
| **합계** | 0 |

`retrieved_topk`이 Excel 셀 상한(32767자)을 넘어 청크 본문을 뺀 행: **15** (본문은 `results/phase4/retrieval_294/`에 그대로 있다).
총 소요 1.14시간.
