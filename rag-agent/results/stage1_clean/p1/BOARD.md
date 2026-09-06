# 1단계 보드 — 쿼리 종류 × (표 검색, 표 조건부 셀 검색)

계측기 `analysis/stage1_board.py`. 리더 없음. 각 행의 출처 파일은 맨 아래.

| 모집단 | n | m 중앙/최대 | 표@1 | 표@3 | 셀\|표@1 | 셀\|표@3 | 셀\|표@5 | 셀\|표@10 | 셀\|표@20 | 전체@10 | 표@1×셀\|표@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hitab_dev_corpus_arith_clean` | 167 | 2/12 | 0.7784 | 0.9880 | 0.1677 | 0.4192 | 0.5928 | 0.7126 | 0.7844 | 0.6407 | 0.6407 |
| `hitab_dev_lookup_all_clean` | 809 | 1/1 | 0.9357 | 0.9839 | 0.7355 | 0.8900 | 0.9468 | 0.9629 | 0.9839 | 0.9481 | 0.9098 |
| `hitab_dev_lookup_multi_clean` | 31 | 2/5 | 0.9032 | 0.9677 | 0.0000 | 0.5161 | 0.7419 | 0.9032 | 0.9355 | 0.8065 | 0.8065 |
| `hitab_test_corpus_arith_clean` | 179 | 2/11 | 0.9162 | 0.9944 | 0.2458 | 0.6201 | 0.7542 | 0.8715 | 0.9441 | 0.8268 | 0.7933 |
| `hitab_test_lookup_all_clean` | 727 | 1/1 | 0.9395 | 0.9890 | 0.7428 | 0.9010 | 0.9312 | 0.9491 | 0.9697 | 0.9409 | 0.8927 |
| `hitab_test_lookup_multi_clean` | 29 | 2/3 | 0.9310 | 0.9655 | 0.0000 | 0.3793 | 0.6897 | 0.9310 | 1.0000 | 0.8966 | 0.8966 |

## 전체@10 실패의 위치

| 모집단 | 실패 | 표를 틀림 | 표는 맞고 셀을 틀림 |
|---|---:|---:|---:|
| `hitab_dev_corpus_arith_clean` | 60 | 32 | 28 |
| `hitab_dev_lookup_all_clean` | 42 | 18 | 24 |
| `hitab_dev_lookup_multi_clean` | 6 | 2 | 4 |
| `hitab_test_corpus_arith_clean` | 31 | 5 | 26 |
| `hitab_test_lookup_all_clean` | 43 | 9 | 34 |
| `hitab_test_lookup_multi_clean` | 3 | 1 | 2 |

## 질의 종류 간 비교 — 요구 셀당 같은 여유 `all-covered@(c·m)`

**사후 지표** (사전등록 주지표는 전체@10 그대로). k 를 셀 수로 고정하면 정답 셀이 1개인 조회와 최대 12개인 산술이 같은 시험을 보지 않는다. 예산을 요구량 m 에 비례시켜 정답 셀 1개당 c 칸을 준다. c=10 은 m=1 인 조회에서 @10 과 **정의상 동일**하므로(아래 표에서 확인) 사전등록 조회 숫자를 바꾸지 않는 유일한 c 다 — 결과를 보고 고른 상수가 아니다. c≥1 에서는 m>k 로 불가능한 질의가 없다.

| 모집단 | n | m 최대 | c=1 | c=3 | c=5 | c=10 | (참고) 전체@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hitab_dev_corpus_arith_clean` | 167 | 12 | 0.2994 | 0.5389 | 0.7126 | 0.7665 | 0.6407 |
| `hitab_dev_lookup_all_clean` | 809 | 1 | 0.6959 | 0.8591 | 0.9197 | 0.9481 | 0.9481 |
| `hitab_dev_lookup_multi_clean` | 31 | 5 | 0.3226 | 0.7742 | 0.8387 | 0.8710 | 0.8065 |
| `hitab_test_corpus_arith_clean` | 179 | 11 | 0.4581 | 0.7709 | 0.8492 | 0.9218 | 0.8268 |
| `hitab_test_lookup_all_clean` | 727 | 1 | 0.7043 | 0.8721 | 0.9161 | 0.9409 | 0.9409 |
| `hitab_test_lookup_multi_clean` | 29 | 3 | 0.3448 | 0.8621 | 0.8966 | 0.9655 | 0.8966 |

## 전체@10 을 m 으로 층화

같은 지표를 요구 셀 수로 쪼갠 것. 새 지표가 아니다.

| 모집단 | m=1 (n) | m=2 (n) | m≥3 (n) |
|---|---:|---:|---:|
| `hitab_dev_corpus_arith_clean` | 0.9429 (35) | 0.6979 (96) | 0.1944 (36) |
| `hitab_dev_lookup_all_clean` | 0.9481 (809) | — | — |
| `hitab_dev_lookup_multi_clean` | — | 0.8696 (23) | 0.6250 (8) |
| `hitab_test_corpus_arith_clean` | 0.9038 (52) | 0.8421 (114) | 0.3846 (13) |
| `hitab_test_lookup_all_clean` | 0.9409 (727) | — | — |
| `hitab_test_lookup_multi_clean` | — | 0.9130 (23) | 0.8333 (6) |

## 각 k 에서 `m>k` 라 정의상 0 인 건수

분모는 옮기지 않는다 (모집단은 감사에서 고정). 세어서 밝히기만 한다. 이 수가 큰 칸의 @k 는 검색기가 아니라 정의를 재고 있다.

| 모집단 | @1 | @3 | @5 | @10 | @20 | @50 |
|---|---:|---:|---:|---:|---:|---:|
| `hitab_dev_corpus_arith_clean` | **132** | 20 | 13 | 5 | 0 | 0 |
| `hitab_dev_lookup_all_clean` | 0 | 0 | 0 | 0 | 0 | 0 |
| `hitab_dev_lookup_multi_clean` | **31** | 3 | 0 | 0 | 0 | 0 |
| `hitab_test_corpus_arith_clean` | **127** | 10 | 7 | 1 | 0 | 0 |
| `hitab_test_lookup_all_clean` | 0 | 0 | 0 | 0 | 0 | 0 |
| `hitab_test_lookup_multi_clean` | **29** | 0 | 0 | 0 | 0 | 0 |

## 출처

- `hitab_dev_corpus_arith_clean` ← `results/stage1_clean/p1/hitab_dev_corpus_arith_S3c_page_clean_hybrid0.8_tp0.0_ranks.jsonl`
- `hitab_dev_lookup_all_clean` ← `results/stage1_clean/p1/hitab_dev_lookup_all_S3c_page_clean_hybrid0.8_tp0.0_ranks.jsonl`
- `hitab_dev_lookup_multi_clean` ← `results/stage1_clean/p1/hitab_dev_lookup_multi_S3c_page_clean_hybrid0.8_tp0.0_ranks.jsonl`
- `hitab_test_corpus_arith_clean` ← `results/stage1_clean/p1/hitab_test_corpus_arith_S3c_page_clean_hybrid0.8_tp0.0_ranks.jsonl`
- `hitab_test_lookup_all_clean` ← `results/stage1_clean/p1/hitab_test_lookup_all_S3c_page_clean_hybrid0.8_tp0.0_ranks.jsonl`
- `hitab_test_lookup_multi_clean` ← `results/stage1_clean/p1/hitab_test_lookup_multi_S3c_page_clean_hybrid0.8_tp0.0_ranks.jsonl`

셀|표@k 는 **오라클** 표 게이팅이다 — 표를 맞힌다고 가정하고 그 표 안에서만 정렬한 값. 표@1×셀|표@k 는 실제 2단계 캐스케이드가 내는 값이고, 전체@k 는 현 파이프라인(코퍼스 전체 한 번 정렬)의 값이다. m>k 인 질의는 정의상 @k 에서 0이다.
