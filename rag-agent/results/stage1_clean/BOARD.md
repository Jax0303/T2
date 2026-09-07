# 1단계 보드 — 쿼리 종류 × (표 검색, 표 조건부 셀 검색)

계측기 `analysis/stage1_board.py`. 리더 없음. 각 행의 출처 파일은 맨 아래.

**주지표는 `전체@20`** — 고정 예산이고 오라클이 아니다. K 는 리더에 넘기는 셀 개수이며 모든 질의에 같은 값을 준다. K=20 인 이유는 사전 논거다: gold 셀 수 m 의 최대가 12 이므로 K<12 면 정의상 못 맞히는 질의가 생긴다 (아래 불가능 건수 표). 20 은 등록된 사다리 (1,3,5,10,20,50) 에서 12 이상의 가장 낮은 칸이다.

⚠️ **`전체@10` 은 `PREREG-2026-09-06-cell90.md` 의 사전등록 값이고 그 판정은 그대로 남는다** (산술 목표 .9 → test .8268 / dev .6407 미달). @20 이 더 높다고 "목표 달성"이라고 쓰지 말 것 — 못 넘긴 것을 보고 K 를 올린 것이 된다. **@20 에 대한 판정은 아직 돌리지 않은 실험의 사전등록에서 목표값을 먼저 정한 뒤에 한다.**

| 모집단 | n | m 중앙/최대 | **전체@20** | 전체@10 | 셀recall@20 | 표@1 | 표@3 | 셀\|표@1 | 셀\|표@3 | 셀\|표@5 | 셀\|표@10 | 셀\|표@20 | 표@1×셀\|표@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hitab_dev_corpus_arith_clean` | 167 | 2/12 | **0.6527** | 0.5509 | 0.6729 | 0.7545 | 0.9880 | 0.1677 | 0.3413 | 0.5090 | 0.6587 | 0.7246 | 0.5928 |
| `hitab_dev_lookup_all_clean` | 809 | 1/1 | **0.9604** | 0.9468 | 0.9604 | 0.9246 | 0.9827 | 0.7602 | 0.9048 | 0.9444 | 0.9642 | 0.9827 | 0.8974 |
| `hitab_dev_lookup_multi_clean` | 31 | 2/5 | **0.8710** | 0.8065 | 0.9189 | 0.9032 | 0.9677 | 0.0000 | 0.5161 | 0.6129 | 0.8387 | 0.9355 | 0.7742 |
| `hitab_test_corpus_arith_clean` | 179 | 2/11 | **0.8771** | 0.7374 | 0.8927 | 0.8994 | 0.9832 | 0.2291 | 0.5642 | 0.6760 | 0.7877 | 0.9330 | 0.7039 |
| `hitab_test_lookup_all_clean` | 727 | 1/1 | **0.9574** | 0.9381 | 0.9574 | 0.9367 | 0.9849 | 0.7387 | 0.9010 | 0.9257 | 0.9532 | 0.9711 | 0.8955 |
| `hitab_test_lookup_multi_clean` | 29 | 2/3 | **0.9310** | 0.8621 | 0.9375 | 0.9310 | 0.9655 | 0.0000 | 0.5517 | 0.6552 | 0.8966 | 0.9655 | 0.8621 |

`셀recall@20` 은 gold 셀 단위 회수율(부분 점수)이다 — `전체@20` 은 하나만 놓쳐도 0 이라 "얼마나 가까웠나"가 안 보인다. `@1`·`@3` 열은 **조회형(m=1)에만 의미가 있다** — 산술·다중은 m>k 라 정의상 0 인 질의가 대부분이다 (아래 표).

## 전체@10 실패의 위치

| 모집단 | 실패 | 표를 틀림 | 표는 맞고 셀을 틀림 |
|---|---:|---:|---:|
| `hitab_dev_corpus_arith_clean` | 75 | 39 | 36 |
| `hitab_dev_lookup_all_clean` | 43 | 21 | 22 |
| `hitab_dev_lookup_multi_clean` | 6 | 2 | 4 |
| `hitab_test_corpus_arith_clean` | 47 | 8 | 39 |
| `hitab_test_lookup_all_clean` | 45 | 13 | 32 |
| `hitab_test_lookup_multi_clean` | 4 | 1 | 3 |

## 질의 종류 간 비교 — 요구 셀당 같은 여유 `all-covered@(c·m)`

**사후 지표** (사전등록 주지표는 전체@10 그대로). k 를 셀 수로 고정하면 정답 셀이 1개인 조회와 최대 12개인 산술이 같은 시험을 보지 않는다. 예산을 요구량 m 에 비례시켜 정답 셀 1개당 c 칸을 준다. c=10 은 m=1 인 조회에서 @10 과 **정의상 동일**하므로(아래 표에서 확인) 사전등록 조회 숫자를 바꾸지 않는 유일한 c 다 — 결과를 보고 고른 상수가 아니다. c≥1 에서는 m>k 로 불가능한 질의가 없다.

⚠️ **오라클 예산이다 — 논문 표에 싣지 말 것.** k=c·m 을 정하려면 gold 셀 개수 m 을 알아야 하는데 배포된 시스템은 m 을 모른다 (셀|표@k 와 같은 종류의 오라클). **진단 전용**이다: "예산만 공평하면 산술이 조회만큼 하는가"를 알아낸 것으로 역할이 끝났고, 같은 문제를 오라클 없이 푸는 것이 위의 고정 예산 `전체@20` 이다. 답변 성능의 대리 지표로도 전체@10 보다 나쁘다 (`VERDICT_METRIC.md` §3 한계 절).

| 모집단 | n | m 최대 | c=1 | c=3 | c=5 | c=10 | (참고) 전체@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hitab_dev_corpus_arith_clean` | 167 | 12 | 0.2515 | 0.4611 | 0.6108 | 0.7006 | 0.5509 |
| `hitab_dev_lookup_all_clean` | 809 | 1 | 0.7281 | 0.8714 | 0.9209 | 0.9468 | 0.9468 |
| `hitab_dev_lookup_multi_clean` | 31 | 5 | 0.2903 | 0.7097 | 0.8065 | 0.9032 | 0.8065 |
| `hitab_test_corpus_arith_clean` | 179 | 11 | 0.3520 | 0.6983 | 0.7486 | 0.8883 | 0.7374 |
| `hitab_test_lookup_all_clean` | 727 | 1 | 0.6933 | 0.8735 | 0.9065 | 0.9381 | 0.9381 |
| `hitab_test_lookup_multi_clean` | 29 | 3 | 0.4138 | 0.7931 | 0.8966 | 0.9310 | 0.8621 |

## 전체@10 을 m 으로 층화

같은 지표를 요구 셀 수로 쪼갠 것. 새 지표가 아니다.

| 모집단 | m=1 (n) | m=2 (n) | m≥3 (n) |
|---|---:|---:|---:|
| `hitab_dev_corpus_arith_clean` | 0.9429 (35) | 0.5312 (96) | 0.2222 (36) |
| `hitab_dev_lookup_all_clean` | 0.9468 (809) | — | — |
| `hitab_dev_lookup_multi_clean` | — | 0.8261 (23) | 0.7500 (8) |
| `hitab_test_corpus_arith_clean` | 0.9231 (52) | 0.6930 (114) | 0.3846 (13) |
| `hitab_test_lookup_all_clean` | 0.9381 (727) | — | — |
| `hitab_test_lookup_multi_clean` | — | 0.9130 (23) | 0.6667 (6) |

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

- `hitab_dev_corpus_arith_clean` ← `results/stage1_clean/hitab_dev_corpus_arith_S3c_page_clean_hybrid0.8_tp0.0_ranks.jsonl`
- `hitab_dev_lookup_all_clean` ← `results/stage1_clean/hitab_dev_lookup_all_S3c_page_clean_hybrid0.8_tp0.0_ranks.jsonl`
- `hitab_dev_lookup_multi_clean` ← `results/stage1_clean/hitab_dev_lookup_multi_S3c_page_clean_hybrid0.8_tp0.0_ranks.jsonl`
- `hitab_test_corpus_arith_clean` ← `results/stage1_clean/hitab_test_corpus_arith_S3c_page_clean_hybrid0.8_tp0.0_ranks.jsonl`
- `hitab_test_lookup_all_clean` ← `results/stage1_clean/hitab_test_lookup_all_S3c_page_clean_hybrid0.8_tp0.0_ranks.jsonl`
- `hitab_test_lookup_multi_clean` ← `results/stage1_clean/hitab_test_lookup_multi_S3c_page_clean_hybrid0.8_tp0.0_ranks.jsonl`

셀|표@k 는 **오라클** 표 게이팅이다 — 표를 맞힌다고 가정하고 그 표 안에서만 정렬한 값. 표@1×셀|표@k 는 실제 2단계 캐스케이드가 내는 값이고, 전체@k 는 현 파이프라인(코퍼스 전체 한 번 정렬)의 값이다. m>k 인 질의는 정의상 @k 에서 0이다.
