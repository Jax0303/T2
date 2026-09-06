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

## 출처

- `hitab_dev_corpus_arith_clean` ← `results/stage1_clean/p1/hitab_dev_corpus_arith_S3c_page_clean_hybrid0.8_tp0.0_ranks.jsonl`
- `hitab_dev_lookup_all_clean` ← `results/stage1_clean/p1/hitab_dev_lookup_all_S3c_page_clean_hybrid0.8_tp0.0_ranks.jsonl`
- `hitab_dev_lookup_multi_clean` ← `results/stage1_clean/p1/hitab_dev_lookup_multi_S3c_page_clean_hybrid0.8_tp0.0_ranks.jsonl`
- `hitab_test_corpus_arith_clean` ← `results/stage1_clean/p1/hitab_test_corpus_arith_S3c_page_clean_hybrid0.8_tp0.0_ranks.jsonl`
- `hitab_test_lookup_all_clean` ← `results/stage1_clean/p1/hitab_test_lookup_all_S3c_page_clean_hybrid0.8_tp0.0_ranks.jsonl`
- `hitab_test_lookup_multi_clean` ← `results/stage1_clean/p1/hitab_test_lookup_multi_S3c_page_clean_hybrid0.8_tp0.0_ranks.jsonl`

셀|표@k 는 **오라클** 표 게이팅이다 — 표를 맞힌다고 가정하고 그 표 안에서만 정렬한 값. 표@1×셀|표@k 는 실제 2단계 캐스케이드가 내는 값이고, 전체@k 는 현 파이프라인(코퍼스 전체 한 번 정렬)의 값이다. m>k 인 질의는 정의상 @k 에서 0이다.
