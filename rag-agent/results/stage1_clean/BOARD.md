# 1단계 보드 — 쿼리 종류 × (표 검색, 표 조건부 셀 검색)

계측기 `analysis/stage1_board.py`. 리더 없음. 각 행의 출처 파일은 맨 아래.

| 모집단 | n | m 중앙/최대 | 표@1 | 표@3 | 셀\|표@1 | 셀\|표@3 | 셀\|표@5 | 셀\|표@10 | 셀\|표@20 | 전체@10 | 표@1×셀\|표@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hitab_dev_lookup_all_clean` | 809 | 1/1 | 0.9246 | 0.9827 | 0.7602 | 0.9048 | 0.9444 | 0.9642 | 0.9827 | 0.9468 | 0.8974 |
| `hitab_dev_lookup_multi_clean` | 31 | 2/5 | 0.9032 | 0.9677 | 0.0000 | 0.5161 | 0.6129 | 0.8387 | 0.9355 | 0.8065 | 0.7742 |
| `hitab_dev_corpus_arith_clean` | 167 | 2/12 | 0.7545 | 0.9880 | 0.1677 | 0.3413 | 0.5090 | 0.6587 | 0.7246 | 0.5509 | 0.5928 |
| `hitab_test_lookup_all_clean` | 727 | 1/1 | 0.9367 | 0.9849 | 0.7387 | 0.9010 | 0.9257 | 0.9532 | 0.9711 | 0.9381 | 0.8955 |
| `hitab_test_lookup_multi_clean` | 29 | 2/3 | 0.9310 | 0.9655 | 0.0000 | 0.5517 | 0.6552 | 0.8966 | 0.9655 | 0.8621 | 0.8621 |
| `hitab_test_corpus_arith_clean` | 179 | 2/11 | 0.8994 | 0.9832 | 0.2291 | 0.5642 | 0.6760 | 0.7877 | 0.9330 | 0.7374 | 0.7039 |

## 전체@10 실패의 위치

| 모집단 | 실패 | 표를 틀림 | 표는 맞고 셀을 틀림 |
|---|---:|---:|---:|
| `hitab_dev_lookup_all_clean` | 43 | 21 | 22 |
| `hitab_dev_lookup_multi_clean` | 6 | 2 | 4 |
| `hitab_dev_corpus_arith_clean` | 75 | 39 | 36 |
| `hitab_test_lookup_all_clean` | 45 | 13 | 32 |
| `hitab_test_lookup_multi_clean` | 4 | 1 | 3 |
| `hitab_test_corpus_arith_clean` | 47 | 8 | 39 |

## 출처

- `hitab_dev_lookup_all_clean` ← `results/stage1_clean/hitab_dev_lookup_all_S3c_page_clean_hybrid0.8_tp0.0_ranks.jsonl`
- `hitab_dev_lookup_multi_clean` ← `results/stage1_clean/hitab_dev_lookup_multi_S3c_page_clean_hybrid0.8_tp0.0_ranks.jsonl`
- `hitab_dev_corpus_arith_clean` ← `results/stage1_clean/hitab_dev_corpus_arith_S3c_page_clean_hybrid0.8_tp0.0_ranks.jsonl`
- `hitab_test_lookup_all_clean` ← `results/stage1_clean/hitab_test_lookup_all_S3c_page_clean_hybrid0.8_tp0.0_ranks.jsonl`
- `hitab_test_lookup_multi_clean` ← `results/stage1_clean/hitab_test_lookup_multi_S3c_page_clean_hybrid0.8_tp0.0_ranks.jsonl`
- `hitab_test_corpus_arith_clean` ← `results/stage1_clean/hitab_test_corpus_arith_S3c_page_clean_hybrid0.8_tp0.0_ranks.jsonl`

셀|표@k 는 **오라클** 표 게이팅이다 — 표를 맞힌다고 가정하고 그 표 안에서만 정렬한 값. 표@1×셀|표@k 는 실제 2단계 캐스케이드가 내는 값이고, 전체@k 는 현 파이프라인(코퍼스 전체 한 번 정렬)의 값이다. m>k 인 질의는 정의상 @k 에서 0이다.
