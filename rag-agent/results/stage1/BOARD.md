# 1단계 보드 — 쿼리 종류 × (표 검색, 표 조건부 셀 검색)

계측기 `analysis/stage1_board.py`. 리더 없음. 각 행의 출처 파일은 맨 아래.

| 모집단 | n | m 중앙/최대 | 표@1 | 표@3 | 셀\|표@1 | 셀\|표@3 | 셀\|표@5 | 셀\|표@10 | 셀\|표@20 | 전체@10 | 표@1×셀\|표@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hitab_dev_lookup_all` | 830 | 1/1 | 0.9193 | 0.9819 | 0.7566 | 0.9036 | 0.9446 | 0.9639 | 0.9819 | 0.9446 | 0.8916 |
| `hitab_dev_lookup_multi` | 33 | 2/5 | 0.9091 | 0.9697 | 0.0000 | 0.4848 | 0.6364 | 0.8485 | 0.9394 | 0.8182 | 0.7879 |
| `hitab_dev_corpus_arith` | 175 | 2/12 | 0.7543 | 0.9886 | 0.1600 | 0.3257 | 0.4914 | 0.6343 | 0.7086 | 0.5314 | 0.5714 |
| `hitab_test_lookup_all` | 769 | 1/1 | 0.9324 | 0.9844 | 0.7178 | 0.8830 | 0.9090 | 0.9415 | 0.9649 | 0.9246 | 0.8804 |
| `hitab_test_lookup_multi` | 31 | 2/3 | 0.9355 | 0.9677 | 0.0000 | 0.5161 | 0.6129 | 0.9032 | 0.9677 | 0.8387 | 0.8710 |
| `hitab_test_corpus_arith` | 183 | 2/11 | 0.8907 | 0.9781 | 0.2240 | 0.5519 | 0.6667 | 0.7760 | 0.9180 | 0.7213 | 0.6885 |

## 전체@10 실패의 위치

| 모집단 | 실패 | 표를 틀림 | 표는 맞고 셀을 틀림 |
|---|---:|---:|---:|
| `hitab_dev_lookup_all` | 46 | 23 | 23 |
| `hitab_dev_lookup_multi` | 6 | 2 | 4 |
| `hitab_dev_corpus_arith` | 82 | 41 | 41 |
| `hitab_test_lookup_all` | 58 | 16 | 42 |
| `hitab_test_lookup_multi` | 5 | 1 | 4 |
| `hitab_test_corpus_arith` | 51 | 10 | 41 |

## 출처

- `hitab_dev_lookup_all` ← `results/p0_confirm/hitab_dev_lookup_all_S3c_page_hybrid0.8_tp0.0_ranks.jsonl`
- `hitab_dev_lookup_multi` ← `results/p0_confirm/hitab_dev_lookup_multi_S3c_page_hybrid0.8_tp0.0_ranks.jsonl`
- `hitab_dev_corpus_arith` ← `results/stage1/hitab_dev_corpus_arith_S3c_page_hybrid0.8_tp0.0_ranks.jsonl`
- `hitab_test_lookup_all` ← `results/p0_confirm/hitab_test_lookup_all_S3c_page_hybrid0.8_tp0.0_ranks.jsonl`
- `hitab_test_lookup_multi` ← `results/p0_confirm/hitab_test_lookup_multi_S3c_page_hybrid0.8_tp0.0_ranks.jsonl`
- `hitab_test_corpus_arith` ← `results/stage1/hitab_test_corpus_arith_S3c_page_hybrid0.8_tp0.0_ranks.jsonl`

셀|표@k 는 **오라클** 표 게이팅이다 — 표를 맞힌다고 가정하고 그 표 안에서만 정렬한 값. 표@1×셀|표@k 는 실제 2단계 캐스케이드가 내는 값이고, 전체@k 는 현 파이프라인(코퍼스 전체 한 번 정렬)의 값이다. m>k 인 질의는 정의상 @k 에서 0이다.
