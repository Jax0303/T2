# gold_cell 조건 오답 15건 — 리더 상한을 막는 것

출처 `results/phase4/reader_records.jsonl` (policy=gold_cell) + S3c 셀 문장 재생성.
계측기 `analysis/ceiling_cases.py`. 모델 실행 없음.
컨텍스트에 gold 셀 문장 하나만 있는 조건이다 (EM 0.9206, n=189).

| 분류 | n |
|---|---:|
| other_reader_error | 7 |
| sign | 5 |
| complement_100_minus_x | 2 |
| percent_scale | 1 |

| gold | 셀 문장의 값 | 모델 답 | 분류 |
|---|---|---|---|
| [6] | -6.0 | -6.0 | sign |
| [13.8] | 13.8 | 86.2 | complement_100_minus_x |
| [0.9] | -0.9 | -0.9 | sign |
| [0.9] | -0.9 | -0.9 | sign |
| [3.4] | 3.4 | 2.8 | other_reader_error |
| [3.6] | 3.6 | 0.8 | other_reader_error |
| [1.2] | -1.2 | -1.2 | sign |
| [145408.0] | 145408 | 147934.68 | other_reader_error |
| [3.1] | 3.1 | 2.8 | other_reader_error |
| [46.1] | -46.1 | -46.1 | sign |
| [14.9] | 14.9 | 85.1 | complement_100_minus_x |
| [9849051.0] | 9849051 | 523.1 | other_reader_error |
| [3.9] | 3.9 | 1.0 | other_reader_error |
| [0.165] | 0.165 | 16.5% | percent_scale |
| [82.2] | 82.2 | 81.3 | other_reader_error |

`sign` = 모델이 셀 값을 그대로 옮겼고 gold만 부호를 뺐다.
`percent_scale` = 셀 0.165를 16.5%로 바꿔 답했다.
`complement_100_minus_x` = 셀 값의 100 − x를 답했다.
`other_reader_error` = 셀 문장 하나만 주고도 다른 수를 답했다.
