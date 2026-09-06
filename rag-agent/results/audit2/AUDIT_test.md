# 데이터셋 감사 — test (page 제목)

계측기 `analysis/dataset_audit.py`. gold 는 `answer_formulas` 기준. 규칙은 스크립트 머리.

## `hitab_test_lookup_all` — 769 → **728** 통과 (제외 41)

| 규칙 | 건수 |
|---|---:|
| A | 29 |
| F | 12 |

수식 gold 가 기존(pooled) gold 와 다른 통과 질의: 1
m 분포(통과): {0: 1, 1: 727}

## `hitab_test_lookup_multi` — 31 → **29** 통과 (제외 2)

| 규칙 | 건수 |
|---|---:|
| A | 1 |
| V | 1 |

수식 gold 가 기존(pooled) gold 와 다른 통과 질의: 0
m 분포(통과): {2: 23, 3: 6}

## `hitab_test_corpus_arith` — 183 → **180** 통과 (제외 3)

| 규칙 | 건수 |
|---|---:|
| A | 2 |
| V | 1 |

수식 gold 가 기존(pooled) gold 와 다른 통과 질의: 1
m 분포(통과): {0: 1, 1: 52, 2: 114, 3: 3, 4: 1, 5: 2, 6: 4, 8: 2, 11: 1}

