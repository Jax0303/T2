# 데이터셋 감사 — dev (page 제목)

계측기 `analysis/dataset_audit.py`. gold 는 `answer_formulas` 기준. 규칙은 스크립트 머리.

## `hitab_dev_lookup_all` — 830 → **809** 통과 (제외 21)

| 규칙 | 건수 |
|---|---:|
| A | 12 |
| F | 7 |
| Q | 2 |

수식 gold 가 기존(pooled) gold 와 다른 통과 질의: 0
m 분포(통과): {1: 809}

## `hitab_dev_lookup_multi` — 33 → **31** 통과 (제외 2)

| 규칙 | 건수 |
|---|---:|
| F | 2 |

수식 gold 가 기존(pooled) gold 와 다른 통과 질의: 0
m 분포(통과): {2: 23, 3: 5, 4: 2, 5: 1}

## `hitab_dev_corpus_arith` — 175 → **167** 통과 (제외 8)

| 규칙 | 건수 |
|---|---:|
| A | 6 |
| F | 2 |

수식 gold 가 기존(pooled) gold 와 다른 통과 질의: 0
m 분포(통과): {1: 35, 2: 96, 3: 16, 4: 5, 5: 2, 6: 2, 7: 3, 8: 1, 10: 2, 11: 1, 12: 4}

