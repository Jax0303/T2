# 무결성 통계 (사전 개선 진단)

## query / table / indexed-cell 수

- 전체 query: 1584 (채점 1581, 제외 3)
- 제외 사유: {'no_gold_annotation': 1, 'gold_unmappable': 1, 'unreadable_formula:=IF(F10=MIN(F5:F18),"Quebec","")': 1}
- query가 참조하는 distinct table: 538
- split corpus가 색인하는 table: 538

- value_only 색인 단위 수: 67664 (predicted_path fallback to value_only: None)
- gold_path 색인 단위 수: 67664 (predicted_path fallback to value_only: None)
- predicted_path 색인 단위 수: 67664 (predicted_path fallback to value_only: 0)

## gold 개수(m) 분포

| m | query count |
|---|---|
| 1 | 1052 |
| 2 | 187 |
| 3 | 27 |
| 4 | 38 |
| 5 | 20 |
| 6 | 43 |
| 7 | 9 |
| 8 | 27 |
| 9 | 20 |
| 10 | 10 |
| 11 | 13 |
| 12 | 14 |
| 13 | 5 |
| 14 | 6 |
| 15 | 8 |
| 16 | 4 |
| 17 | 5 |
| 18 | 3 |
| 19 | 9 |
| 20 | 3 |
| 21 | 3 |
| 22 | 1 |
| 24 | 4 |
| 25 | 1 |
| 26 | 3 |
| 27 | 1 |
| 28 | 1 |
| 30 | 2 |
| 32 | 5 |
| 34 | 1 |
| 35 | 6 |
| 36 | 1 |
| 38 | 1 |
| 39 | 1 |
| 40 | 2 |
| 41 | 1 |
| 42 | 1 |
| 43 | 6 |
| 44 | 2 |
| 48 | 1 |
| 51 | 1 |
| 54 | 2 |
| 62 | 1 |
| 64 | 1 |
| 65 | 3 |
| 76 | 1 |
| 77 | 1 |
| 84 | 5 |
| 86 | 2 |
| 90 | 1 |
| 99 | 2 |
| 104 | 1 |
| 109 | 1 |
| 116 | 1 |
| 120 | 1 |
| 124 | 2 |
| 154 | 1 |
| 156 | 1 |
| 168 | 1 |
| 186 | 2 |
| 216 | 1 |
| 264 | 1 |
| 408 | 1 |

## (mode, aggregation) 분포

| mode | aggregation | query count |
|---|---|---|
| all | none | 1029 |
| any | none | 103 |
| any | pair-argmax | 92 |
| all | div | 76 |
| all | diff | 55 |
| any | argmax | 53 |
| all | opposite | 49 |
| all | sum | 22 |
| any | pair-argmin | 21 |
| any | argmin | 19 |
| any | max | 12 |
| any | topk-argmax | 10 |
| any | greater_than | 7 |
| any | less_than | 6 |
| any | min | 4 |
| any | kth-argmax | 4 |
| all | max | 4 |
| all | range | 3 |
| all | min | 2 |
| any | counta | 2 |
| any | diff | 1 |
| all | pair-argmax | 1 |
| all | argmax | 1 |
| all | counta | 1 |
| any | opposite | 1 |
| all | kth-argmax | 1 |
| all | average | 1 |
| any | average | 1 |

## duplicate cell_id / 직렬화 텍스트 충돌 (representation별)

| representation | n_units | duplicate coords | text-collision distinct-texts | text-collision 관련 cell 수 |
|---|---|---|---|---|
| value_only | 67664 | 0 | 3378 | 60868 |
| gold_path | 67664 | 0 | 943 | 1886 |
| predicted_path | 67664 | 0 | 938 | 1876 |

## representation별 embedding cache key 충돌 여부

- 확인한 representation 수: 3
- 텍스트 집합이 동일한 representation 쌍(있으면 잠재적 충돌): []
- 각 representation의 사전 실행 embed_cache_hit: {'value_only': False, 'gold_path': True, 'predicted_path': False}


## index에 없는 gold (primary population, m=1)

- gold_path corpus에 색인되지 않은 gold cell 수: 0 / 1051

## unknown-rank 6건 원인

- 4c25190668c859cfffef1f14bd69ffe0 (table=2062, cell=[8, 1], value=62.9, indexed=True, gold_table_in_context=0): gold cell IS indexed but scored outside the top-500 hybrid-ranked units -- a genuine retrieval miss, not a corpus/indexing gap
- 7c2d4c7010f3d02a4cb5b1da1c4313a4 (table=947, cell=[0, 3], value=110800.0, indexed=True, gold_table_in_context=1): gold cell IS indexed and its table DOES appear in the delivered top-20 via a different cell, but the exact gold coordinate itself still scored outside the top-500 -- a cell-level retrieval miss, not a table-level one
- 89bab78aee8ef8065330506d7128f3f6 (table=557, cell=[43, 0], value=56.3, indexed=True, gold_table_in_context=0): gold cell IS indexed but scored outside the top-500 hybrid-ranked units -- a genuine retrieval miss, not a corpus/indexing gap
- 8db22461be5d09fe27d62d3b758fad97 (table=120_totto14023-2, cell=[1, 0], value=117.0, indexed=True, gold_table_in_context=0): gold cell IS indexed but scored outside the top-500 hybrid-ranked units -- a genuine retrieval miss, not a corpus/indexing gap
- b8b332eb07c52c9c44666293f5ea8e2b (table=2062, cell=[6, 1], value=6.1, indexed=True, gold_table_in_context=0): gold cell IS indexed but scored outside the top-500 hybrid-ranked units -- a genuine retrieval miss, not a corpus/indexing gap
- d7848bd4a92822affd40df34e647de7d (table=1852, cell=[1, 0], value=67.4, indexed=True, gold_table_in_context=0): gold cell IS indexed but scored outside the top-500 hybrid-ranked units -- a genuine retrieval miss, not a corpus/indexing gap

## table 크기 / path 길이 / 직렬화 길이 분포 (split corpus, 538 tables)

- n_rows per table: {'n': 538, 'min': 2, 'max': 62, 'mean': 17.1, 'median': 14.0}
- n_cols per table: {'n': 538, 'min': 1, 'max': 19, 'mean': 8.22, 'median': 8.0}
- row_path 길이(segment 수): {'n': 9200, 'min': 0, 'max': 4, 'mean': 2.13, 'median': 2.0}
- col_path 길이(segment 수): {'n': 4424, 'min': 1, 'max': 3, 'mean': 1.97, 'median': 2.0}

- value_only 직렬화 텍스트 길이(chars): {'n': 67664, 'min': 1, 'max': 180, 'mean': 3.55, 'median': 3.0}
- gold_path 직렬화 텍스트 길이(chars): {'n': 67664, 'min': 29, 'max': 685, 'mean': 167.8, 'median': 138.0}
- predicted_path 직렬화 텍스트 길이(chars): {'n': 67664, 'min': 25, 'max': 685, 'mean': 164.06, 'median': 137.0}