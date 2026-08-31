# hitab_lookup: origin of the wrong value (bucket B)

## P4_path_cell — n=38

| origin | n |
|---|---:|
| same_row_wrong_col | 13 |
| not_in_context | 10 |
| same_col_wrong_row | 9 |
| other_table | 5 |
| same_table_elsewhere | 1 |

gold_rank == 0: 15
gold chunk in context whose sentence does NOT carry the gold answer: 3


## P1_fixed_512 — n=58

| origin | n |
|---|---:|
| in_context | 52 |
| not_in_context | 6 |

gold_rank == 0: 37
gold chunk in context whose sentence does NOT carry the gold answer: N/A (512-token block, no single asserted value)
row/column attribution is N/A for P1: a matched block holds many cells.

