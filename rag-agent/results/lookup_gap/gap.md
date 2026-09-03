# hitab_lookup EM gap (re-scored from results/phase4/reader_records.jsonl)

queries 189, single-gold-cell 189

| policy | n | em | recall_all_gold_in_context | n_wrong | A_retrieval_miss | B_distractor | C_reader_limit |
|---|---|---|---|---|---|---|---|
| P1_fixed_512 | 189 | 0.4444 | 0.7989 | 105 | 37 | 58 | 10 |
| P4_path_cell | 189 | 0.672 | 0.9312 | 62 | 12 | 38 | 12 |
| gold_cell | 189 | 0.9206 | 1.0 | 15 | 0 | 0 | 15 |

B_distractor under P4_path_cell: 38 queries, listed in gap.json.
