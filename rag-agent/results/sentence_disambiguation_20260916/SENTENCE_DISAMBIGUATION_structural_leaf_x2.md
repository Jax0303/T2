# 문장 축-구분 개입 (structural_leaf_x2) — 2026-09-17

모집단: hitab test primary (mode=all, m=1, aggregation=none), query count=991. encoder/hybrid(alpha=0.7)/corpus(split)는 baseline과 동일 — 바뀐 변수는 cell 문장 템플릿 하나뿐.

## R@1/5/10/20/MRR — baseline .. structural_leaf_x2

| repr | R@1 | R@5 | R@10 | R@20 | MRR | median rank |
|---|---|---|---|---|---|---|
| gold_path (s3c, baseline) | 0.5752 | 0.7982 | 0.8698 | 0.9142 | 0.6773 | 1 |
| predicted_path (s3c, baseline) | 0.553 | 0.7891 | 0.8668 | 0.9082 | 0.6644 | 1 |
| structural_leaf (one leaf-prefix repeat) | 0.6176 | 0.8032 | 0.8789 | 0.9213 | 0.7049 | 1 |
| structural_leaf_x2 (new) | 0.6145 | 0.7972 | 0.8749 | 0.9162 | 0.7025 | 1 |

## top-1 오류 클래스 — baseline(gold_path, s3c) .. structural_leaf_x2

| class | gold_path (s3c, baseline) n | gold_path (s3c, baseline) % | structural_leaf (one leaf-prefix repeat) n | structural_leaf (one leaf-prefix repeat) % | structural_leaf_x2 (new) n | structural_leaf_x2 (new) % |
|---|---|---|---|---|---|---|
| same_value | 22 | 0.0523 | 20 | 0.0528 | 19 | 0.0497 |
| wrong_row | 113 | 0.2684 | 91 | 0.2401 | 98 | 0.2565 |
| wrong_column | 143 | 0.3397 | 126 | 0.3325 | 125 | 0.3272 |
| same_leaf_header | 15 | 0.0356 | 18 | 0.0475 | 18 | 0.0471 |
| nearby_cell | 11 | 0.0261 | 6 | 0.0158 | 6 | 0.0157 |
| wrong_table | 102 | 0.2423 | 98 | 0.2586 | 98 | 0.2565 |
| other | 15 | 0.0356 | 20 | 0.0528 | 18 | 0.0471 |
| **total errors (non-exact-match)** | 421 | 1.0 | 379 | 1.0 | 382 | 1.0 |
