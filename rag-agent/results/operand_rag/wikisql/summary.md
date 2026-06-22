# Operand-targeted RAG — wikisql

seed=42 · n=300 (operand-bearing=300) · split=validation · dense=True

## 1. Decomposition ceiling (header_path_match_accuracy)

| matcher | accuracy |
|---|---|
| fuzzy | 0.7413 |
| embedding | 0.7112 |
| hybrid | 0.7874 |

## 2. operand_recall@k

| serialization | mode | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|---|
| S1(flat) | plain | 0.7555 | 0.9031 | 0.9657 | 0.9881 |
| S1(flat) | operand | 0.2679 | 0.5128 | 0.6908 | 0.8692 |
| S1(flat) | oracle | 0.1971 | 0.4081 | 0.589 | 0.8414 |
| S2(header-path) | plain | 0.7555 | 0.9031 | 0.9657 | 0.9881 |
| S2(header-path) | operand | 0.2679 | 0.5128 | 0.6908 | 0.8692 |
| S2(header-path) | oracle | 0.1971 | 0.4081 | 0.589 | 0.8414 |

## 3. Coverage + fallback

- mean coverage_rate: **0.5042**
- fallback rate: **0.7133**
- operand_recall@5: no-fallback 0.5257 → +fallback **0.8647**
- histogram: {'0.0': 22, '0.2': 71, '0.5': 121, '0.8': 52, '1.0': 34}
