# Operand-targeted RAG — finqa

seed=42 · n=300 (operand-bearing=238) · split=validation · dense=True

## 1. Decomposition ceiling (header_path_match_accuracy)

| matcher | accuracy |
|---|---|
| fuzzy | 0.4943 |
| embedding | 0.5716 |
| hybrid | 0.5437 |

## 2. operand_recall@k

| serialization | mode | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|---|
| S1(flat) | plain | 0.585 | 0.9182 | 0.9622 | 1.0 |
| S1(flat) | operand | 0.7396 | 0.9314 | 0.9734 | 1.0 |
| S1(flat) | oracle | 0.9034 | 1.0 | 1.0 | 1.0 |
| S2(header-path) | plain | 0.5955 | 0.9196 | 0.9713 | 1.0 |
| S2(header-path) | operand | 0.7438 | 0.9335 | 0.9762 | 1.0 |
| S2(header-path) | oracle | 0.9034 | 1.0 | 1.0 | 1.0 |

## 3. Coverage + fallback

- mean coverage_rate: **0.875**
- fallback rate: **0.1471**
- operand_recall@5: no-fallback 0.9713 → +fallback **0.9832**
- histogram: {'0.0': 16, '0.2': 4, '0.5': 15, '0.8': 13, '1.0': 190}
