# Operand-targeted RAG — hitab

seed=42 · n=300 (operand-bearing=241) · split=dev · dense=True

## 1. Decomposition ceiling (header_path_match_accuracy)

| matcher | accuracy |
|---|---|
| fuzzy | 0.3029 |
| embedding | 0.4855 |
| hybrid | 0.409 |

## 2. operand_recall@k

| serialization | mode | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|---|
| S1(flat) | plain | 0.5299 | 0.7067 | 0.7972 | 0.8966 |
| S1(flat) | operand | 0.6159 | 0.7067 | 0.7903 | 0.8854 |
| S1(flat) | oracle | 0.7654 | 0.8667 | 0.9355 | 0.9855 |
| S2(header-path) | plain | 0.6934 | 0.8387 | 0.9053 | 0.952 |
| S2(header-path) | operand | 0.6947 | 0.7863 | 0.83 | 0.9005 |
| S2(header-path) | oracle | 0.896 | 0.9566 | 0.9742 | 0.9938 |

## 3. Coverage + fallback

- mean coverage_rate: **0.8485**
- fallback rate: **0.1577**
- operand_recall@5: no-fallback 0.8362 → +fallback **0.8676**
- histogram: {'0.0': 28, '0.2': 4, '0.5': 6, '0.8': 10, '1.0': 193}
