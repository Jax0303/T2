# 3단계 보드 — 리더 사다리 (단일 셀 → 다중 셀 → 산술)

계측기 `analysis/stage3_board.py`, 원본 `analysis/retrieved_answer_em.py` 출력.
리더 Qwen2.5-7B-Instruct 4-bit NF4 temp=0 seed=42 max_new_tokens=32, 채점 `phase4_summary.em`.
arm `models/bge-base-cell-ft-p0` α=0.8 `--title-mode page`. 사전 기록 `PREREG-2026-09-05-reader-ladder.md`.

## `e4_arith_dev_value`  (n=167, 출처 `results/stage3/e4_arith_dev_value.jsonl`)

| 조건 | EM | EM(표기허용) | gold 전부 주입 | EM\|주입됨 | EM\|안됨 | ptok 중앙 | vs top10 (앞만:뒤만, p) |
|---|---:|---:|---:|---:|---:|---:|---|
| `gold` | 0.3892 | 0.6707 | 1.0000 | 0.3892 | — | 156 | 41:6, p=1.77e-07 |
| `top10` | 0.1796 | 0.3413 | 0.6407 | 0.2617 | 0.0333 | 623 |  |

## `e4_arith_dev_cot`  (n=167, 출처 `results/stage3/e4_arith_dev_cot.jsonl`)

| 조건 | EM | EM(표기허용) | gold 전부 주입 | EM\|주입됨 | EM\|안됨 | ptok 중앙 | vs top10 (앞만:뒤만, p) |
|---|---:|---:|---:|---:|---:|---:|---|
| `gold` | 0.3713 | 0.7605 | 1.0000 | 0.3713 | — | 180 | 25:6, p=0.000878 |
| `top10` | 0.2575 | 0.4431 | 0.6407 | 0.3738 | 0.0500 | 647 |  |

## `e4_arith_test_value`  (n=179, 출처 `results/stage3/e4_arith_test_value.jsonl`)

| 조건 | EM | EM(표기허용) | gold 전부 주입 | EM\|주입됨 | EM\|안됨 | ptok 중앙 | vs top10 (앞만:뒤만, p) |
|---|---:|---:|---:|---:|---:|---:|---|
| `gold` | 0.5587 | 0.7654 | 1.0000 | 0.5587 | — | 168 | 70:4, p=1.29e-16 |
| `top10` | 0.1899 | 0.4022 | 0.8268 | 0.2297 | 0.0000 | 677 |  |

## `e4_arith_test_cot`  (n=179, 출처 `results/stage3/e4_arith_test_cot.jsonl`)

| 조건 | EM | EM(표기허용) | gold 전부 주입 | EM\|주입됨 | EM\|안됨 | ptok 중앙 | vs top10 (앞만:뒤만, p) |
|---|---:|---:|---:|---:|---:|---:|---|
| `gold` | 0.5363 | 0.7821 | 1.0000 | 0.5363 | — | 192 | 48:4, p=1.31e-10 |
| `top10` | 0.2905 | 0.4693 | 0.8268 | 0.3514 | 0.0000 | 701 |  |

## `e4_single_dev200_cot`  (n=200, 출처 `results/stage3/e4_single_dev200_cot.jsonl`)

| 조건 | EM | EM(표기허용) | gold 전부 주입 | EM\|주입됨 | EM\|안됨 | ptok 중앙 | vs top10 (앞만:뒤만, p) |
|---|---:|---:|---:|---:|---:|---:|---|
| `gold` | 0.8300 | 0.8750 | 1.0000 | 0.8300 | — | 132 | 61:9, p=1.28e-10 |
| `top10` | 0.5700 | 0.5950 | 0.9250 | 0.6108 | 0.0667 | 653 |  |

