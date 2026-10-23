#!/usr/bin/env bash
# 1단계 모델 선택 — PREREG-2026-10-15-prompt-sweep.md. 인자: <모델 hf id> <출력 이름> [추가 인자…]
set -uo pipefail
cd "$(dirname "$0")/../.."
M="$1"; N="$2"; shift 2; D=results/prompt_sweep_20261015
echo "[start] $(date -Is) $M"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python scripts/prompt_sweep.py --model "$M" \
  --records $D/dev_C_records.jsonl,$D/dev_A_records.jsonl --arm C,A --variants "R0X0F0B0D0C0E0,R0X1F1B1D1C2E1" \
  --out-dir $D/stage0_$N --resume --check-batch 10 "$@"
echo "[exit $?] $(date -Is)"
