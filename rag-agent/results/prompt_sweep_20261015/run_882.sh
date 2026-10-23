#!/usr/bin/env bash
# 882 확인(vLLM) — PREREG-2026-10-15-prompt-sweep.md. 인자: <모델 hf id> <변형(쉼표, C·A 둘 다)> [추가 인자…]
set -uo pipefail
cd "$(dirname "$0")/../.."
M="$1"; V="$2"; shift 2; D=results/prompt_sweep_20261015
echo "[start] $(date -Is) $M $V"
CUDA_HOME=$HOME/vllm-venv/lib/python3.12/site-packages/nvidia/cu13 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN \
PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 ~/vllm-venv/bin/python scripts/prompt_sweep.py --engine vllm --model "$M" \
  --records results/cellform_20261012/C_ours_budget_records.jsonl,results/cellform_20261012/A_chunk_md_records.jsonl --arm C,A \
  --variants "$V" --out-dir $D/final_882 --resume "$@"
echo "[exit $?] $(date -Is)"
