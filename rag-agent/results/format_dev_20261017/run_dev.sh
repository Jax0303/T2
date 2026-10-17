#!/usr/bin/env bash
# 전달 형식 dev 395 × 6 팔 × P*(R1X1F1B2D1C1E0), vLLM, 최대 1,536 토큰 — PREREG-2026-10-17-delivery-format.md
set -uo pipefail
cd "$(dirname "$0")/../.."
D=results/format_dev_20261017; SW=results/prompt_sweep_20261015; M=Qwen/Qwen3.5-4B; V=R1X1F1B2D1C1E0
ENV="CUDA_HOME=$HOME/vllm-venv/lib/python3.12/site-packages/nvidia/cu13 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
echo "[start] $(date -Is)"
env $ENV ~/vllm-venv/bin/python scripts/prompt_sweep.py --engine vllm --model $M --gpu-mem 0.89 --max-model-len 6144 \
  --records $SW/dev_C_records.jsonl,$D/S0_records.jsonl,$D/G_records.jsonl,$D/S_records.jsonl,$D/D_records.jsonl,$SW/dev_A_records.jsonl --arm C,S0,G,S,D,A \
  --variants "$V" --out-dir $D/dev_qwen3_5_4b --max-new-tokens 1536 --resume
echo "[exit $?] $(date -Is)"
env PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python $D/score_formats.py $D/dev_qwen3_5_4b | tee $D/score.txt
echo "[all done] $(date -Is)"
