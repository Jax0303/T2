#!/usr/bin/env bash
# 4·5단계 dev 후속(vLLM) — PREREG-2026-10-15-prompt-sweep.md. 인자: <P* 변형>
# (5) P* 를 최대 1,536 토큰으로 → dev_cap1536_qwen3_5_4b ; (4) P* 생각 모드 최대 2,048 → dev_think_qwen3_5_4b (참고)
set -uo pipefail
cd "$(dirname "$0")/../.."
V="$1"; D=results/prompt_sweep_20261015; M=Qwen/Qwen3.5-4B
ENV="CUDA_HOME=$HOME/vllm-venv/lib/python3.12/site-packages/nvidia/cu13 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
echo "[start] $(date -Is) $V cap1536"
env $ENV ~/vllm-venv/bin/python scripts/prompt_sweep.py --engine vllm --model $M --gpu-mem 0.89 --max-model-len 6144 \
  --records $D/dev_C_records.jsonl,$D/dev_A_records.jsonl --arm C,A --variants "$V" --out-dir $D/dev_cap1536_qwen3_5_4b --max-new-tokens 1536 --resume
echo "[exit $?] $(date -Is)"
echo "[start] $(date -Is) $V thinking"
env $ENV ~/vllm-venv/bin/python scripts/prompt_sweep.py --engine vllm --model $M --gpu-mem 0.89 --max-model-len 6144 \
  --records $D/dev_C_records.jsonl,$D/dev_A_records.jsonl --arm C,A --variants "$V" --out-dir $D/dev_think_qwen3_5_4b --max-new-tokens 2048 --thinking --resume
echo "[exit $?] $(date -Is)"
echo "[all done] $(date -Is)"
