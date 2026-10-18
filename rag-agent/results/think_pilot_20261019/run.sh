#!/usr/bin/env bash
# 생각 모드 타당성 측정(채택 아님): dev 앞 60문항, 팔 D(채운 표)·A(LangChain), P*, 생각 모드, 최대 8,192 토큰.
set -uo pipefail
cd "$(dirname "$0")/../.."
D=results/format_dev_20261017; SW=results/prompt_sweep_20261015; O=results/think_pilot_20261019; M=Qwen/Qwen3.5-4B; V=R1X1F1B2D1C1E0
ENV="CUDA_HOME=$HOME/vllm-venv/lib/python3.12/site-packages/nvidia/cu13 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
echo "[start] $(date -Is)"
env $ENV ~/vllm-venv/bin/python scripts/prompt_sweep.py --engine vllm --model $M --gpu-mem 0.89 --max-model-len 12288 --max-num-seqs 16 \
  --records $D/D_records.jsonl,$SW/dev_A_records.jsonl --arm D,A --variants "$V" --out-dir $O/think8192 --max-new-tokens 8192 --thinking \
  --ids results/determinism_20261018/ids60.json --resume
echo "[exit $?] $(date -Is)"
echo "[all done] $(date -Is)"
