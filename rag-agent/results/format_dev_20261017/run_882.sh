#!/usr/bin/env bash
# 882 확인: D(채운 부분 표)·C·A × P*, vLLM — 1단계 최대 1,536 토큰(오전 882 와 같은 조건), 2단계 최대 3,072 토큰. PREREG-2026-10-17-delivery-format-882.md
set -uo pipefail
cd "$(dirname "$0")/../.."
D=results/format_dev_20261017; CF=results/cellform_20261012; M=Qwen/Qwen3.5-4B; V=R1X1F1B2D1C1E0
ENV="CUDA_HOME=$HOME/vllm-venv/lib/python3.12/site-packages/nvidia/cu13 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
echo "[start stage1 cap1536] $(date -Is)"
env $ENV ~/vllm-venv/bin/python scripts/prompt_sweep.py --engine vllm --model $M --gpu-mem 0.89 --max-model-len 6144 \
  --records $D/D882_records.jsonl,$CF/C_ours_budget_records.jsonl,$CF/A_chunk_md_records.jsonl --arm D,C,A \
  --variants "$V" --out-dir $D/final882_qwen3_5_4b --max-new-tokens 1536 --resume
echo "[exit $?] $(date -Is)"
env PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python $D/score_882.py $D/final882_qwen3_5_4b | tee $D/score_882_cap1536.txt
echo "[start stage2 cap3072] $(date -Is)"
env $ENV ~/vllm-venv/bin/python scripts/prompt_sweep.py --engine vllm --model $M --gpu-mem 0.89 --max-model-len 7168 --max-num-seqs 24 \
  --records $D/D882_records.jsonl,$CF/C_ours_budget_records.jsonl,$CF/A_chunk_md_records.jsonl --arm D,C,A \
  --variants "$V" --out-dir $D/final882_cap3072_qwen3_5_4b --max-new-tokens 3072 --resume
echo "[exit $?] $(date -Is)"
env PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python $D/score_882.py $D/final882_cap3072_qwen3_5_4b | tee $D/score_882_cap3072.txt
echo "[all done] $(date -Is)"
