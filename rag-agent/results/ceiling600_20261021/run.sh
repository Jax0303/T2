#!/usr/bin/env bash
# 전달 단계 상한 진단 O1·O3 × P*, 새 문항 600, 4,096 토큰, 1회(동시 32) — PREREG-2026-10-21-ceiling-600.md
set -uo pipefail
cd "$(dirname "$0")/../.."
O=results/ceiling600_20261021; M=Qwen/Qwen3.5-4B; V=R1X1F1B2D1C1E0
ENV="CUDA_HOME=$HOME/vllm-venv/lib/python3.12/site-packages/nvidia/cu13 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
for a in O1 O3; do [ -f $O/${a}_records.jsonl ] || { echo "[missing $O/${a}_records.jsonl] $(date -Is)"; exit 1; }; done
echo "[start seqs=32] $(date -Is)"
env $ENV ~/vllm-venv/bin/python scripts/prompt_sweep.py --engine vllm --model $M --gpu-mem 0.89 --max-model-len 8192 --max-num-seqs 32 \
  --records $O/O1_records.jsonl,$O/O3_records.jsonl --arm O1,O3 --variants "$V" --out-dir $O/run1 --max-new-tokens 4096 --resume
echo "[exit $?] $(date -Is)"
env PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/format_dev_20261017/score_formats2.py $O/run1 --gold train | tee $O/score.txt
env PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python $O/compare.py | tee $O/compare.txt
echo "[all done] $(date -Is)"
