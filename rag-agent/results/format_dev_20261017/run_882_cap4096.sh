#!/usr/bin/env bash
# 882 확인 실행 3회(D·C·A, 4,096 토큰 표준, 동시 32 / 24 / 16) — PREREG-2026-10-19-882-cap4096.md
set -uo pipefail
cd "$(dirname "$0")/../.."
D=results/format_dev_20261017; CF=results/cellform_20261012; M=Qwen/Qwen3.5-4B; V=R1X1F1B2D1C1E0
ENV="CUDA_HOME=$HOME/vllm-venv/lib/python3.12/site-packages/nvidia/cu13 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
for RUN in 1 2 3; do
  case $RUN in 1) SEQS=32;; 2) SEQS=24;; *) SEQS=16;; esac
  echo "[start run$RUN seqs=$SEQS] $(date -Is)"
  env $ENV ~/vllm-venv/bin/python scripts/prompt_sweep.py --engine vllm --model $M --gpu-mem 0.89 --max-model-len 8192 --max-num-seqs $SEQS \
    --records $D/D882_records.jsonl,$CF/C_ours_budget_records.jsonl,$CF/A_chunk_md_records.jsonl --arm D,C,A \
    --variants "$V" --out-dir $D/final882_cap4096_run$RUN --max-new-tokens 4096 --resume
  echo "[exit $? run$RUN] $(date -Is)"
done
env PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python $D/score_formats2.py $D/final882_cap4096_run1 $D/final882_cap4096_run2 $D/final882_cap4096_run3 --gold train | tee $D/score_882_cap4096.txt
env PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python $D/score_rep.py $D/final882_cap4096_run1 $D/final882_cap4096_run2 $D/final882_cap4096_run3 --gold train | tee $D/score_882_cap4096_majority.txt
echo "[all done] $(date -Is)"
