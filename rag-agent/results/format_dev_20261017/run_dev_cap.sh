#!/usr/bin/env bash
# 표 수 제한 변형 dev 395 × 팔 5(A·D·D1·D2·DF2) × P*, vLLM, 최대 3,072 토큰, 실행 2회(동시 32 / 동시 24 — 묶음 구성 차이) — PREREG-2026-10-18-table-cap.md
set -uo pipefail
cd "$(dirname "$0")/../.."
D=results/format_dev_20261017; SW=results/prompt_sweep_20261015; M=Qwen/Qwen3.5-4B; V=R1X1F1B2D1C1E0
ENV="CUDA_HOME=$HOME/vllm-venv/lib/python3.12/site-packages/nvidia/cu13 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
for RUN in 1 2; do
  SEQS=$([ "$RUN" = 1 ] && echo 32 || echo 24)
  echo "[start run$RUN seqs=$SEQS] $(date -Is)"
  env $ENV ~/vllm-venv/bin/python scripts/prompt_sweep.py --engine vllm --model $M --gpu-mem 0.89 --max-model-len 7168 --max-num-seqs $SEQS \
    --records $SW/dev_A_records.jsonl,$D/D_records.jsonl,$D/D1_records.jsonl,$D/D2_records.jsonl,$D/DF2_records.jsonl --arm A,D,D1,D2,DF2 \
    --variants "$V" --out-dir $D/devcap_run$RUN --max-new-tokens 3072 --resume
  echo "[exit $? run$RUN] $(date -Is)"
done
env PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python $D/score_formats2.py $D/devcap_run1 $D/devcap_run2 | tee $D/score_devcap.txt
echo "[all done] $(date -Is)"
