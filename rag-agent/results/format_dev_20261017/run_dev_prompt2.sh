#!/usr/bin/env bash
# 지시문 표 규칙(P*T = R1X2F1B2D2C1E0) 대 P*, 팔 A·D, dev 395, 최대 4,096 토큰, 실행 2회 — PREREG-2026-10-19-prompt-table-rule.md
set -uo pipefail
cd "$(dirname "$0")/../.."
D=results/format_dev_20261017; SW=results/prompt_sweep_20261015; M=Qwen/Qwen3.5-4B
ENV="CUDA_HOME=$HOME/vllm-venv/lib/python3.12/site-packages/nvidia/cu13 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
for RUN in 1 2; do
  SEQS=$([ "$RUN" = 1 ] && echo 32 || echo 20)
  echo "[start run$RUN seqs=$SEQS] $(date -Is)"
  env $ENV ~/vllm-venv/bin/python scripts/prompt_sweep.py --engine vllm --model $M --gpu-mem 0.89 --max-model-len 8192 --max-num-seqs $SEQS \
    --records $SW/dev_A_records.jsonl,$D/D_records.jsonl --arm A,D --variants "R1X1F1B2D1C1E0,R1X2F1B2D2C1E0" --out-dir $D/devprompt2_run$RUN --max-new-tokens 4096 --resume
  echo "[exit $? run$RUN] $(date -Is)"
done
for V in R1X1F1B2D1C1E0 R1X2F1B2D2C1E0; do
  echo "=== $V ==="; env PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python $D/score_formats2.py $D/devprompt2_run1 $D/devprompt2_run2 --variant $V
done | tee $D/score_devprompt2.txt
echo "[all done] $(date -Is)"
