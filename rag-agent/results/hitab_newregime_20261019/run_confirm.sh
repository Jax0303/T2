#!/usr/bin/env bash
# HiTab 확인 표본 691(300 밖 모집단 전부): A·C·D × R0X1F1B2D1C1E0, 4,096 토큰, 3회(동시 32/24/16) — PREREG-2026-10-19-hitab-confirm.md
set -uo pipefail
cd "$(dirname "$0")/../.."
H=results/hitab_newregime_20261019; O=$H/confirm; M=Qwen/Qwen3.5-4B; V=R0X1F1B2D1C1E0
ENV="CUDA_HOME=$HOME/vllm-venv/lib/python3.12/site-packages/nvidia/cu13 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
for a in A C D; do [ -f $O/${a}_records.jsonl ] || { echo "[missing $O/${a}_records.jsonl] $(date -Is)"; exit 1; }; done
for RUN in 1 2 3; do
  case $RUN in 1) SEQS=32;; 2) SEQS=24;; *) SEQS=16;; esac
  echo "[start run$RUN seqs=$SEQS] $(date -Is)"
  env $ENV ~/vllm-venv/bin/python scripts/prompt_sweep.py --engine vllm --model $M --gpu-mem 0.89 --max-model-len 8192 --max-num-seqs $SEQS \
    --records $O/A_records.jsonl,$O/C_records.jsonl,$O/D_records.jsonl --arm A,C,D --variants "$V" --out-dir $O/run$RUN --max-new-tokens 4096 --resume
  echo "[exit $? run$RUN] $(date -Is)"
done
env HITAB_DIR=$O PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python $H/score_hitab.py $O/run1 $O/run2 $O/run3 | tee $O/score.txt
env HITAB_DIR=$O PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python $H/analyze.py run1 run2 run3 | tee $O/analyze.txt
echo "[all done] $(date -Is)"
