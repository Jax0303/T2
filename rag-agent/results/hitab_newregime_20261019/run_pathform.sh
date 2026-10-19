#!/usr/bin/env bash
# HiTab 확인 691 경로형 목록(Cp) 3회(동시 32/24/16) — PREREG-2026-10-19-hitab-pathform.md. 기존 A·C·D(run1~3)와 다수결로 비교
set -uo pipefail
cd "$(dirname "$0")/../.."
H=results/hitab_newregime_20261019; O=$H/confirm; M=Qwen/Qwen3.5-4B; V=R0X1F1B2D1C1E0
ENV="CUDA_HOME=$HOME/vllm-venv/lib/python3.12/site-packages/nvidia/cu13 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
for RUN in 1 2 3; do
  case $RUN in 1) SEQS=32;; 2) SEQS=24;; *) SEQS=16;; esac
  echo "[start runp$RUN seqs=$SEQS] $(date -Is)"
  env $ENV ~/vllm-venv/bin/python scripts/prompt_sweep.py --engine vllm --model $M --gpu-mem 0.89 --max-model-len 8192 --max-num-seqs $SEQS \
    --records $O/Cp_records.jsonl --arm Cp --variants "$V" --out-dir $O/runp$RUN --max-new-tokens 4096 --resume
  echo "[exit $? runp$RUN] $(date -Is)"
done
env HITAB_DIR=$O PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python $H/score_hitab.py $O/run1 $O/run2 $O/run3 $O/runp1 $O/runp2 $O/runp3 | tee $O/score_pathform.txt
echo "[all done] $(date -Is)"
