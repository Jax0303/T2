#!/usr/bin/env bash
# 꺽쇠 경로 설명 지시문(X3) 확인: 새 문항 600, D·C·A × R1X3F1B2D1C1E0, 4,096 토큰, 3회(동시 32/24/16) — PREREG-2026-10-20-bracket-explain.md
set -uo pipefail
cd "$(dirname "$0")/../.."
O=results/mh_confirm_20261019; FD=results/format_dev_20261017; M=Qwen/Qwen3.5-4B; V=R1X3F1B2D1C1E0
ENV="CUDA_HOME=$HOME/vllm-venv/lib/python3.12/site-packages/nvidia/cu13 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
for a in A C D; do [ -f $O/${a}_records.jsonl ] || { echo "[missing $O/${a}_records.jsonl] $(date -Is)"; exit 1; }; done
for RUN in 1 2 3; do
  case $RUN in 1) SEQS=32;; 2) SEQS=24;; *) SEQS=16;; esac
  echo "[start run${RUN}x3 seqs=$SEQS] $(date -Is)"
  env $ENV ~/vllm-venv/bin/python scripts/prompt_sweep.py --engine vllm --model $M --gpu-mem 0.89 --max-model-len 8192 --max-num-seqs $SEQS \
    --records $O/D_records.jsonl,$O/C_records.jsonl,$O/A_records.jsonl --arm D,C,A --variants "$V" --out-dir $O/run${RUN}x3 --max-new-tokens 4096 --resume
  echo "[exit $? run${RUN}x3] $(date -Is)"
done
env PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python $FD/score_formats2.py $O/run1x3 $O/run2x3 $O/run3x3 --gold train --variant $V | tee $O/score_x3.txt
env PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python $O/score_x3.py --base $O/run1 $O/run2 $O/run3 --new $O/run1x3 $O/run2x3 $O/run3x3 | tee $O/score_x3_pair.txt
echo "[all done] $(date -Is)"
