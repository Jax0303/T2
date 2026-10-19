#!/usr/bin/env bash
# MultiHiertt 새 문항 확인 600(산술 셀2+): D·C·A × P*, 4,096 토큰, 3회(동시 32/24/16) — PREREG-2026-10-19-mh-confirm-arith.md
set -uo pipefail
cd "$(dirname "$0")/../.."
O=results/mh_confirm_20261019; FD=results/format_dev_20261017; M=Qwen/Qwen3.5-4B; V=R1X1F1B2D1C1E0
ENV="CUDA_HOME=$HOME/vllm-venv/lib/python3.12/site-packages/nvidia/cu13 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
for a in A C D; do [ -f $O/${a}_records.jsonl ] || { echo "[missing $O/${a}_records.jsonl] $(date -Is)"; exit 1; }; done
for RUN in 1 2 3; do
  case $RUN in 1) SEQS=32;; 2) SEQS=24;; *) SEQS=16;; esac
  echo "[start run$RUN seqs=$SEQS] $(date -Is)"
  env $ENV ~/vllm-venv/bin/python scripts/prompt_sweep.py --engine vllm --model $M --gpu-mem 0.89 --max-model-len 8192 --max-num-seqs $SEQS \
    --records $O/D_records.jsonl,$O/C_records.jsonl,$O/A_records.jsonl --arm D,C,A --variants "$V" --out-dir $O/run$RUN --max-new-tokens 4096 --resume
  echo "[exit $? run$RUN] $(date -Is)"
done
env PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python $FD/score_formats2.py $O/run1 $O/run2 $O/run3 --gold train | tee $O/score.txt
env PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python $FD/score_rep.py $O/run1 $O/run2 $O/run3 --gold train | tee $O/score_majority.txt
echo "[all done] $(date -Is)"
