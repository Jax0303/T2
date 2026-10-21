#!/usr/bin/env bash
# 셀 고르기 단계 dev 395 — PREREG-2026-10-21-select-dev.md. 1) 줄 선택(A·D·Cp, 같은 LLM) 2) 남긴 줄로 P* 답변 2회(동시 32/20, 기준 실행과 같음) 3) 비교
set -uo pipefail
cd "$(dirname "$0")/../.."
O=results/select_dev_20261021; SW=results/prompt_sweep_20261015; FD=results/format_dev_20261017; SD=results/sentence_dev_20261019; M=Qwen/Qwen3.5-4B; V=R1X1F1B2D1C1E0
ENV="CUDA_HOME=$HOME/vllm-venv/lib/python3.12/site-packages/nvidia/cu13 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
if [ ! -f $O/CpF_records.jsonl ]; then
  echo "[start select] $(date -Is)"
  env $ENV ~/vllm-venv/bin/python scripts/select_lines.py --model $M --records $SW/dev_A_records.jsonl,$FD/D_records.jsonl,$SD/Cp_records.jsonl --arm A,D,Cp --out-dir $O --max-num-seqs 32
  echo "[exit $? select] $(date -Is)"
fi
for a in A D Cp; do [ -f $O/${a}F_records.jsonl ] || { echo "[missing $O/${a}F_records.jsonl] $(date -Is)"; exit 1; }; done
for RUN in 1 2; do
  SEQS=$([ "$RUN" = 1 ] && echo 32 || echo 20)
  echo "[start run$RUN seqs=$SEQS] $(date -Is)"
  env $ENV ~/vllm-venv/bin/python scripts/prompt_sweep.py --engine vllm --model $M --gpu-mem 0.89 --max-model-len 8192 --max-num-seqs $SEQS \
    --records $O/AF_records.jsonl,$O/DF_records.jsonl,$O/CpF_records.jsonl --arm AF,DF,CpF --variants "$V" --out-dir $O/run$RUN --max-new-tokens 4096 --resume
  echo "[exit $? run$RUN] $(date -Is)"
done
env PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python $FD/score_formats2.py $O/run1 $O/run2 | tee $O/score.txt
env PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python $O/compare.py | tee $O/compare.txt
echo "[all done] $(date -Is)"
