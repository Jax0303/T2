#!/usr/bin/env bash
# 밤 사슬 2026-10-22: 600b 지시문 4개 실행(PID 223885)이 끝나면 → 1) 잘린 답변 마무리(budget forcing) 전 집합 → 2) self-consistency dev. 사전등록 두 건 참조.
set -uo pipefail
cd "$(dirname "$0")/.."
ENV="CUDA_HOME=$HOME/vllm-venv/lib/python3.12/site-packages/nvidia/cu13 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
M=Qwen/Qwen3.5-4B; BF=results/budget_forcing_20261022; SC=results/sc_dev_20261022; SD=results/sentence_dev_20261019
while kill -0 223885 2>/dev/null; do sleep 60; done
echo "[chain] 600b 실행 종료 확인 $(date -Is)"
echo "[chain] 1 마무리 단계 시작 $(date -Is)"
.venv/bin/python $BF/make_jobs.py
env $ENV ~/vllm-venv/bin/python scripts/force_final.py --model $M --jobs $BF/jobs.json > $BF/force.log 2>&1; echo "[chain] force exit $?"
env PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python $BF/compare.py > $BF/compare.txt 2> $BF/compare.err; echo "[chain] force compare exit $?"
echo "[chain] 2 self-consistency dev 시작 $(date -Is)"
env $ENV ~/vllm-venv/bin/python scripts/prompt_sweep.py --engine vllm --model $M --gpu-mem 0.89 --max-model-len 8192 --max-num-seqs 16 \
  --records results/prompt_sweep_20261015/dev_A_records.jsonl,results/format_dev_20261017/D_records.jsonl,$SD/Cp_records.jsonl --arm A,D,Cp --variants R1X1F1B2D1C1E0 \
  --out-dir $SC/run1 --max-new-tokens 4096 --n-samples 5 --temperature 0.7 --top-p 0.8 --top-k 20 --resume > $SC/run1.log 2>&1; echo "[chain] sc exit $?"
env PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python $SC/compare.py > $SC/compare.txt 2> $SC/compare.err; echo "[chain] sc compare exit $?"
echo "[chain done] $(date -Is)"
