#!/usr/bin/env bash
# 문장 형태 × 꺽쇠 설명 지시문 dev 395 — PREREG-2026-10-19-sentence-form.md. 입력 7개 × 지시문 2개, 4,096 토큰, 2회(동시 32/20)
set -uo pipefail
cd "$(dirname "$0")/../.."
O=results/sentence_dev_20261019; SW=results/prompt_sweep_20261015; FD=results/format_dev_20261017; M=Qwen/Qwen3.5-4B; V="R1X1F1B2D1C1E0,R1X3F1B2D1C1E0"
ENV="CUDA_HOME=$HOME/vllm-venv/lib/python3.12/site-packages/nvidia/cu13 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
REC="$SW/dev_A_records.jsonl,$FD/D_records.jsonl,$O/Cp_records.jsonl,$O/S1T_records.jsonl,$O/S2T_records.jsonl,$O/S1same_records.jsonl,$O/S2same_records.jsonl"
MML=$(.venv/bin/python -c "import json; m=max(json.loads(l)['n_tok_ctx'] for p in '$REC'.split(',') for l in open(p)); print(8192 if m+4600<=8192 else (10240 if m+4600<=10240 else 12288))")
for RUN in 1 2; do
  SEQS=$([ "$RUN" = 1 ] && echo 32 || echo 20)
  echo "[start run$RUN seqs=$SEQS max-model-len=$MML] $(date -Is)"
  env $ENV ~/vllm-venv/bin/python scripts/prompt_sweep.py --engine vllm --model $M --gpu-mem 0.89 --max-model-len $MML --max-num-seqs $SEQS \
    --records $REC --arm A,D,Cp,S1T,S2T,S1same,S2same --variants "$V" --out-dir $O/run$RUN --max-new-tokens 4096 --resume
  echo "[exit $? run$RUN] $(date -Is)"
done
for v in R1X1F1B2D1C1E0 R1X3F1B2D1C1E0; do echo "=== $v ==="; env PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python $FD/score_formats2.py $O/run1 $O/run2 --variant $v; done | tee $O/score.txt
echo "[all done] $(date -Is)"
