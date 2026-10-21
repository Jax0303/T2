#!/usr/bin/env bash
# 질문 쪼개기 검색 dev 395 — PREREG-2026-10-21-decompose-dev.md. 1) 쪼개기(LLM) 2) 검색: 기준 재실행·쪼개기 (validation 332 + train 63) 3) 비교
set -uo pipefail
cd "$(dirname "$0")/../.."
O=results/decompose_dev_20261021; M=Qwen/Qwen3.5-4B
ENV="CUDA_HOME=$HOME/vllm-venv/lib/python3.12/site-packages/nvidia/cu13 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
if [ ! -f $O/sub_dev.json ]; then
  echo "[start decompose] $(date -Is)"
  env $ENV ~/vllm-venv/bin/python scripts/decompose_queries.py --model $M --records $O/dev_questions.jsonl --out $O/sub_dev.json
  echo "[exit $? decompose] $(date -Is)"
fi
[ -f $O/sub_dev.json ] || { echo "[missing sub_dev.json]"; exit 1; }
for SPLIT in validation train; do
  U=$O/uids_$( [ $SPLIT = validation ] && echo val || echo train ).json
  for MODE in base sub; do
    TAG=${SPLIT}_${MODE}_k100; [ -f $O/${TAG}_records.jsonl ] && continue
    EXTRA=$( [ $MODE = sub ] && echo "--sub-queries $O/sub_dev.json" || echo "" )
    echo "[start retrieve $TAG] $(date -Is)"
    env PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python scripts/mh_arms.py --split $SPLIT --unit cell --template s3c --header-rule v3.3u --alpha 0.7 --budget 20 \
      --dump-ranked 100 --only-uids $U --tag $TAG --out-dir $O $EXTRA > $O/$TAG.log 2>&1
    echo "[exit $? $TAG] $(date -Is)"
  done
done
env PYTHONPATH=. .venv/bin/python $O/compare.py | tee $O/compare.txt
echo "[all done] $(date -Is)"
