#!/usr/bin/env bash
# PREREG-2026-10-01-alpha-dev.md: dev 332, 본 방법 s3c, α ∈ {0,.3,.5,.6,.8,.9,1}; 0.7 = results/mh_only_20260928/retrieval/mh_dev_s3c_b50
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
D=results/alpha_dev_20261001
echo "[start] $(date -Is) commit $(git rev-parse --short HEAD)"
for a in 0.0 0.3 0.5 0.6 0.8 0.9 1.0; do
  t=$(date +%s)
  .venv/bin/python scripts/mh_arms.py --split validation --header-rule v3.3u --alpha $a --budget 50 --dump-ranked 50 \
    --embed-model BAAI/bge-base-en-v1.5 --k-ladder 1,3,5,10,15,20,30,50 --cache-dir .cache/rerun_20260926_mh \
    --unit cell --template s3c --out-dir $D --tag mh_dev_s3c_a${a} > $D/mh_dev_s3c_a${a}.log 2>&1
  echo "[end] $(date -Is) alpha=$a rc=$? $(( $(date +%s) - t ))s"
done
echo ALL_DONE
