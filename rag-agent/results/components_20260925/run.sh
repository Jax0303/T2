#!/bin/bash
# 2026-09-25 구성요소 비교: HiTab test, 538표 한 색인, bge-base a5beb1e3, α=.7, 20셀
cd "$(dirname "$0")/../.."
for tpl in value s3label flat; do
  .venv/bin/python scripts/retrieval_accuracy.py --split test --corpus split --unit cell \
    --template $tpl --alpha 0.7 --budget 20 \
    --embed-model BAAI/bge-base-en-v1.5 --embed-revision a5beb1e3e68b9ab74eb54cfd186867f64f240e1a \
    --dump-context 1 --out-dir results/components_20260925 --tag $tpl \
    > results/components_20260925/$tpl.log 2>&1 || echo "FAIL $tpl" >> results/components_20260925/run.log
  echo "done $tpl" >> results/components_20260925/run.log
done
