#!/usr/bin/env bash
# PREREG-2026-09-29-header-ablation.md — 헤더 제거(색인만 값) 검색 2,885 → 답변 882 → 멈춘 비교군 재개
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
O=results/header_ablation_20260929
flock .gpu.lock bash -c '
O='$O'
echo "[start] $(date -Is) commit $(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
.venv/bin/python scripts/mh_arms.py --split train --header-rule v3.3u --alpha 0.7 --budget 20 --embed-model BAAI/bge-base-en-v1.5 \
  --k-ladder 1,5,10,20 --dump-ranked 100 --cache-dir .cache/header_ablation_20260929 --unit cell --template s3c --cell-text value \
  --out-dir $O --tag mh_train_s3c_value > $O/mh_train_s3c_value.log 2>&1
echo "[end] $(date -Is) retrieval rc=$?"
.venv/bin/python scripts/answer_accuracy_mh.py --records $O/mh_train_s3c_value_records.jsonl --scope doc --condition retrieved \
  --split train --header-rule v3.3u --label-rule none --prompt cot --max-tokens 384 --batch-size 128 \
  --same-queries-as results/mh_arms/cap300_20260924/cell_uniq.jsonl --out $O/mh_value_answer.jsonl > $O/mh_value_answer.log 2>&1
echo "[end] $(date -Is) answer rc=$?"
'
kill -CONT 1170410
echo "[resumed] $(date -Is) run_rest.sh"
echo ALL_DONE
