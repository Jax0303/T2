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
# 2026-09-29 11:0x 추가: Yu 검색이 GPU 메모리 포화로 0.9→7.5초/문항으로 느려져 끊고(1,2xx/2,885) 여기서 이어 돈다
.venv/bin/python scripts/baselines/tablerag_yu.py --out-dir results/mh_only_20260928/baselines/tablerag_yu --stage retrieve --resume \
  > results/mh_only_20260928/baselines/tablerag_yu/retrieve_resume.log 2>&1
echo "[end] $(date -Is) yu_retrieve_resume rc=$?"
'
kill -CONT 1170410
echo "[resumed] $(date -Is) run_rest.sh"
echo ALL_DONE
