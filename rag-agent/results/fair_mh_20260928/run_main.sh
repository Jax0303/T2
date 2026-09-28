#!/usr/bin/env bash
# PREREG-2026-09-28-fair-mh.md: 재검색이 끝난 뒤 필터 → 답변 (우선순위 순), 마지막에 정답 셀만 조건
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
D=results/fair_mh_20260928; R=$D/retrieval; Q=results/mh_arms/cap300_20260924/cell.jsonl
while ! grep -q ALL_DONE $D/run_retrieval.log 2>/dev/null; do sleep 20; done
echo "[start] $(date -Is) commit $(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
stage() {  # $1 arm
  .venv/bin/python scripts/mh_filter.py --records $R/mh_train_$1_records.jsonl --same-queries-as $Q \
    --out $D/filter/$1_filtered_records.jsonl > $D/filter/$1_filter.log 2>&1
  echo "[end] $(date -Is) filter $1 rc=$?"
  .venv/bin/python scripts/answer_accuracy_mh.py --records $D/filter/$1_filtered_records.jsonl --scope doc \
    --condition retrieved --header-rule v3.3u --batch-size 128 --same-queries-as $Q \
    --out $D/answer/$1_filtered_answer.jsonl > $D/answer/$1_filtered_answer.log 2>&1
  echo "[end] $(date -Is) answer $1 rc=$?"
}
mkdir -p $D/filter $D/answer
for a in ${ARMS:-s3c chunk}; do stage $a; done
.venv/bin/python scripts/answer_accuracy_mh.py --records $R/mh_train_s3c_records.jsonl --scope doc \
  --condition gold --header-rule v3.3u --batch-size 128 --same-queries-as $Q \
  --out $D/answer/gold_answer.jsonl > $D/answer/gold_answer.log 2>&1
echo "[end] $(date -Is) answer gold rc=$?"
for a in ${ARMS_LATE:-trag_hetero}; do stage $a; done
echo ALL_DONE
