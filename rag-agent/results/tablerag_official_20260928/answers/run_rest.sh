#!/usr/bin/env bash
# 2026-09-28 사후 추가(PREREG-2026-09-28-tablerag-official-dtype.md): run_hitab.sh(PID 8988) 뒤에 HiTab 필터 2회 → MultiHiertt 답변 2회
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
D=results/tablerag_official_20260928
while kill -0 8988 2>/dev/null; do sleep 20; done
echo "[start] $(date -Is) commit $(git rev-parse --short HEAD)"
for arm in leaf path; do
  .venv/bin/python scripts/fair_filter_eval.py --arms tablerag_$arm \
    --records $D/hitab/hitab_test_gold_tablerag_${arm}_official_records.jsonl \
    --out $D/answers/hitab_tablerag_${arm}_official_filter_rows.jsonl > $D/answers/hitab_tablerag_${arm}_official_filter.log 2>&1
  echo "[end] $(date -Is) hitab filter $arm rc=$?"
done
for arm in leaf path; do
  .venv/bin/python scripts/answer_accuracy_mh.py --scope doc --batch-size 128 --header-rule v3.3u \
    --same-queries-as results/mh_arms/cap300_20260924/cell.jsonl --condition retrieved \
    --records $D/mh/mh_train_tablerag_${arm}_official_records.jsonl \
    --out $D/answers/mh_tablerag_${arm}_official.jsonl > $D/answers/mh_tablerag_${arm}_official.log 2>&1
  echo "[end] $(date -Is) mh $arm rc=$?"
done
echo ALL_DONE
