#!/usr/bin/env bash
# 2026-09-28 PREREG-2026-09-28-tablerag-official-dtype.md 사후 추가: HiTab 300 답변, TableRAG 공식 규칙 문맥
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
D=results/tablerag_official_20260928
echo "[start] $(date -Is) commit $(git rev-parse --short HEAD)"
for arm in leaf path; do
  .venv/bin/python scripts/fair_filter_eval.py --no-filter --arms tablerag_$arm \
    --records $D/hitab/hitab_test_gold_tablerag_${arm}_official_records.jsonl \
    --out $D/answers/hitab_tablerag_${arm}_official_rows.jsonl > $D/answers/hitab_tablerag_${arm}_official.log 2>&1
  echo "[end] $(date -Is) $arm rc=$?"
done
