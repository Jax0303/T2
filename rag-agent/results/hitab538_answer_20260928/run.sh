#!/usr/bin/env bash
# PREREG-2026-09-28-hitab538-answer.md: 다른 세션 run_rest.sh(PID 10604) 와 GPU 작업이 끝난 뒤 조건마다 1회
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
D=results/hitab538_answer_20260928; R=results/rerun_20260926/hitab
while kill -0 10604 2>/dev/null || pgrep -f '^\.venv/bin/python scripts/(fair_filter_eval|answer_accuracy|hitab_fulltable)' >/dev/null; do sleep 30; done
echo "[start] $(date -Is) commit $(git rev-parse --short HEAD)"
for arm in s3c chunk; do
  .venv/bin/python scripts/fair_filter_eval.py --arms $arm --no-filter \
    --records $R/hitab_test_split_${arm}_records.jsonl --out $D/${arm}_rows.jsonl > $D/${arm}_run.log 2>&1
  echo "[end] $(date -Is) $arm rc=$?"
done
.venv/bin/python scripts/hitab_fulltable_answer.py \
  --top1-records $R/hitab_test_split_table_records.jsonl --out $D/table_top1_rows.jsonl > $D/table_top1_run.log 2>&1
echo "[end] $(date -Is) table_top1 rc=$?"
.venv/bin/python $D/analyze.py > $D/analyze.log 2>&1
echo "[end] $(date -Is) analyze rc=$?"
echo ALL_DONE
