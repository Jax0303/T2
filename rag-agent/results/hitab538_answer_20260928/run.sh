#!/usr/bin/env bash
# PREREG-2026-09-28-hitab538-answer.md: 다른 세션의 MultiHiertt 답변 두 실행과 GPU 작업이 끝난 뒤 조건마다 1회
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
D=results/hitab538_answer_20260928; R=results/rerun_20260926/hitab
# 대기 조건: run_rest.sh 종료, MultiHiertt 답변 두 실행(leaf·path)의 완료 요약 파일 존재(끝날 때만 쓰임), GPU 답변 프로세스 없음
O=results/tablerag_official_20260928/answers
while kill -0 10604 2>/dev/null || [ ! -f $O/mh_tablerag_leaf_official.json ] || [ ! -f $O/mh_tablerag_path_official.json ] \
      || pgrep -f '^\.venv/bin/python scripts/(fair_filter_eval|answer_accuracy|hitab_fulltable)' >/dev/null; do sleep 30; done
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
