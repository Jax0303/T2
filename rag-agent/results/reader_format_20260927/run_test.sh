#!/usr/bin/env bash
# PREREG-2026-09-27-reader-format.md — test 실행(GATE 2 이후 사용자 지시가 있을 때만). 로그에는 진행 문항 수만 남는다.
# 실행: mkdir -p results/reader_format_20260927/test && setsid nohup bash results/reader_format_20260927/run_test.sh > results/reader_format_20260927/test/run.log 2>&1 &
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
PY=.venv/bin/python; D=results/reader_format_20260927/test; mkdir -p $D
echo "[start] $(date -Is) commit $(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
SAME=results/mh_arms/cap300_20260924/cell_uniq.jsonl
MH="--split train --header-rule v3.3u --same-queries-as $SAME --batch-size 128 --quiet-accuracy"
[ -e $D/hitab_rowexp.jsonl ] || $PY scripts/fair_filter_eval.py --arms s3c --no-filter --context rowexp --out $D/hitab_rowexp.jsonl
echo "[end] $(date -Is) hitab_rowexp rc=$?"
$PY scripts/answer_accuracy_mh.py --records results/mh_arms/mh_train_cell_hv3.3u_none_doc_records.jsonl $MH \
  --condition rowexp --out $D/mh_rowexp.jsonl; echo "[end] $(date -Is) mh_rowexp rc=$?"
$PY scripts/answer_accuracy_mh.py --records results/rerun_20260926/mh/mh_train_chunk_records.jsonl $MH \
  --condition retrieved --out $D/mh_chunk_final.jsonl; echo "[end] $(date -Is) mh_chunk_final rc=$?"
echo "[done] $(date -Is)"
