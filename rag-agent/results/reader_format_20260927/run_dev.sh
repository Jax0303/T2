#!/usr/bin/env bash
# PREREG-2026-09-27-reader-format.md — dev 실행(두 조건). 로그에는 진행 문항 수만 남는다.
# 실행: mkdir -p results/reader_format_20260927/dev && setsid nohup bash results/reader_format_20260927/run_dev.sh > results/reader_format_20260927/dev/run.log 2>&1 &
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
PY=.venv/bin/python; D=results/reader_format_20260927/dev; mkdir -p $D
echo "[start] $(date -Is) commit $(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
HREC=results/dev_alpha_20260926/hitab_dev/hitab_dev_gold_prefix_records.jsonl
HPOP=results/reader_format_20260927/hitab_dev_pop300.json
MREC=results/dev_alpha_20260926/mh_dev/mh_dev_a1.0_records.jsonl
MPOP=results/reader_format_20260927/mh_dev_pop332.jsonl   # 판정 집합 332건(표 근거만 필요한 문항), 911건은 돌리지 않는다
MH="--records $MREC --split validation --header-rule v3.3u --same-queries-as $MPOP --batch-size 128 --quiet-accuracy"
for c in cell rowexp; do
  [ -e $D/hitab_$c.jsonl ] || $PY scripts/fair_filter_eval.py --arms s3c --no-filter --records $HREC --pop-file $HPOP \
    --context $c --out $D/hitab_$c.jsonl; echo "[end] $(date -Is) hitab_$c rc=$?"
done
$PY scripts/answer_accuracy_mh.py $MH --condition retrieved --out $D/mh_cell.jsonl; echo "[end] $(date -Is) mh_cell rc=$?"
$PY scripts/answer_accuracy_mh.py $MH --condition rowexp --out $D/mh_rowexp.jsonl; echo "[end] $(date -Is) mh_rowexp rc=$?"
echo "[done] $(date -Is)"
