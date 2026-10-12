#!/usr/bin/env bash
# 2026-09-29 이어 돌리기: 09-28 밤 generate_batch 오류("No requests can be scheduled")로 중간에 끊긴 답변 3개를
# 원래 인자 그대로 + --resume. test k=50 답변·run_r100 은 넣지 않았다(사용자 결정 전).
# 원래 스크립트의 rc=$? 는 $(date) 의 종료 코드를 읽었다 — 여기서는 명령 직후에 받고, 줄 수가 모자라면 멈춘다.
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
D=results/mh_only_20260928; P=results/reader_format_20260927/mh_dev_pop332.jsonl; SAME=results/mh_arms/cap300_20260924/cell_uniq.jsonl
DEV="--split validation --header-rule v3.3u --batch-size 128 --same-queries-as $P --quiet-accuracy --scope doc"
TEST="--split train --header-rule v3.3u --same-queries-as $SAME --batch-size 128 --quiet-accuracy --scope doc --condition retrieved"
run() {  # $1 = 기대 줄 수, $2 = 출력, 나머지 = 인자
  want=$1; out=$2; shift 2
  .venv/bin/python scripts/answer_accuracy_mh.py "$@" --out $out --resume >> ${out%.jsonl}.log 2>&1; rc=$?
  n=$(wc -l < $out); echo "[end] $(date -Is) $out rc=$rc rows=$n/$want"
  [ "$rc" = 0 ] && [ "$n" = "$want" ] || { echo STOPPED; exit 1; }
}
echo "[start] $(date -Is) commit $(git rev-parse --short HEAD)"
run 332 $D/dev_k/dev_s3c_k100_answer.jsonl --records $D/dev_k/dev_s3c_k100_records.jsonl --condition retrieved $DEV
run 332 $D/dev_k/dev_fulltable_answer.jsonl --records $D/dev_k/dev_s3c_k20_records.jsonl --condition fulltable $DEV
run 882 $D/test/mh_table_answer.jsonl --records $D/retrieval/mh_train_table_records.jsonl $TEST
echo ALL_DONE
