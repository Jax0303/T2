#!/usr/bin/env bash
# PREREG-2026-09-28-mh-only.md "k 선택(dev)" 보충: dev 332 표 전체 입력(참고 조건, k=문서 전체) 답변 1회 — k 곡선의 끝점.
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
D=results/mh_only_20260928; P=results/reader_format_20260927/mh_dev_pop332.jsonl
until grep -q ALL_DONE $D/dev_k2.log 2>/dev/null; do sleep 30; done
echo "[start] $(date -Is) commit $(git rev-parse --short HEAD)"
.venv/bin/python scripts/answer_accuracy_mh.py --records $D/dev_k/dev_s3c_k20_records.jsonl --scope doc --condition fulltable \
  --split validation --header-rule v3.3u --batch-size 128 --same-queries-as $P --quiet-accuracy \
  --out $D/dev_k/dev_fulltable_answer.jsonl > $D/dev_k/dev_fulltable_answer.log 2>&1
echo "[end] $(date -Is) fulltable rc=$?"
echo ALL_DONE
