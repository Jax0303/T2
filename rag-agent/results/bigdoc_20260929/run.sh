#!/usr/bin/env bash
# PREREG-2026-09-29-bigdoc.md: 합친 문서(표 ~20개) 검색 1회 → 본 방법 답변 → 표 전체 입력 답변
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
D=results/bigdoc_20260929; IDS=results/mh_only_20260928/baselines/ids200.jsonl
echo "[start] $(date -Is) commit $(git rev-parse --short HEAD)"
t=$(date +%s)
.venv/bin/python scripts/mh_bigdoc.py --stage retrieve -- --split train --header-rule v3.3u --alpha 0.7 --budget 20 \
  --embed-model BAAI/bge-base-en-v1.5 --k-ladder 1,5,10,20 --dump-ranked 20 --cache-dir .cache/bigdoc_20260929 \
  --unit cell --template s3c --out-dir $D --tag bigdoc_s3c > $D/retrieve.log 2>&1
echo "[end] $(date -Is) retrieve rc=$? $(( $(date +%s) - t ))s"
t=$(date +%s)
.venv/bin/python scripts/mh_bigdoc.py --stage answer -- --records $D/bigdoc_s3c_records.jsonl --scope doc --condition retrieved \
  --split train --header-rule v3.3u --batch-size 128 --same-queries-as $IDS --quiet-accuracy \
  --out $D/bigdoc_s3c_answer.jsonl > $D/answer_s3c.log 2>&1
echo "[end] $(date -Is) answer_s3c rc=$? $(( $(date +%s) - t ))s"
t=$(date +%s)
.venv/bin/python scripts/mh_bigdoc.py --stage answer -- --records $D/bigdoc_s3c_records.jsonl --scope doc --condition fulltable \
  --split train --header-rule v3.3u --batch-size 8 --same-queries-as $IDS --quiet-accuracy \
  --out $D/bigdoc_fulltable_answer.jsonl > $D/answer_fulltable.log 2>&1
echo "[end] $(date -Is) answer_fulltable rc=$? $(( $(date +%s) - t ))s"
echo ALL_DONE
