#!/usr/bin/env bash
# 표 전체 입력(합친 문서) 재실행: 긴 프롬프트에서 continuous batching(8)이 8건 중 1건만 돌려줘 실패 → 배치 1로
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
D=results/bigdoc_20260929; IDS=results/mh_only_20260928/baselines/ids200.jsonl
echo "[start] $(date -Is) commit $(git rev-parse --short HEAD)"
t=$(date +%s)
.venv/bin/python scripts/mh_bigdoc.py --stage answer -- --records $D/bigdoc_s3c_records.jsonl --scope doc --condition fulltable \
  --split train --header-rule v3.3u --batch-size 1 --same-queries-as $IDS --quiet-accuracy \
  --out $D/bigdoc_fulltable_answer.jsonl > $D/answer_fulltable_b1.log 2>&1
echo "[end] $(date -Is) answer_fulltable_b1 rc=$? $(( $(date +%s) - t ))s"
echo ALL_DONE
