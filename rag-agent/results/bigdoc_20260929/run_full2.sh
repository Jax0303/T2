#!/usr/bin/env bash
# 표 전체 입력(합친 문서) 이어 돌리기: GPU 메모리가 7.9/8.2GB 로 차서 문항당 50초→140초로 느려짐 → 프로세스 재시작(--resume, 같은 조건)
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
D=results/bigdoc_20260929; IDS=results/mh_only_20260928/baselines/ids200.jsonl
echo "[restart] $(date -Is) commit $(git rev-parse --short HEAD) rows=$(wc -l < $D/bigdoc_fulltable_answer.jsonl)" >> $D/run_full1.log
t=$(date +%s)
.venv/bin/python scripts/mh_bigdoc.py --stage answer -- --records $D/bigdoc_s3c_records.jsonl --scope doc --condition fulltable \
  --split train --header-rule v3.3u --batch-size 1 --same-queries-as $IDS --quiet-accuracy --resume \
  --out $D/bigdoc_fulltable_answer.jsonl > $D/answer_fulltable_b1_resume.log 2>&1
echo "[end] $(date -Is) answer_fulltable_b1_resume rc=$? $(( $(date +%s) - t ))s" >> $D/run_full1.log
echo ALL_DONE >> $D/run_full1.log
