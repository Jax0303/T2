#!/usr/bin/env bash
# 표 전체 입력 이어 돌리기 2: --resume 이 code_sha256(mh_bigdoc.py 에 langchain 단계 추가) 불일치로 거부 →
# LangChain 이 끝난 뒤, 처음 실행과 같은 mh_bigdoc.py(fc02c8e)로 되돌려 이어 돌리고, 끝나면 d931abb 판으로 복구.
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
D=results/bigdoc_20260929; IDS=results/mh_only_20260928/baselines/ids200.jsonl
until grep -q ALL_DONE $D/run_lc.log 2>/dev/null; do sleep 30; done
git show fc02c8e:rag-agent/scripts/mh_bigdoc.py > scripts/mh_bigdoc.py
echo "[restart2] $(date -Is) script=fc02c8e rows=$(wc -l < $D/bigdoc_fulltable_answer.jsonl)"
t=$(date +%s)
.venv/bin/python scripts/mh_bigdoc.py --stage answer -- --records $D/bigdoc_s3c_records.jsonl --scope doc --condition fulltable \
  --split train --header-rule v3.3u --batch-size 1 --same-queries-as $IDS --quiet-accuracy --resume \
  --out $D/bigdoc_fulltable_answer.jsonl > $D/answer_fulltable_b1_resume2.log 2>&1
echo "[end] $(date -Is) resume2 rc=$? $(( $(date +%s) - t ))s rows=$(wc -l < $D/bigdoc_fulltable_answer.jsonl)"
git checkout d931abb -- scripts/mh_bigdoc.py
echo ALL_DONE
