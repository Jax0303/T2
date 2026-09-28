#!/usr/bin/env bash
# PREREG-2026-09-28-mh-only.md "test 실행(사용자 승인 20:0x)": ① 표 단위 검색 답변 882 1회 ② dev 선택 k*≠20 이면 본 방법 test 검색(예산 k*) + 답변 882 1회.
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
D=results/mh_only_20260928; SAME=results/mh_arms/cap300_20260924/cell_uniq.jsonl
MH="--split train --header-rule v3.3u --same-queries-as $SAME --batch-size 128 --quiet-accuracy --scope doc --condition retrieved"
until grep -q ALL_DONE $D/dev_full.log 2>/dev/null; do sleep 30; done
echo "[start] $(date -Is) commit $(git rev-parse --short HEAD)"
mkdir -p $D/test
.venv/bin/python scripts/answer_accuracy_mh.py --records $D/retrieval/mh_train_table_records.jsonl $MH \
  --out $D/test/mh_table_answer.jsonl > $D/test/mh_table_answer.log 2>&1
echo "[end] $(date -Is) table rc=$?"
K=$(.venv/bin/python -c "
import json, os
acc = {}
for k in (5, 10, 15, 20, 30, 50, 75, 100):
    f = f'$D/dev_k/dev_s3c_k{k}_answer.jsonl'
    if os.path.exists(f): acc[k] = sum(json.loads(l)['answer_correct'] for l in open(f))
print(max(sorted(acc), key=lambda k: (acc[k], -k)))")
echo "k_star=$K"
if [ "$K" != 20 ]; then
  .venv/bin/python scripts/mh_arms.py --split train --header-rule v3.3u --alpha 0.7 --budget $K --dump-ranked $(( K > 20 ? K : 20 )) \
    --embed-model BAAI/bge-base-en-v1.5 --k-ladder 1,5,10,20 --cache-dir .cache/rerun_20260926_mh --unit cell --template s3c \
    --out-dir $D/test --tag mh_train_s3c_k$K > $D/test/mh_train_s3c_k$K.log 2>&1
  echo "[end] $(date -Is) retrieval k=$K rc=$?"
  .venv/bin/python scripts/answer_accuracy_mh.py --records $D/test/mh_train_s3c_k${K}_records.jsonl $MH \
    --out $D/test/mh_s3c_k${K}_answer.jsonl > $D/test/mh_s3c_k${K}_answer.log 2>&1
  echo "[end] $(date -Is) answer k=$K rc=$?"
fi
echo ALL_DONE
