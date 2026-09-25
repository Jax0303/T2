# 2026-09-24 22:4x 사용자 결정: 논문 필수 조건(chunk 재실행·HiTab 표 전체)을 먼저 돌리고 나머지를 잇는다.
# run_rest.sh·run_fix.sh 는 중단했다(tablerag_path 256행 저장, 코드 해시가 같아 --resume 으로 잇는다).
# chunk 는 docmath_match inf 예측으로 859/882 에서 죽었고 채점 수정으로 해시가 바뀌어 처음부터 다시 돈다 — 죽은 파일은 *.crashed.* 로 옮긴다.
set -u
cd /home/user/T2-1/rag-agent
export PYTHONPATH=. HF_HUB_OFFLINE=1
D=results/mh_arms/cap300_20260924
run() {
  name=$1; shift
  echo "##### $name start $(date +%F' '%T)"
  .venv/bin/python scripts/answer_accuracy_mh.py --scope doc --stratum-cap 300 --sample-seed 20260913 \
    --batch-size 128 --out $D/$name.jsonl --records results/mh_arms/mh_train_${name}_hv1_none_doc_records.jsonl \
    --condition retrieved "$@" >> $D/$name.log 2>&1
  echo "##### $name exit=$? $(date +%F' '%T)"
}
for ext in jsonl run.json log; do [ -f $D/chunk.$ext ] && mv $D/chunk.$ext $D/chunk.crashed.$ext; done
run chunk
echo "##### hitab_fulltable start $(date +%F' '%T)"
.venv/bin/python scripts/hitab_fulltable_answer.py > $D/hitab_fulltable.log 2>&1
echo "##### hitab_fulltable exit=$? $(date +%F' '%T)"
run tablerag_path --resume
run tablerag_leaf
run randrow
echo RUN_DONE
