# PREREG-2026-09-24-mh-answer-cap300.md — 배치 확인 통과 후 실행. 조건 하나가 실패해도 다음으로 넘어간다(종료 코드를 남긴다).
set -u
cd /home/user/T2-1/rag-agent
export PYTHONPATH=. HF_HUB_OFFLINE=1
D=results/mh_arms/cap300_20260924
run() {  # $1 = 이름, 나머지 = 추가 인자
  name=$1; shift
  echo "##### $name start $(date +%F' '%T)"
  .venv/bin/python scripts/answer_accuracy_mh.py --scope doc --stratum-cap 300 --sample-seed 20260913 \
    --batch-size 128 --out $D/$name.jsonl "$@" > $D/$name.log 2>&1
  echo "##### $name exit=$? $(date +%F' '%T)"
}
run cell --records results/mh_arms/mh_train_cell_hv1_none_doc_records.jsonl --condition retrieved
run fulltable --records results/mh_arms/mh_train_cell_hv1_none_doc_records.jsonl --condition fulltable
for arm in chunk rowcol trag_hetero tablerag_path tablerag_leaf randrow; do
  run $arm --records results/mh_arms/mh_train_${arm}_hv1_none_doc_records.jsonl --condition retrieved
done
echo "##### hitab_fulltable start $(date +%F' '%T)"
.venv/bin/python scripts/hitab_fulltable_answer.py > $D/hitab_fulltable.log 2>&1
echo "##### hitab_fulltable exit=$? $(date +%F' '%T)"
echo RUN_DONE
