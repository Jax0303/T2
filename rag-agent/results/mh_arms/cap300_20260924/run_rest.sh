# run.sh 체인이 fulltable 도중 끊겼다(2026-09-24 18:39, 생성 프로세스는 계속 돎). 그 프로세스가 끝나기를 기다렸다가
# fulltable 이 미완이면 --resume 으로 잇고, 나머지 조건을 run.sh 와 같은 인자로 돌린다.
set -u
cd /home/user/T2-1/rag-agent
export PYTHONPATH=. HF_HUB_OFFLINE=1
D=results/mh_arms/cap300_20260924
while kill -0 113502 2>/dev/null; do sleep 30; done
run() {
  name=$1; shift
  echo "##### $name start $(date +%F' '%T)"
  .venv/bin/python scripts/answer_accuracy_mh.py --scope doc --stratum-cap 300 --sample-seed 20260913 \
    --batch-size 128 --out $D/$name.jsonl "$@" >> $D/$name.log 2>&1
  echo "##### $name exit=$? $(date +%F' '%T)"
}
[ -f $D/fulltable.json ] || run fulltable --records results/mh_arms/mh_train_cell_hv1_none_doc_records.jsonl --condition fulltable --resume
for arm in chunk rowcol trag_hetero tablerag_path tablerag_leaf randrow; do
  run $arm --records results/mh_arms/mh_train_${arm}_hv1_none_doc_records.jsonl --condition retrieved
done
echo "##### hitab_fulltable start $(date +%F' '%T)"
.venv/bin/python scripts/hitab_fulltable_answer.py > $D/hitab_fulltable.log 2>&1
echo "##### hitab_fulltable exit=$? $(date +%F' '%T)"
echo RUN_DONE
