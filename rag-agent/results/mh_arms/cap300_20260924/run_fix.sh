# chunk 가 docmath_match 의 inf 예측으로 859/882 에서 죽었다(2026-09-24 19:37). multihiertt_em.py 에 isfinite 가드를 넣어
# 코드 해시가 바뀌었으므로 --resume 은 거부된다 — run_rest.sh 가 끝나면 .json 이 없는 조건을 처음부터 다시 돌린다.
# 죽은 실행의 파일은 *.crashed.* 로 옮겨 둔다(지우지 않는다).
set -u
cd /home/user/T2-1/rag-agent
export PYTHONPATH=. HF_HUB_OFFLINE=1
D=results/mh_arms/cap300_20260924
while kill -0 128812 2>/dev/null; do sleep 30; done
for arm in chunk rowcol trag_hetero tablerag_path tablerag_leaf randrow; do
  [ -f $D/$arm.json ] && continue
  for ext in jsonl run.json log; do [ -f $D/$arm.$ext ] && mv $D/$arm.$ext $D/$arm.crashed.$ext; done
  echo "##### $arm rerun start $(date +%F' '%T)"
  .venv/bin/python scripts/answer_accuracy_mh.py --scope doc --stratum-cap 300 --sample-seed 20260913 \
    --batch-size 128 --out $D/$arm.jsonl --records results/mh_arms/mh_train_${arm}_hv1_none_doc_records.jsonl \
    --condition retrieved >> $D/$arm.log 2>&1
  echo "##### $arm rerun exit=$? $(date +%F' '%T)"
done
echo FIX_DONE
