# 사후 조건 본 방법 v3.3 을 먼저 → randrow 이어 쓰기 → tablerag_leaf 재실행(CUDA unknown error 로 0행 중단). 2026-09-25 01:4x
set -u
cd /home/user/T2-1/rag-agent
export PYTHONPATH=. HF_HUB_OFFLINE=1
D=results/mh_arms/cap300_20260924

echo "##### cell_hv33 start $(date +%F' '%T)"
.venv/bin/python scripts/answer_accuracy_mh.py --scope doc --batch-size 128 --header-rule v3.3 \
  --same-queries-as $D/cell.jsonl --out $D/cell_hv33.jsonl \
  --records results/mh_interim200_v33/mh_cell_hv33_records.jsonl --condition retrieved >> $D/cell_hv33.log 2>&1
echo "##### cell_hv33 exit=$? $(date +%F' '%T)"
echo "##### randrow start $(date +%F' '%T)"
.venv/bin/python scripts/answer_accuracy_mh.py --scope doc --stratum-cap 300 --sample-seed 20260913 --batch-size 128 \
  --out $D/randrow.jsonl --records results/mh_arms/mh_train_randrow_hv1_none_doc_records.jsonl \
  --condition retrieved --resume >> $D/randrow.log 2>&1
echo "##### randrow exit=$? $(date +%F' '%T)"
echo "##### tablerag_leaf start $(date +%F' '%T)"
.venv/bin/python scripts/answer_accuracy_mh.py --scope doc --stratum-cap 300 --sample-seed 20260913 --batch-size 128 \
  --out $D/tablerag_leaf.jsonl --records results/mh_arms/mh_train_tablerag_leaf_hv1_none_doc_records.jsonl \
  --condition retrieved --resume >> $D/tablerag_leaf.log 2>&1
echo "##### tablerag_leaf exit=$? $(date +%F' '%T)"
echo AFTER_DONE
