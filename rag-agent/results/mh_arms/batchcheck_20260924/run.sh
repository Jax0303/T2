set -u
cd /home/user/T2-1/rag-agent
export PYTHONPATH=. HF_HUB_OFFLINE=1
REF=results/mh_arms/mh_train_cell_hv1_none_doc_answer_doc_retrieved_fulln_cot384.jsonl
REC=results/mh_arms/mh_train_cell_hv1_none_doc_records.jsonl
D=results/mh_arms/batchcheck_20260924
# (1) 배치 1 재실행 20건 — 같은 코드 경로의 재현성 기준선
.venv/bin/python scripts/answer_accuracy_mh.py --records $REC --scope doc --condition retrieved \
  --same-contexts-as $REF --limit 20 --batch-size 1 --out $D/b1_20.jsonl > $D/b1_20.log 2>&1
echo "b1 exit=$?"
# (2) continuous batching 120건, 한 번에 넘김
.venv/bin/python scripts/answer_accuracy_mh.py --records $REC --scope doc --condition retrieved \
  --same-contexts-as $REF --limit 120 --batch-size 120 --out $D/cb_120.jsonl > $D/cb_120.log 2>&1
echo "cb exit=$?"
