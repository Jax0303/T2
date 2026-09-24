set -u
cd /home/user/T2-1/rag-agent
export PYTHONPATH=. HF_HUB_OFFLINE=1
# PREREG-2026-09-23-reader-thinking-pilot.md — (가) 재실행으로 행 보존, 그 뒤 채택 설정 본 실행
echo "##### pilot cot rerun $(date +%H:%M:%S)"
.venv/bin/python scripts/reader_cot_pilot.py --split validation --n 60 --label-rule none \
  --reader "local:Qwen/Qwen3-8B?quantization=4bit" --conditions cot \
  --out results/mh_arms/reader_pilot_cot_rerun_20260923_validation.jsonl \
  > results/mh_arms/reader_pilot_cot_rerun_20260923_validation.log 2>&1
echo "exit=$? $(date +%H:%M:%S)"
echo "##### main fulln cot384 $(date +%H:%M:%S)"
.venv/bin/python scripts/answer_accuracy_mh.py \
  --records results/mh_arms/mh_train_cell_hv1_none_doc_records.jsonl --scope doc --condition retrieved \
  --reader "local:Qwen/Qwen3-8B?quantization=4bit" --prompt cot --max-tokens 384 \
  --same-contexts-as results/mh_arms/mh_train_cell_hv1_none_doc_answer_doc_retrieved_fulln.jsonl \
  --out results/mh_arms/mh_train_cell_hv1_none_doc_answer_doc_retrieved_fulln_cot384.jsonl \
  > results/mh_arms/mh_train_cell_hv1_none_doc_answer_fulln_cot384.log 2>&1
echo "exit=$? $(date +%H:%M:%S)"
echo RUN_DONE
