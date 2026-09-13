set -u
# results/mh_interim200/PLAN.md — 200문항 탐색 비교: 두 방법 모두 200건 생성 후 분석. gold 레그 보류.
cd /home/user/T2-1/rag-agent
export PYTHONPATH=. HF_HUB_OFFLINE=1
D=results/mh_arms
H=results/mh_interim200
leg () { name=$1; shift; out="$H/${name}_qwen3_8b_cot_200.jsonl"; echo "##### ${name} 200 $(date +%H:%M:%S)"
  # 본 실행 레그와 같은 인자 — --same-contexts-as 와 --out 만 다르다
  .venv/bin/python scripts/answer_accuracy_mh.py --scope doc --header-rule v2 \
    --reader "local:Qwen/Qwen3-8B?quantization=4bit" --prompt cot --max-tokens 384 \
    --out "$out" --condition retrieved "$@" > "${out%.jsonl}.log" 2>&1
  echo "exit=$? $(date +%H:%M:%S)"; }
leg mh_cell_hv2_answer_doc --records $D/mh_cell_hv2_records.jsonl --same-contexts-as $H/ref_contexts_ours_200.jsonl
leg mh_mt2net_answer_doc --records $D/mh_mt2net_records.jsonl --same-contexts-as $H/ref_contexts_mt2net_200.jsonl
echo "##### analyze $(date +%H:%M:%S)"
.venv/bin/python $H/analyze.py --out $H/interim200_qwen3_8b_cot.jsonl > $H/analyze.log 2>&1
echo "exit=$? $(date +%H:%M:%S)"
echo INTERIM_DONE
