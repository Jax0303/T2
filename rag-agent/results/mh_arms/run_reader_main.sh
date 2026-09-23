set -u
# PREREG-2026-09-13-reader-qwen3.md §2 — 사용: bash run_reader_main.sh <태그> <리더> <프롬프트> <출력토큰>
#   예: bash results/mh_arms/run_reader_main.sh qwen3_8b_cot "local:Qwen/Qwen3-8B?quantization=4bit" cot 384
cd /home/user/T2-1/rag-agent
export PYTHONPATH=. HF_HUB_OFFLINE=1
TAG=$1 READER=$2 PROMPT=$3 MAXTOK=$4
D=results/mh_arms
leg () { name=$1; shift; out="$D/${name}_${TAG}.jsonl"; echo "##### ${name} $(date +%H:%M:%S)"
  .venv/bin/python scripts/answer_accuracy_mh.py --scope doc --header-rule v2 --reader "$READER" \
    --prompt "$PROMPT" --max-tokens "$MAXTOK" --out "$out" "$@" > "${out%.jsonl}.log" 2>&1
  echo "exit=$? $(date +%H:%M:%S)"; }
leg mh_cell_hv2_answer_doc --records $D/mh_cell_hv2_records.jsonl --condition retrieved \
  --same-contexts-as $D/mh_cell_hv2_answer_doc.jsonl
leg mh_GOLD_hv2_doc --records $D/mh_cell_hv2_records.jsonl --condition gold --label-rule none \
  --same-contexts-as $D/mh_GOLD_hv2_doc.jsonl
echo MAIN_DONE
