set -u
cd /home/user/T2-1/rag-agent
export PYTHONPATH=. HF_HUB_OFFLINE=1
# PREREG-2026-09-13-reader-qwen3.md §1 — validation 산술 60건, gold 셀만, 헤더 v2, 라벨 none
pilot () { tag=$1; reader=$2; echo "##### pilot ${tag} $(date +%H:%M:%S)"
  .venv/bin/python scripts/reader_cot_pilot.py --split validation --n 60 --label-rule none --reader "$reader" \
    --out "results/mh_arms/reader_pilot_${tag}_none_validation.jsonl" \
    > "results/mh_arms/reader_pilot_${tag}_none_validation.log" 2>&1
  echo "exit=$? $(date +%H:%M:%S)"; }
pilot qwen25_7b "local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit"
pilot qwen3_8b "local:Qwen/Qwen3-8B?quantization=4bit"
echo PILOT_DONE
