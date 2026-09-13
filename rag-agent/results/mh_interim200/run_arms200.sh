set -u
# results/mh_interim200/PLAN_ARMS.md — 비교군 네 개, interim200 id, Qwen3-8B cot 384.
# 중단되면 같은 명령으로 다시 돌린다: 완료된 레그는 건너뛰고, 진행 중이던 레그는 --resume 으로 이어 쓴다.
cd /home/user/T2-1/rag-agent
export PYTHONPATH=. HF_HUB_OFFLINE=1
D=results/mh_arms
H=results/mh_interim200
leg () { arm=$1 tag=$2; out="$H/${tag}_answer_doc_qwen3_8b_cot_200.jsonl"
  if [ -f "${out%.jsonl}.json" ]; then echo "##### ${arm} already complete"; return; fi
  echo "##### ${arm} $(date '+%m-%d %H:%M:%S')"
  .venv/bin/python scripts/answer_accuracy_mh.py --scope doc --header-rule v2 \
    --reader "local:Qwen/Qwen3-8B?quantization=4bit" --prompt cot --max-tokens 384 \
    --records "$D/${tag}_records.jsonl" --condition retrieved \
    --same-contexts-as "$H/ref_contexts_${arm}_200.jsonl" --out "$out" --resume >> "${out%.jsonl}.log" 2>&1
  echo "exit=$? $(date '+%m-%d %H:%M:%S')"; }
leg chunk mh_chunk_hv2
leg huawei mh_huawei_hv2
leg rowcol mh_rowcol_hv2
leg tablerag mh_tablerag_allobj_path_hv2
echo "##### analyze $(date '+%m-%d %H:%M:%S')"
.venv/bin/python $H/analyze_arms.py --out $H/interim200_arms_qwen3_8b_cot.jsonl > $H/analyze_arms.log 2>&1
echo "exit=$? $(date '+%m-%d %H:%M:%S')"
echo ARMS_DONE
