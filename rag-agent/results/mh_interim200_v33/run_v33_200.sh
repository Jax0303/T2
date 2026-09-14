set -u
# results/mh_interim200_v33/PLAN.md — 헤더 규칙 v3.3, 본 방법·chunk. 정정 4 채택 기준 통과 뒤 사용자 지시(2026-09-14)로 실행한다.
# 중단되면 같은 명령으로 다시 돌린다: 완료된 레그는 건너뛰고, 진행 중이던 답변 레그는 --resume 으로 이어 쓴다.
#   bash results/mh_interim200_v33/run_v33_200.sh >> results/mh_interim200_v33/run_v33_200.log 2>&1
cd /home/user/T2-1/rag-agent
export PYTHONPATH=. HF_HUB_OFFLINE=1
V=results/mh_interim200_v33
H=results/mh_interim200
# 검색: train 모집단 전체(2,908 질의). 인자는 v2 요약(mh_arms/mh_{cell,chunk}_hv2.json)과 header_rule·tag·out_dir 만 다르다.
ret () { arm=$1 tag=$2; shift 2
  if [ -f "$V/${tag}.json" ]; then echo "##### retrieval ${arm} already complete"; return; fi
  echo "##### retrieval ${arm} $(date '+%m-%d %H:%M:%S')"
  .venv/bin/python scripts/mh_arms.py "$@" --header-rule v3.3 --label-rule none --tag "$tag" --out-dir $V \
    > "$V/${tag}.log" 2>&1
  echo "exit=$? $(date '+%m-%d %H:%M:%S')"; }
# 답변: interim200 의 200 id. 문맥은 규칙 때문에 v2 와 달라지므로 --same-contexts-as 가 아니라 --same-queries-as.
ans () { arm=$1 tag=$2; out="$V/${tag}_answer_doc_qwen3_8b_cot_200.jsonl"
  if [ -f "${out%.jsonl}.json" ]; then echo "##### answer ${arm} already complete"; return; fi
  echo "##### answer ${arm} $(date '+%m-%d %H:%M:%S')"
  .venv/bin/python scripts/answer_accuracy_mh.py --scope doc --header-rule v3.3 \
    --reader "local:Qwen/Qwen3-8B?quantization=4bit" --prompt cot --max-tokens 384 \
    --records "$V/${tag}_records.jsonl" --condition retrieved \
    --same-queries-as "$H/ref_contexts_${arm}_200.jsonl" --out "$out" --resume >> "${out%.jsonl}.log" 2>&1
  echo "exit=$? $(date '+%m-%d %H:%M:%S')"; }
ret ours mh_cell_hv33 --unit cell --template s3c
ret chunk mh_chunk_hv33 --unit chunk --template s3c --chunk-chars 1000
ans ours mh_cell_hv33
ans chunk mh_chunk_hv33
echo "##### analyze $(date '+%m-%d %H:%M:%S')"
.venv/bin/python $V/analyze_v33.py --out $V/interim200_v33_qwen3_8b_cot.jsonl > $V/analyze_v33.log 2>&1
echo "exit=$? $(date '+%m-%d %H:%M:%S')"
echo V33_DONE
