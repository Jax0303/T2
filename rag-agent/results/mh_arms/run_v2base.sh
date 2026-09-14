set -u
cd /home/user/T2-1/rag-agent
export PYTHONPATH=.
SAME=results/mh_arms/mh_cell_answer_doc.jsonl
ret () { tag=$1; shift; echo "##### retrieval ${tag} $(date +%H:%M:%S)"
  .venv/bin/python scripts/mh_arms.py "$@" --header-rule v2 --label-rule none --tag "mh_${tag}" --out-dir results/mh_arms \
    > "results/mh_arms/mh_${tag}.log" 2>&1; echo "exit=$? $(date +%H:%M:%S)"; }
ans () { tag=$1; echo "##### answer ${tag} $(date +%H:%M:%S)"
  .venv/bin/python scripts/answer_accuracy_mh.py --records "results/mh_arms/mh_${tag}_records.jsonl" \
    --scope doc --condition retrieved --header-rule v2 --same-queries-as $SAME \
    --out "results/mh_arms/mh_${tag}_answer_doc.jsonl" > "results/mh_arms/mh_${tag}_answer_doc.log" 2>&1
  echo "exit=$? $(date +%H:%M:%S)"; }
# 공정 비교 기준 조건: 헤더 v2, 라벨 없음 — 비교군 검색 + 답변
ret chunk_hv2 --unit chunk --template s3c --chunk-chars 1000
ret huawei_hv2 --unit trag_hetero --template s3c --chunk-chars 1000 --chunk-overlap 200
ret rowcol_hv2 --unit rowcol --template s3c --row-text values
ret tablerag_allobj_path_hv2 --unit tablerag --tablerag-dtype all_object --tablerag-colmode path
for t in chunk_hv2 huawei_hv2 rowcol_hv2 tablerag_allobj_path_hv2; do ans $t; done
echo V2BASE_DONE
echo "##### pilot $(date +%H:%M:%S)"
.venv/bin/python scripts/reader_cot_pilot.py --split validation --n 60 \
  --out results/mh_arms/reader_cot_pilot_validation.jsonl > results/mh_arms/reader_cot_pilot_validation.log 2>&1
echo "exit=$? $(date +%H:%M:%S)"
echo "##### report $(date +%H:%M:%S)"
.venv/bin/python analysis/mh_final_report.py > results/mh_arms/report.log 2>&1
echo "exit=$? $(date +%H:%M:%S)"
echo FINAL_DONE
# 사용자 요청: 끝나면 PC 종료. 10분 여유 (취소: shutdown.exe /a)
/mnt/c/Windows/System32/shutdown.exe /s /t 600 /c "rag-agent experiments finished" && echo "shutdown scheduled $(date +%H:%M:%S)"
