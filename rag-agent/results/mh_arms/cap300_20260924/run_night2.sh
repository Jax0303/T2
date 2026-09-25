# 2026-09-25 03:3x: run_night.sh(PID 254451) 가 끝나면 사후 조건 (가) cell_hv33r, (나) 비교군 머리글 v3.3 → 집계
set -u
cd /home/user/T2-1/rag-agent
export PYTHONPATH=. HF_HUB_OFFLINE=1
D=results/mh_arms/cap300_20260924
while kill -0 254451 2>/dev/null; do sleep 60; done
step() { echo "##### $1 start $(date +%F' '%T)"; }
done_() { echo "##### $1 exit=$2 $(date +%F' '%T)"; }
ans() {  # 이름 레코드 머리글
  .venv/bin/python scripts/answer_accuracy_mh.py --scope doc --batch-size 128 --header-rule $3 \
    --same-queries-as $D/cell.jsonl --out $D/$1.jsonl --records $2 --condition retrieved > $D/$1.log 2>&1
}
step cell_hv33r; ans cell_hv33r results/mh_interim200_v33/mh_cell_hv33_records.jsonl v3.3; done_ cell_hv33r $?
for arm in rowcol tablerag_path tablerag_leaf; do
  case $arm in rowcol) u="--unit rowcol --embed-overflow truncate";;
               tablerag_path) u="--unit tablerag --tablerag-colmode path";;
               tablerag_leaf) u="--unit tablerag --tablerag-colmode leaf";; esac
  step retrieval_${arm}_hv33
  .venv/bin/python scripts/mh_arms.py --split train $u --template s3c --header-rule v3.3 --alpha 0.7 --budget 20 \
    --tag mh_train_${arm}_hv3.3_none_doc > $D/retrieval_${arm}_hv33.log 2>&1
  rc=$?; done_ retrieval_${arm}_hv33 $rc
  [ $rc -eq 0 ] || continue
  step ${arm}_hv33; ans ${arm}_hv33 results/mh_arms/mh_train_${arm}_hv3.3_none_doc_records.jsonl v3.3; done_ ${arm}_hv33 $?
done
step report2
.venv/bin/python scripts/mh_cap300_report.py > $D/report2.log 2>&1
done_ report2 $?
echo NIGHT2_DONE
