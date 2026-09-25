# 2026-09-25 03:04 밤 실행: 본 방법 v3.3u 검색 → 답변(같은 882건) → randrow 이어 쓰기 → tablerag_leaf → 집계
set -u
cd /home/user/T2-1/rag-agent
export PYTHONPATH=. HF_HUB_OFFLINE=1
D=results/mh_arms/cap300_20260924
step() { echo "##### $1 start $(date +%F' '%T)"; }
done_() { echo "##### $1 exit=$2 $(date +%F' '%T)"; }

step retrieval_v33u
.venv/bin/python scripts/mh_arms.py --split train --unit cell --template s3c --header-rule v3.3u \
  --alpha 0.7 --budget 20 --tag mh_train_cell_hv3.3u_none_doc > $D/retrieval_v33u.log 2>&1
rc=$?; done_ retrieval_v33u $rc
[ $rc -eq 0 ] || { echo NIGHT_FAILED; exit 1; }

step cell_uniq
.venv/bin/python scripts/answer_accuracy_mh.py --scope doc --batch-size 128 --header-rule v3.3u \
  --same-queries-as $D/cell.jsonl --out $D/cell_uniq.jsonl \
  --records results/mh_arms/mh_train_cell_hv3.3u_none_doc_records.jsonl --condition retrieved > $D/cell_uniq.log 2>&1
done_ cell_uniq $?

for arm in randrow tablerag_leaf; do
  step $arm
  .venv/bin/python scripts/answer_accuracy_mh.py --scope doc --stratum-cap 300 --sample-seed 20260913 --batch-size 128 \
    --out $D/$arm.jsonl --records results/mh_arms/mh_train_${arm}_hv1_none_doc_records.jsonl \
    --condition retrieved --resume >> $D/$arm.log 2>&1
  done_ $arm $?
done

step report
.venv/bin/python scripts/mh_cap300_report.py > $D/report.log 2>&1
done_ report $?
echo NIGHT_DONE
