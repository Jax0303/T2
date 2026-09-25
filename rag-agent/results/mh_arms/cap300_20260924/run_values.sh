# 2026-09-25: rowcol·randrow 를 발표된 단위(values)로 다시 — 검색 → 답변(같은 882건) → 집계
set -u
cd /home/user/T2-1/rag-agent
export PYTHONPATH=. HF_HUB_OFFLINE=1
D=results/mh_arms/cap300_20260924
for arm in rowcol randrow; do
  extra=""; [ $arm = rowcol ] && extra="--embed-overflow truncate"
  echo "##### retrieval_${arm}_values start $(date +%F' '%T)"
  .venv/bin/python scripts/mh_arms.py --split train --unit $arm --row-text values --template s3c --header-rule v1 \
    --alpha 0.7 --budget 20 $extra --tag mh_train_${arm}_values_hv1_none_doc > $D/retrieval_${arm}_values.log 2>&1
  rc=$?; echo "##### retrieval_${arm}_values exit=$rc $(date +%F' '%T)"; [ $rc -eq 0 ] || continue
  echo "##### ${arm}_values start $(date +%F' '%T)"
  .venv/bin/python scripts/answer_accuracy_mh.py --scope doc --batch-size 128 --header-rule v1 \
    --same-queries-as $D/cell.jsonl --out $D/${arm}_values.jsonl \
    --records results/mh_arms/mh_train_${arm}_values_hv1_none_doc_records.jsonl --condition retrieved > $D/${arm}_values.log 2>&1
  echo "##### ${arm}_values exit=$? $(date +%F' '%T)"
done
.venv/bin/python scripts/mh_cap300_report.py > $D/report3.log 2>&1
echo "##### report3 exit=$? $(date +%F' '%T)"
echo VALUES_DONE
