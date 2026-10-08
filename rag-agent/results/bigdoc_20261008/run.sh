#!/usr/bin/env bash
# PREREG-2026-10-08-bigdoc-scale.md — 합친 문서 25·40개(+5개 청크 cot). flock 으로 GPU 독점.
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONHASHSEED=0
P=.venv/bin/python; D=results/bigdoc_20261008; IDS=results/mh_only_20260928/baselines/ids200.jsonl
ARMS="--split train --header-rule v3.3u --alpha 0.7 --budget 20 --embed-model BAAI/bge-base-en-v1.5 --k-ladder 1,5,10,20 --dump-ranked 20 --unit cell --template s3c"
ANS="--scope doc --condition retrieved --split train --header-rule v3.3u --batch-size 128 --same-queries-as $IDS --quiet-accuracy"
run() { local name=$1; shift; local t=$(date +%s); echo "[start] $(date -Is) $name"; "$@" > $D/logs/$name.log 2>&1; echo "[end] $(date -Is) $name rc=$? $(( $(date +%s) - t ))s"; }
mkdir -p $D/logs $D/n25 $D/n40 $D/n5
echo "[run] $(date -Is) commit $(git rev-parse --short HEAD)"
for N in 25 40; do
  run n${N}_measure $P - <<PY
import json, sys, statistics as st
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import mh_bigdoc, answer_accuracy_mh as A
from transformers import AutoTokenizer
ids = [json.loads(l)["query_id"] for l in open("$IDS")]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
A.load_population = mh_bigdoc.merged_loader($N, "20260929"); full = A.full_tables("train", "v3.3u", "none")
toks = [len(tok("\n".join(full[q]), add_special_tokens=False)["input_ids"]) for q in ids]; ntab = [len(full[q]) for q in ids]
json.dump({"n_docs": $N, "tables_median": st.median(ntab), "tables_mean": st.mean(ntab), "fulltable_tokens_median": st.median(toks), "fulltable_tokens_mean": st.mean(toks), "fulltable_tokens_max": max(toks), "over_40000": sum(t > 40000 for t in toks), "over_40576": sum(t > 40960 - 384 for t in toks)}, open("$D/n$N/measure.json", "w"), indent=1)
print(open("$D/n$N/measure.json").read())
PY
  run n${N}_ours_retrieve $P scripts/mh_bigdoc.py --stage retrieve --n-docs $N -- $ARMS --cache-dir .cache/bigdoc${N}_20261008 --out-dir $D/n$N --tag bigdoc${N}_s3c
  run n${N}_ours_answer $P scripts/mh_bigdoc.py --stage answer --n-docs $N -- --records $D/n$N/bigdoc${N}_s3c_records.jsonl $ANS --out $D/n$N/ours_answer.jsonl --resume
  run n${N}_lc_retrieve $P scripts/mh_bigdoc.py --stage langchain --n-docs $N -- --stage retrieve --out-dir $D/n$N/langchain_chunk
  run n${N}_lc_cot $P scripts/mh_bigdoc.py --stage langchain --n-docs $N -- --stage answer --prompt cot --out-dir $D/n$N/langchain_cot --records $D/n$N/langchain_chunk/langchain_chunk_records.jsonl --resume
  run n${N}_lc_tut $P scripts/mh_bigdoc.py --stage langchain --n-docs $N -- --stage answer --out-dir $D/n$N/langchain_tut --records $D/n$N/langchain_chunk/langchain_chunk_records.jsonl --resume
  if [ $N = 25 ]; then
    run n5_lc_cot $P scripts/mh_bigdoc.py --stage langchain --n-docs 5 -- --stage answer --prompt cot --out-dir $D/n5/langchain_cot --records results/bigdoc_20260929/langchain_chunk/langchain_chunk_records.jsonl --resume
  fi
done
echo ALL_DONE
