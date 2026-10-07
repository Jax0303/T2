#!/usr/bin/env bash
# PREREG-2026-10-07-claims.md 5 — Qwen3-8B 양자화 없음(bf16, 24GB GPU 임대). 저장소 루트에서 실행.
set -e; cd "$(dirname "$0")/../.."
export PYTHONPATH=. PYTHONHASHSEED=0
P=.venv/bin/python S=results/post_defense_20261006/bf16_8b B=results/mh_only_20260928/baselines IDS=results/mh_only_20260928/baselines/ids200.jsonl
R="local:Qwen/Qwen3-8B?quantization=none"; mkdir -p $S
$P scripts/answer_accuracy_mh.py --records results/mh_arms/mh_train_cell_hv3.3u_none_doc_records.jsonl --scope doc --condition retrieved --split train --header-rule v3.3u --batch-size 32 --same-contexts-as $IDS --reader $R --quiet-accuracy --out $S/ours_answer.jsonl --resume
$P scripts/baselines/langchain_chunk.py --stage answer --limit 200 --out-dir $S/langchain_chunk --records $B/langchain_chunk/langchain_chunk_records.jsonl --reader $R --resume
$P scripts/baselines/tablerag_yu.py --out-dir $S/tablerag_yu --stage answer --answer-ids $IDS --reader $R --resume
$P scripts/baselines/raptor.py --out-dir $S/raptor --limit 200 --reader $R --resume
