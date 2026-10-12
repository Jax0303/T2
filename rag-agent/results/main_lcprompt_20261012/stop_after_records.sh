#!/usr/bin/env bash
# RAPTOR 882 트리·기록(raptor_records.json)이 끝나면 raptor.py 와 run.sh 를 멈춘다(뒤 답변 단계는 지시문 결정으로 필요 없어짐).
S=/home/user/T2-1/rag-agent/results/main_lcprompt_20261012
until [ -f $S/raptor882/raptor_records.json ]; do sleep 15; done
for p in $(pgrep -f "results/main_lcprompt_20261012/run.sh"); do [ "$(ps -o comm= -p $p)" = bash ] && kill $p; done
for p in $(pgrep -f "scripts/baselines/raptor.py --limit 882"); do [ "$(ps -o comm= -p $p)" = python ] && kill $p; done
echo "[stop] $(date -Is) 기록 $(wc -l < $S/raptor882/raptor_records.jsonl)줄"
