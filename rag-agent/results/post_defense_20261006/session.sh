#!/usr/bin/env bash
# PREREG-2026-10-06-post-defense.md — 4시간 세션 실행기.
#   bash results/post_defense_20261006/session.sh [시간=4]
# jobs.txt 를 위에서부터 읽어 done.txt·failed.txt 에 없는 작업을 마감까지 돌린다. 마감에 걸린 작업은 SIGINT 로 끊고(문항 단위 저장),
# 다음 세션이 같은 인자(--resume)로 잇는다. rc=0 이면 done.txt, 그 밖(마감 124 제외)은 failed.txt 에 적고 다음 작업으로 넘어간다.
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONHASHSEED=0
export P=.venv/bin/python S=results/post_defense_20261006 B=results/mh_only_20260928/baselines
export IDS=results/mh_only_20260928/baselines/ids200.jsonl
export MAIN="local:Qwen/Qwen3-8B?quantization=4bit" L2="local:Qwen/Qwen3-1.7B?quantization=none" L3="local:meta-llama/Llama-3.2-3B-Instruct?quantization=none"
H=${1:-4}; deadline=$(( $(date +%s) + H*3600 ))
mkdir -p $S/logs; touch $S/done.txt $S/failed.txt
log=$S/session_$(date +%Y%m%d_%H%M).log
exec > >(tee -a "$log") 2>&1
echo "[session] $(date -Is) hours=$H commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
while IFS=$'\t' read -r name cmd; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  grep -qx "$name" $S/done.txt && continue
  grep -qx "$name" $S/failed.txt && { echo "[skip] $name (failed.txt — 지우면 다시 돈다)"; continue; }
  left=$(( deadline - $(date +%s) ))
  (( left < 300 )) && { echo "[session] 마감 — 다음 세션은 $name 부터"; break; }
  echo "[start] $(date -Is) $name (남은 $((left / 60))분)"; t=$(date +%s)
  echo "[start] $(date -Is) $cmd" >> $S/logs/$name.log
  flock .gpu.lock timeout -s INT -k 120 ${left}s bash -c "exec $cmd" >> $S/logs/$name.log 2>&1; rc=$?
  echo "[end] $(date -Is) $name rc=$rc $(( $(date +%s) - t ))s"
  if [ $rc = 0 ]; then echo "$name" >> $S/done.txt
  elif [ $rc = 124 ]; then echo "[cut] $name 마감에 끊김 — 다음 세션에서 --resume 으로 잇는다"; break
  else echo "[fail] $name rc=$rc (logs/$name.log) — failed.txt 에 적고 다음 작업"; echo "$name" >> $S/failed.txt; fi
done < $S/jobs.txt
echo "[session] $(date -Is) end"
