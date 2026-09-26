#!/usr/bin/env bash
# 2026-09-26 PREREG-2026-09-26-rerun-dev-alpha.md — 항목 1(주 결과표 재실행)·항목 2(dev 에서 라벨 섞기 비율 선택).
# 리더 생성 없음. 이미 있는 출력은 건너뛴다(스크립트가 덮어쓰기를 거부). 실행: setsid nohup bash results/rerun_20260926/run.sh
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
PY=.venv/bin/python
R1=results/rerun_20260926; R2=results/dev_alpha_20260926
C1=.cache/rerun_20260926; C1M=.cache/rerun_20260926_mh; C2=.cache/dev_alpha_20260926
mkdir -p $R1/hitab $R1/mh $R2/hitab_dev $R2/mh_dev $R2/apply $C1 $C1M $C2
echo "[start] $(date -Is) commit $(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"

run() {  # run <out_dir> <tag> <script> <args...>
  local out=$1 tag=$2 script=$3; shift 3
  if [ -e "$out/$tag.json" ]; then echo "[skip] $tag"; return 0; fi
  echo "[run] $(date -Is) $tag"
  $PY scripts/$script "$@" --out-dir "$out" --tag "$tag" > "$out/$tag.log" 2>&1
  local rc=$?; echo "[end] $(date -Is) $tag rc=$rc"
}

# ---- 항목 1: HiTab test, 질문의 표 안(gold) / 538표(split)
hitab_arm() {  # hitab_arm <arm> -> retrieval_accuracy.py 인자
  case $1 in
    s3c)           echo "--template s3c --unit cell" ;;
    sleaf)         echo "--template sleaf --unit cell" ;;
    table)         echo "--template s3c --unit table --row-text values --embed-overflow truncate" ;;
    row)           echo "--template s3c --unit row --row-text values --embed-overflow truncate" ;;
    chunk)         echo "--template s3c --unit chunk --embed-overflow truncate" ;;
    trag_hetero)   echo "--template s3c --unit trag_hetero --embed-overflow truncate" ;;
    tablerag_leaf) echo "--template s3c --unit tablerag --tablerag-colmode leaf" ;;
    tablerag_path) echo "--template s3c --unit tablerag --tablerag-colmode path" ;;
    rowcol)        echo "--template s3c --unit rowcol --row-text values" ;;
    randrow)       echo "--template s3c --unit randrow --row-text values" ;;
  esac
}
ARMS="s3c sleaf table row chunk trag_hetero tablerag_leaf tablerag_path rowcol randrow"
HB="--split test --alpha 0.7 --budget 20 --embed-model BAAI/bge-base-en-v1.5 --cache-dir $C1"
for scope in gold split; do
  for arm in $ARMS; do
    [ "$arm $scope" = "randrow split" ] && continue   # 코드가 거부: randrow 는 질문의 표 안에서만 정의
    run $R1/hitab "hitab_test_${scope}_${arm}" retrieval_accuracy.py $HB --corpus $scope $(hitab_arm $arm)
  done
done

# ---- 항목 2: dev 후보 (a<α> = 라벨 섞기 비율 α, prefix = 라벨을 문장 접두어로)
CANDS="1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 prefix"
hitab_cand() { [ "$1" = prefix ] && echo "--template s3c" || echo "--template s2 --label-mix $1"; }
mh_cand()    { [ "$1" = prefix ] && echo "--label-rule L1" || echo "--label-rule none --label-mix $1"; }
cname()      { [ "$1" = prefix ] && echo prefix || echo "a$1"; }
for scope in gold split; do
  for c in $CANDS; do
    run $R2/hitab_dev "hitab_dev_${scope}_$(cname $c)" retrieval_accuracy.py --split dev --corpus $scope --unit cell \
      --alpha 0.7 --budget 20 --embed-model BAAI/bge-base-en-v1.5 --cache-dir $C2 $(hitab_cand $c)
  done
done
MB="--header-rule v3.3u --alpha 0.7 --budget 20 --embed-model BAAI/bge-base-en-v1.5"
for c in $CANDS; do
  run $R2/mh_dev "mh_dev_$(cname $c)" mh_arms.py --split validation --keep-hybrid --unit cell --template s3c \
    $MB --cache-dir $C2 $(mh_cand $c)
done

# ---- 항목 1: MultiHiertt train, 최종 머리글(v3.3u), 문서 안
mh_arm() {
  case $1 in
    s3c)           echo "--unit cell --template s3c" ;;
    sleaf)         echo "--unit cell --template sleaf" ;;
    table)         echo "--unit table --template s3c --embed-overflow truncate" ;;
    row)           echo "--unit row --template s3c --row-text values --embed-overflow truncate" ;;
    chunk)         echo "--unit chunk --template s3c --embed-overflow truncate" ;;
    trag_hetero)   echo "--unit trag_hetero --template s3c --embed-overflow truncate" ;;
    tablerag_leaf) echo "--unit tablerag --template s3c --tablerag-colmode leaf" ;;
    tablerag_path) echo "--unit tablerag --template s3c --tablerag-colmode path" ;;
    rowcol)        echo "--unit rowcol --template s3c --row-text values --embed-overflow truncate" ;;
    randrow)       echo "--unit randrow --template s3c --row-text values" ;;
  esac
}
for arm in $ARMS; do
  run $R1/mh "mh_train_${arm}" mh_arms.py --split train $MB --cache-dir $C1M $(mh_arm $arm)
done

# ---- 항목 2: 선택 → test(HiTab)·train(MH)에 한 번만 적용
$PY $R2/select_alpha.py > $R2/select_alpha.log 2>&1 || { echo "[fail] select_alpha"; exit 1; }
while read -r ds scope c; do
  if [ "$ds" = hitab ]; then
    run $R2/apply "hitab_test_${scope}_$(cname $c)" retrieval_accuracy.py --split test --corpus $scope --unit cell \
      --alpha 0.7 --budget 20 --embed-model BAAI/bge-base-en-v1.5 --cache-dir $C1 $(hitab_cand $c)
  else
    run $R2/apply "mh_train_$(cname $c)" mh_arms.py --split train --unit cell --template s3c $MB --cache-dir $C1M $(mh_cand $c)
  fi
done < $R2/apply_list.txt
echo "[done] $(date -Is)"
