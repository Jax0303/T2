#!/usr/bin/env bash
# PREREG-2026-09-28-mh-only.md "k 선택(dev)": dev 332 답변 k 추가 {15,30}; 5~50 중 최고가 50 이면 {75,100} 추가(dev 예산 100 재검색 1회).
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
D=results/mh_only_20260928; R=$D/retrieval; P=results/reader_format_20260927/mh_dev_pop332.jsonl
until grep -q "k=50 rc" $D/dev_k_answer.log; do sleep 20; done
cut_k() {  # $1 = 원본 기록 태그, 나머지 = k 들. dev_k_answer.sh 와 같은 방식으로 앞 k 셀을 자른다.
.venv/bin/python - "$@" <<'PY'
import json, hashlib, sys
from pathlib import Path
R = Path("results/mh_only_20260928/retrieval"); O = Path("results/mh_only_20260928/dev_k")
src, ks = sys.argv[1], [int(x) for x in sys.argv[2:]]
meta = json.loads((R / f"{src}.json").read_text())
recs = [json.loads(l) for l in open(R / f"{src}_records.jsonl", encoding="utf-8")]
def digest(x): return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
for k in ks:
    out = O / f"dev_s3c_k{k}_records.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in recs:
            if "excluded" in r or "doc" not in r:
                f.write(json.dumps(r, ensure_ascii=False) + "\n"); continue
            d = dict(r["doc"]); ctx = d["context"][:k]; units = d["ranked_units"][:k]
            got = {c for u in units for c in u}
            d.update(context=ctx, context_sha256=digest(ctx), cells_in_context=len(got),
                     correct=int(set(r["gold_ids"]) <= got), any_DIAGNOSTIC=int(bool(set(r["gold_ids"]) & got)))
            f.write(json.dumps({**{kk: v for kk, v in r.items() if kk not in ("corpus", "table")}, "doc": d}, ensure_ascii=False) + "\n")
    m = dict(meta, budget_cells=k, derived_from=f"{src}_records.jsonl (앞 k 셀)", records_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    (O / f"dev_s3c_k{k}.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
    print("wrote", out)
PY
}
answer() { for k in "$@"; do
  .venv/bin/python scripts/answer_accuracy_mh.py --records $D/dev_k/dev_s3c_k${k}_records.jsonl --scope doc --condition retrieved \
    --split validation --header-rule v3.3u --batch-size 128 --same-queries-as $P --quiet-accuracy \
    --out $D/dev_k/dev_s3c_k${k}_answer.jsonl > $D/dev_k/dev_s3c_k${k}_answer.log 2>&1
  echo "[end] $(date -Is) k=$k rc=$?"; done; }
echo "[start] $(date -Is) commit $(git rev-parse --short HEAD)"
cut_k mh_dev_s3c_b50 15 30
answer 15 30
best=$(.venv/bin/python -c "
import json
acc={k:sum(json.loads(l)['answer_correct'] for l in open(f'$D/dev_k/dev_s3c_k{k}_answer.jsonl')) for k in (5,10,15,20,30,50)}
print(max(sorted(acc), key=lambda k:(acc[k],-k)))")
echo "best_5_50=$best"
if [ "$best" = 50 ]; then
  .venv/bin/python scripts/mh_arms.py --split validation --header-rule v3.3u --alpha 0.7 --budget 100 --dump-ranked 100 \
    --embed-model BAAI/bge-base-en-v1.5 --k-ladder 1,5,10,20,50,75,100 --cache-dir .cache/rerun_20260926_mh \
    --unit cell --template s3c --out-dir $R --tag mh_dev_s3c_b100 > $R/mh_dev_s3c_b100.log 2>&1
  cut_k mh_dev_s3c_b100 75 100
  answer 75 100
fi
echo ALL_DONE
