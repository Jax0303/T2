#!/usr/bin/env bash
# PREREG-2026-09-28-mh-only.md §3.5 "답변 대 k": dev 332, 본 방법, k∈{5,10,20,50}, 리더 1회씩. 규칙 선택 전용(dev).
# 검색은 예산 50 기록 하나(mh_dev_s3c_b50)에서 앞 k 셀을 잘라 만든다(셀 arm: 단위 = 셀, 순위 순). 재검색 없음.
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
D=results/mh_only_20260928; R=$D/retrieval; P=results/reader_format_20260927/mh_dev_pop332.jsonl
while ! grep -q ALL_DONE $D/run_retrieval.log 2>/dev/null; do sleep 20; done
mkdir -p $D/dev_k
.venv/bin/python - <<'EOF'
import json, hashlib
from pathlib import Path
R = Path("results/mh_only_20260928/retrieval"); O = Path("results/mh_only_20260928/dev_k")
meta = json.loads((R / "mh_dev_s3c_b50.json").read_text())
recs = [json.loads(l) for l in open(R / "mh_dev_s3c_b50_records.jsonl", encoding="utf-8")]
def digest(x): return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
for k in (5, 10, 20, 50):
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
    m = dict(meta, budget_cells=k, derived_from="mh_dev_s3c_b50_records.jsonl (앞 k 셀)", records_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    (O / f"dev_s3c_k{k}.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
    print("wrote", out)
EOF
echo "[start] $(date -Is) commit $(git rev-parse --short HEAD)"
for k in 5 10 20 50; do
  .venv/bin/python scripts/answer_accuracy_mh.py --records $D/dev_k/dev_s3c_k${k}_records.jsonl --scope doc --condition retrieved \
    --split validation --header-rule v3.3u --batch-size 128 --same-queries-as $P --quiet-accuracy \
    --out $D/dev_k/dev_s3c_k${k}_answer.jsonl > $D/dev_k/dev_s3c_k${k}_answer.log 2>&1
  echo "[end] $(date -Is) k=$k rc=$?"
done
echo ALL_DONE
