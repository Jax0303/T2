"""PREREG-2026-10-06-post-defense.md A: dev 332 예산 50 기록(mh_dev_s3c_b50)에서 앞 k 셀을 잘라 k=1·3 레코드를 만든다.
results/mh_only_20260928/dev_k2.sh 의 cut_k 와 같은 방식(재검색 없음, 리더 없음)."""
import hashlib, json
from pathlib import Path

R = Path("results/mh_only_20260928/retrieval"); O = Path("results/post_defense_20261006/dev_k")
O.mkdir(parents=True, exist_ok=True)
meta = json.loads((R / "mh_dev_s3c_b50.json").read_text())
recs = [json.loads(l) for l in open(R / "mh_dev_s3c_b50_records.jsonl", encoding="utf-8")]
digest = lambda x: hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
for k in (1, 3):
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
    n = sum(1 for r in recs if "doc" in r and "excluded" not in r)
    print("wrote", out, "queries", n)
