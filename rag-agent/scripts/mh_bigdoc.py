#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""큰 문서 조건 (PREREG-2026-09-29-bigdoc.md): 답변 200문항의 문서마다 다른 문서 N-1개의 표를 이어 붙인다.

검색·답변 코드는 그대로 두고, 두 스크립트가 쓰는 `load_population` 만 감싼다.

  PYTHONPATH=. .venv/bin/python scripts/mh_bigdoc.py --stage retrieve -- <mh_arms.py 인자>
  PYTHONPATH=. .venv/bin/python scripts/mh_bigdoc.py --stage answer   -- <answer_accuracy_mh.py 인자>
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import mh_arms  # noqa: E402

IDS = ROOT / "results/mh_only_20260928/baselines/ids200.jsonl"


def merged_loader(n_docs: int, seed: str):
    orig = mh_arms.load_population

    def load_population(split, keep_hybrid=False):
        queries, docs, skipped = orig(split, keep_hybrid)
        ids = [json.loads(l)["query_id"] for l in IDS.open(encoding="utf-8") if l.strip()]
        assert set(ids) <= set(docs)
        pool = sorted(docs)
        merged = {}
        for uid in ids:
            rng = random.Random(f"{seed}:{uid}")
            others, seen = [], [tuple(docs[uid][0])]
            while len(others) < n_docs - 1:
                u = rng.choice(pool)
                if u != uid and tuple(docs[u][0]) not in seen:     # 같은 보고서(표 집합 동일) 제외
                    others.append(u); seen.append(tuple(docs[u][0]))
            tabs = list(docs[uid][0]) + [t for u in others for t in docs[u][0]]   # 원래 표가 앞 → 정답 좌표 불변
            merged[uid] = (tabs, docs[uid][1], docs[uid][2])
        queries = [q for q in queries if q["uid"] in merged]
        print(f"[bigdoc] {len(merged)} docs × {n_docs} (표 평균 {sum(len(d[0]) for d in merged.values()) / len(merged):.1f})", flush=True)
        return queries, merged, skipped

    return load_population


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["retrieve", "answer", "langchain"])
    ap.add_argument("--n-docs", type=int, default=5)
    ap.add_argument("--seed", default="20260929")
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    rest = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
    loader = merged_loader(a.n_docs, a.seed)
    mh_arms.load_population = loader
    if a.stage == "retrieve":
        sys.argv = ["mh_arms.py", *rest]
        return mh_arms.main()
    if a.stage == "langchain":
        # 비교군 LangChain 청크: 같은 합친 문서. 질의 집합 대조 파일·문항 목록·출력 폴더만 이 실험 것으로 바꾼다.
        import importlib.util
        spec = importlib.util.spec_from_file_location("langchain_chunk_baseline", ROOT / "scripts/baselines/langchain_chunk.py")
        L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
        L.load_population = loader
        L.S3C = ROOT / "results/bigdoc_20260929/bigdoc_s3c_records.jsonl"
        L.SUBSET = IDS
        L.OUT = ROOT / "results/bigdoc_20260929/langchain_chunk"
        sys.argv = ["langchain_chunk.py", *rest]
        return L.main()
    import answer_accuracy_mh  # noqa: E402
    answer_accuracy_mh.load_population = loader
    sys.argv = ["answer_accuracy_mh.py", *rest]
    return answer_accuracy_mh.main()


if __name__ == "__main__":
    raise SystemExit(main())
