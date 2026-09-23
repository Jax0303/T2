#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""이미 생성된 MultiHiertt 답변을 다시 채점한다 — 생성은 하지 않는다 (2026-09-23).

두 가지를 고친 채점이다:
  1. 정답: bevaya 판의 잘린 값 대신 공식 릴리스 값 (`mh_arms.official_answers`).
  2. 채점기: 공식 `evaluate.py` 의 문항 유형 갈래 그대로 (`multihiertt_em.mh_exact_match`).
보조 지표로 DocMath-Eval 비교(`docmath_match`)를 같이 싣고, 보조 지표만 통과시킨 행을 전부 덤프한다
(CLAUDE.md §5: 느슨한 채점이 오답을 통과시킨 전례 — 사람이 확인할 수 있게).

  PYTHONPATH=. .venv/bin/python scripts/mh_rescore_official.py results/mh_arms/*_answer_*.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from mh_arms import OFFICIAL_REPO, OFFICIAL_REV, official_answers        # noqa: E402
from rag_agent.eval.artifacts import file_digest                          # noqa: E402
from rag_agent.eval.multihiertt_em import docmath_match, mh_exact_match  # noqa: E402


def rescore(path: Path, answers: dict) -> dict:
    rows = [json.loads(l) for l in path.open(encoding="utf-8")]
    by, extra = defaultdict(lambda: defaultdict(int)), []
    for r in rows:
        prog, gold = r["kind"] == "arith", answers[r["query_id"]]
        old = int(r["answer_correct"])
        new = int(mh_exact_match(r["pred"], gold, prog))
        doc = int(docmath_match(r["pred"], gold, prog))
        ceil = int(mh_exact_match(str(gold), gold, prog))
        for k in (r["layer"], "ALL"):
            b = by[k]
            b["n"] += 1
            b["old"] += old
            b["official"] += new
            b["docmath"] += doc
            b["official_ceiling"] += ceil
            b["gain"] += new and not old
            b["loss"] += old and not new
        if doc and not new:
            extra.append({"query_id": r["query_id"], "layer": r["layer"], "question": r["question"],
                          "pred": r["pred"], "official_answer": gold, "bevaya_answer": r["answer"]})
    out = {}
    for k, b in sorted(by.items()):
        d = b["gain"] + b["loss"]
        out[k] = {"n": b["n"], **{f"em_{m}": round(b[m] / b["n"], 4)
                                  for m in ("old", "official", "docmath", "official_ceiling")},
                  "old_to_official_gain": b["gain"], "old_to_official_loss": b["loss"],
                  "mcnemar_p_old_vs_official": binomtest(b["gain"], d, 0.5).pvalue if d else 1.0}
    return {"source": str(path), "source_sha256": file_digest(path), "by_layer": out,
            "docmath_only_rows": extra}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("answers", nargs="+", type=Path)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/mh_rescore_20260923")
    a = ap.parse_args()
    answers = official_answers(a.split)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for p in a.answers:
        res = {"answer_source": f"{OFFICIAL_REPO}@{OFFICIAL_REV}", "split": a.split, **rescore(p, answers)}
        dst = a.out_dir / f"{p.parent.name}__{p.stem}.json"
        if dst.exists():
            raise SystemExit(f"{dst} exists — 덮어쓰지 않는다")
        dst.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        al = res["by_layer"]["ALL"]
        print(f"{p.parent.name}/{p.stem}: old {al['em_old']} -> official {al['em_official']} "
              f"(+{al['old_to_official_gain']} -{al['old_to_official_loss']}), docmath {al['em_docmath']}, "
              f"ceiling {al['em_official_ceiling']}, docmath-only {len(res['docmath_only_rows'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
