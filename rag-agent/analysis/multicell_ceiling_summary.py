#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Score ``results/multicell/ceiling_records.jsonl`` and write the ceiling table.

Scored with ``phase4_summary.em`` -- the same function Phase 4's numbers use,
including R1 (relative tolerance < 0.01 on numeric answers) and the multi-gold
set rule (every element matched, none extra, order-insensitive).

  PYTHONPATH=. .venv/bin/python analysis/multicell_ceiling_summary.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from phase4_summary import em, gold_parts                           # noqa: E402

SRC = Path("results/multicell/ceiling_records.jsonl")
OUT_MD = Path("results/multicell/CEILING.md")
OUT_JSON = Path("results/multicell/ceiling.json")
# cited, not re-measured: results/phase4/FINAL.md, gold_cell / hitab_lookup
SINGLE_CELL = (0.9206, 189)


def main() -> int:
    rows = [json.loads(l) for l in open(SRC)]
    for r in rows:
        r["em"] = em(r["pred_parsed"], r["gold_answer"])

    by_split = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)

    out = {"source": str(SRC), "n_records": len(rows),
           "scorer": "analysis/phase4_summary.em (R1, multi-gold set rule)",
           "condition": "gold_cell, answer cells injected, no retrieval",
           "cited_single_cell_ceiling": {"em": SINGLE_CELL[0], "n": SINGLE_CELL[1],
                                         "source": "results/phase4/FINAL.md"},
           "splits": {}, "by_cell_count": {}, "pooled": {}}

    L = ["# 다중 셀 `gold_cell` 상한 — MEASURED 2026-09-01", "",
         "조건: 검색 생략, `[ANSWER]` 셀 문장만 주입. 리더 Qwen2.5-7B-Instruct "
         f"(rev pinned), 4-bit NF4, temp=0, seed=42, max_new_tokens=32.",
         "채점 `analysis/phase4_summary.em` (R1, 다중 gold 세트 규칙).", "",
         "## 상한", "", "| 모집단 | n | EM | 맞음 | 틀림 |", "|---|---:|---:|---:|---:|"]
    for split in ("dev", "test"):
        rs = by_split.get(split, [])
        if not rs:
            continue
        k = sum(r["em"] for r in rs)
        out["splits"][split] = {"n": len(rs), "em": round(k / len(rs), 4), "correct": k}
        L.append(f"| `hitab_{split}_lookup_multi` | {len(rs)} | "
                 f"**{k/len(rs):.4f}** | {k} | {len(rs)-k} |")
    k = sum(r["em"] for r in rows)
    out["pooled"] = {"n": len(rows), "em": round(k / len(rows), 4), "correct": k}
    L.append(f"| 합산 | {len(rows)} | **{k/len(rows):.4f}** | {k} | {len(rows)-k} |")
    L += ["", f"대조 (인용, 재측정 아님): 단일 셀 `hitab_lookup` gold_cell "
          f"= {SINGLE_CELL[0]:.4f} (n={SINGLE_CELL[1]}, `results/phase4/FINAL.md`).", ""]

    L += ["## 답 셀 수별", "", "| 답 셀 수 | n | EM | 맞음 |", "|---:|---:|---:|---:|"]
    per = defaultdict(list)
    for r in rows:
        per[r["n_answer_cells"]].append(r["em"])
    for m in sorted(per):
        v = per[m]
        out["by_cell_count"][str(m)] = {"n": len(v), "em": round(sum(v)/len(v), 4),
                                        "correct": sum(v)}
        L.append(f"| {m} | {len(v)} | {sum(v)/len(v):.4f} | {sum(v)} |")

    trunc = [r for r in rows if r.get("hit_token_cap")]
    ngold = Counter(len(gold_parts(r["gold_answer"])) for r in rows)
    out["hit_token_cap"] = len(trunc)
    out["gold_answer_parts_hist"] = {str(k2): v for k2, v in sorted(ngold.items())}
    wrong = [r for r in rows if not r["em"]]
    out["wrong"] = [{k: r[k] for k in ("split", "query_id", "n_answer_cells",
                                       "gold_answer", "pred_parsed")} for r in wrong]
    L += ["", "## 틀린 건 전부", "",
          "| split | query_id | 답 셀 | gold | pred |", "|---|---|---:|---|---|"]
    for r in wrong:
        L.append(f'| {r["split"]} | `{r["query_id"][:8]}` | {r["n_answer_cells"]} | '
                 f'`{r["gold_answer"]}` | `{r["pred_parsed"]}` |')

    L += ["", "## 기록해야 할 것", "",
          f"- max_new_tokens 상한에 닿은 생성: **{len(trunc)}/{len(rows)}**"
          + (f", 그중 채점 정답 {sum(r['em'] for r in trunc)}건 "
             f"(query_id: {[r['query_id'][:8] for r in trunc]})" if trunc else ""),
          f"- gold 답 문자열의 원소 수 분포: {dict(sorted(ngold.items()))}",
          "- 실행하지 않은 것: `P1_fixed_512` / `P4_path_cell` 레그. 이 모집단에 대한 "
          "검색 문맥이 없어 별도 검색 실행이 필요하다. 상한만으로 GATE 판정이 서므로 "
          "돌리지 않았다.",
          "- `operand` 주입 레그도 실행하지 않았다: 64건 전부에서 `[ANSWER]` 셀 집합과 "
          "`gold_operands` 집합이 **동일**해 중복이다 "
          "(`multicell_ceiling.py`가 매 실행마다 재확인한다).", ""]

    OUT_MD.write_text("\n".join(L) + "\n")
    json.dump(out, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
