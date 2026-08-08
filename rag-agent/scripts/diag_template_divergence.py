# SPDX-License-Identifier: MIT
"""MEASURE (do not unify) the three coexisting cell-sentence templates.

``RESEARCH_STRUCTURE.md`` §6 (재현성 부채) records three templates living side by
side. Unifying them would move every published number, so this script only sizes
the gap: it renders the same cells through all three and reports exact-match
rate, token Jaccard and length.

    (a) rag_agent/serialization/caption.py   — S3, deployed setting (long, cell)
    (b) rag_agent/serialize/verbalize.py     — "{col} for {row} is {v}."
    (c) scripts/operand_collision_multihiertt.py::cell_text — in-script copy

Writes diag/template_divergence.md. Reads nothing but HiTab tables; touches no
results/ file.
"""
from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_agent.bench.hitab import load_queries
from rag_agent.data.loader import load_table
from rag_agent.retrieve.encoders import _tokenize
from rag_agent.serialization.base import fmt_value as c_fmt
from rag_agent.serialization.caption import serialize as caption_serialize
from rag_agent.serialize.verbalize import _fmt as v_fmt, verbalize_cell
from rag_agent.stores.original_store import build_original_table

PAIRS = [("a_caption_long", "b_verbalize_long"),
         ("a_caption_long", "c_script_S3"),
         ("b_verbalize_long", "c_script_S3"),
         ("a_caption_medium", "c_script_S3")]


def _script_cell_text():
    """Import the in-script copy without running its main()."""
    path = Path(__file__).with_name("operand_collision_multihiertt.py")
    spec = importlib.util.spec_from_file_location("_collision_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.cell_text


def jaccard(x: str, y: str) -> float:
    a, b = set(_tokenize(x)), set(_tokenize(y))
    return len(a & b) / len(a | b) if (a or b) else 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--tables", type=int, default=20)
    ap.add_argument("--cells-per-table", type=int, default=40)
    ap.add_argument("--out", default="diag/template_divergence.md")
    args = ap.parse_args()

    cell_text = _script_cell_text()
    queries, _ = load_queries(args.data_dir, args.split)
    tids, seen = [], set()
    for q in queries:                                   # first N distinct, no RNG
        if q.gold_table_id not in seen:
            seen.add(q.gold_table_id)
            tids.append(q.gold_table_id)
        if len(tids) >= args.tables:
            break

    rows = []                                           # one dict per rendered cell
    for tid in tids:
        raw = load_table(tid, args.data_dir)
        if raw is None:
            continue
        t = build_original_table(raw)
        a_long = {(c.row_index, c.col_index): c.text
                  for c in caption_serialize(t, template="structural", granularity="cell")}
        a_med = {(c.row_index, c.col_index): c.text
                 for c in caption_serialize(t, template="mt2net", granularity="cell")}
        n = 0
        for (r, c), a_txt in a_long.items():
            v = t.cell(r, c)
            if v is None or str(v).strip() == "":        # verbalize skips blanks
                continue
            if n >= args.cells_per_table:
                break
            n += 1
            rows.append({
                "table": tid, "rc": (r, c), "value": v,
                "a_caption_long": a_txt,
                "a_caption_medium": a_med[(r, c)],
                "b_verbalize_long": verbalize_cell(t, r, c, "long"),
                "c_script_S3": cell_text({"row_path": list(t.row_path(r)),
                                          "col_path": list(t.col_path(c)),
                                          "value": v}, "S3"),
            })

    if not rows:
        print("no cells rendered — check --data-dir")
        return 1

    impls = ["a_caption_long", "a_caption_medium", "b_verbalize_long", "c_script_S3"]
    lines = ["# 셀 문장 템플릿 차이 (측정만, 통일 안 함)", "",
             f"표 {len(tids)}개 / 셀 {len(rows)}개, HiTab {args.split}. "
             f"생성: `scripts/diag_template_divergence.py`.", "",
             "구현: (a) `rag_agent/serialization/caption.py` (배포 설정 = S3/long/cell), "
             "(b) `rag_agent/serialize/verbalize.py`, "
             "(c) `scripts/operand_collision_multihiertt.py:174-198` 내부 사본.", "",
             "## 구현별 길이", "",
             "| 구현 | 평균 문자수 | 평균 토큰수 |", "|---|---|---|"]
    for k in impls:
        lines.append(f"| {k} | {statistics.mean(len(r[k]) for r in rows):.1f} | "
                     f"{statistics.mean(len(_tokenize(r[k])) for r in rows):.1f} |")

    lines += ["", "## 쌍별 차이", "",
              "| 쌍 | 완전일치 | 토큰 자카드 중앙값 |", "|---|---|---|"]
    for x, y in PAIRS:
        eq = sum(1 for r in rows if r[x] == r[y]) / len(rows)
        js = statistics.median(jaccard(r[x], r[y]) for r in rows)
        lines.append(f"| {x} vs {y} | {eq:.3f} | {js:.3f} |")

    # value rendering is its own divergence axis: verbalize._fmt collapses an
    # integral float (21.0 -> "21"), caption.fmt_value does not.
    vdiff = sum(1 for r in rows if v_fmt(r["value"]) != c_fmt(r["value"]))
    lines += ["", "## 값 문자열", "",
              f"- `serialize/verbalize._fmt` vs `serialization/base.fmt_value` 불일치: "
              f"{vdiff}/{len(rows)} ({vdiff / len(rows):.3f}) — 정수형 float를 "
              f"`21` / `21.0`로 서로 다르게 렌더한다. 토큰 단위 차이라 BM25 매칭에 영향."]

    diffs = [r for r in rows
             if len({r["a_caption_long"], r["b_verbalize_long"], r["c_script_S3"]}) > 1]
    step = max(1, len(diffs) // 10)
    lines += ["", f"## 차이 예시 10건 (불일치 {len(diffs)}/{len(rows)}건 중 균등 추출)", ""]
    for r in diffs[::step][:10]:
        lines += [f"**{r['table']} r{r['rc'][0]}c{r['rc'][1]}**", "",
                  f"- (a) `{r['a_caption_long']}`",
                  f"- (b) `{r['b_verbalize_long']}`",
                  f"- (c) `{r['c_script_S3']}`", ""]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote -> {args.out}  ({len(rows)} cells, {len(tids)} tables)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
