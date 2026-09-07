#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""데이터셋 감사 — 검색 실험 전에 라벨 결함과 원리적으로 못 찾는 질의를 뺀다.

gold 셀을 HiTab 의 ``answer_formulas`` 에서 다시 만든다. 수식의 셀 참조(``B21``, ``B8:B10``)는
링크 좌표 + 3행 이므로(dev 1,057/1,057 일치) 격자 좌표로 바꾸고, 표의 헤더 블록 오프셋
(H, W)으로 데이터 좌표로 옮긴다. 그 다음 수식을 실제 값으로 계산해 ``answer`` 와 맞는지
본다. 규칙(질의 하나라도 걸리면 제외):

  F  수식을 못 읽거나 참조 셀이 격자 밖/비수치           (라벨 결함)
  V  수식을 계산한 값이 answer 와 1% 넘게 다름            (라벨 결함)
  C  gold 셀이 색인 코퍼스에 없음                          (코퍼스 결함)
  A  gold 셀과 (제목, 행 경로, 열 경로)가 완전히 같은 다른 셀이 코퍼스에 있음
     -- 문장이 값만 다르므로 어떤 텍스트 검색도 구별 못 함 (원리적 불가)
  Q  같은 질문 문장이 다른 표에도 달려 있음                (라벨 결함)
  H  gold 셀의 행·열 경로가 둘 다 비어 있음 (헤더 셀)     (라벨 결함)

산출: ``results/audit2/<pop>_gold.json`` = {query_id: [[tid,i,j],...]} (통과한 질의만,
gold 는 수식 기준), ``results/audit2/AUDIT.md`` 규칙별 건수, 제외 질의 원장 ``*_excluded.jsonl``.

  PYTHONPATH=.:scripts:analysis .venv/bin/python analysis/dataset_audit.py --split dev \
      --population hitab_dev_lookup_all hitab_dev_lookup_multi hitab_dev_corpus_arith
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "analysis"))

import corpus_dump_vs_cell as cdv                                     # noqa: E402
from rag_agent.bench.hitab import _coord_offset, _coords_of, _table_offset, load_samples          # noqa: E402
from rag_agent.data.loader import load_table                           # noqa: E402
from rag_agent.serialization.caption import effective_titles          # noqa: E402
from rag_agent.stores.original_store import _to_float, build_original_table  # noqa: E402

REF = re.compile(r"\$?([A-Z]{1,2})\$?(\d+)")
RANGE = re.compile(r"\$?([A-Z]{1,2})\$?(\d+):\$?([A-Z]{1,2})\$?(\d+)")
ROW_OFF = 3                                    # formula row = linked-cell row + 3
FUNCS = {"SUM": lambda *a: sum(_flat(a)), "AVERAGE": lambda *a: sum(_flat(a)) / len(_flat(a)),
         "MAX": lambda *a: max(_flat(a)), "MIN": lambda *a: min(_flat(a)),
         "COUNT": lambda *a: len(_flat(a)), "COUNTA": lambda *a: len(_flat(a)),
         "ABS": abs}


def _flat(a):
    out = []
    for x in a:
        out.extend(x if isinstance(x, list) else [x])
    return out


def col_idx(letters):
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def parse_formula(f, ot, H, W):
    """-> (python expression, referenced data cells) or raises ValueError."""
    f = re.sub(r'&"[^"]*"', "", str(f)).strip().lstrip("=")   # `=I9&"%"` -> I9
    cells = []

    def data_rc(letters, row):
        r, c = int(row) - ROW_OFF - H, col_idx(letters) - W
        if not (0 <= r < ot.n_rows and 0 <= c < ot.n_cols):
            raise ValueError("ref outside data grid")
        return r, c

    def val(r, c):
        v = _to_float(ot.data[r][c])
        if v is None:
            raise ValueError("non-numeric operand")
        cells.append((r, c))
        return v

    def sub_range(m):
        r1, c1 = data_rc(m.group(1), m.group(2)); r2, c2 = data_rc(m.group(3), m.group(4))
        vals = [val(r, c) for r in range(min(r1, r2), max(r1, r2) + 1)
                for c in range(min(c1, c2), max(c1, c2) + 1)]
        return "[" + ",".join(repr(v) for v in vals) + "]"
    expr = RANGE.sub(sub_range, f)
    expr = REF.sub(lambda m: repr(val(*data_rc(m.group(1), m.group(2)))), expr)
    if not re.fullmatch(r"[\d\.\s\+\-\*/\(\)\[\],A-Z]+", expr):
        raise ValueError(f"unparsed: {expr}")
    return expr, cells


def evaluate(expr):
    return eval(expr, {"__builtins__": {}}, FUNCS)              # noqa: S307 -- regex-restricted


def close(a, b, tol=0.01):
    return abs(a - b) <= tol * max(abs(b), 1e-9) or abs(a - b) < 1e-6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--population", nargs="+", required=True)
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--title-mode", default="page")
    ap.add_argument("--out-dir", default="results/audit2")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    samples = {s["id"]: s for s in load_samples(a.data_dir, a.split)}
    by_q = defaultdict(set)
    for s in samples.values():
        by_q[" ".join(s["question"].split()).lower()].add(s["table_id"])
    linked_by_table = defaultdict(list)
    for s in samples.values():
        linked_by_table[s["table_id"]].append(s.get("linked_cells") or {})
    ot_cache, off_cache = {}, {}

    def table(tid):
        if tid not in ot_cache:
            ot_cache[tid] = build_original_table(load_table(tid, a.data_dir))
            off_cache[tid] = _table_offset(ot_cache[tid], linked_by_table[tid])
        return ot_cache[tid], off_cache[tid]

    md = [f"# 데이터셋 감사 — {a.split} ({a.title_mode} 제목)", "",
          "계측기 `analysis/dataset_audit.py`. gold 는 `answer_formulas` 기준. 규칙은 스크립트 머리.", ""]
    for pop in a.population:
        C = cdv.hitab_corpus(a.data_dir, a.split, pop)
        pt = json.load(open("results/tableconf/totto_page_titles.json")) if a.title_mode == "page" else None
        ti = effective_titles(C.tids, C.title, C.cell_owner, C.cell_paths, a.title_mode, page_titles=pt)
        addr = Counter((ti[t], tuple(rp), tuple(cp)) for (t, _i, _j), (rp, cp, _v)
                       in zip(C.cell_owner, C.cell_paths))
        key_of = {o: (ti[o[0]], tuple(rp), tuple(cp)) for o, (rp, cp, _v) in zip(C.cell_owner, C.cell_paths)}
        have = set(C.cell_owner)
        gold, excluded, why, changed = {}, [], Counter(), 0
        for q in C.queries:
            s = samples[q["query_id"]]; tid = q["gold_table"]
            ot, off = table(tid)
            # the table-level offset fails when one query's link is off; the
            # loader then falls back per query, and so does this audit
            if off is None:
                off = _coord_offset(ot, _coords_of(s.get("linked_cells") or {}))
            flags = []
            cells = []
            if off is None:
                flags.append("F")
            else:
                H, W = off
                try:
                    vals = []
                    for f in s["answer_formulas"]:
                        expr, cs = parse_formula(f, ot, H, W)
                        cells += cs; vals.append(evaluate(expr))
                    ans = [_to_float(x) for x in s["answer"]]
                    if len(ans) != len(vals) or any(x is None for x in ans) or \
                       not all(close(v, x) for v, x in zip(vals, ans)):
                        flags.append("V")
                except (ValueError, ZeroDivisionError, TypeError, SyntaxError):
                    flags.append("F")
            g = sorted({(tid, r, c) for r, c in cells})
            if not flags:
                if any(x not in have for x in g):
                    flags.append("C")
                if any(addr[key_of[x]] > 1 for x in g if x in key_of):
                    flags.append("A")
                if any(not key_of[x][1] and not key_of[x][2] for x in g if x in key_of):
                    flags.append("H")
            if len(by_q[" ".join(s["question"].split()).lower()]) > 1:
                flags.append("Q")
            if flags:
                excluded.append({"query_id": q["query_id"], "rules": flags, "question": s["question"],
                                 "formulas": s["answer_formulas"], "answer": s["answer"],
                                 "pooled_gold": sorted(map(list, q["gold_cells"])), "formula_gold": list(map(list, g))})
                for fl in flags:
                    why[fl] += 1
                continue
            if set(g) != set(q["gold_cells"]):
                changed += 1
            gold[q["query_id"]] = [list(x) for x in g]
        json.dump(gold, open(out / f"{pop}_gold.json", "w"))
        with open(out / f"{pop}_excluded.jsonl", "w") as fh:
            for e in excluded:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
        n = len(C.queries)
        md += [f"## `{pop}` — {n} → **{len(gold)}** 통과 (제외 {len(excluded)})", "",
               "| 규칙 | 건수 |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in sorted(why.items())] + \
              ["", f"수식 gold 가 기존(pooled) gold 와 다른 통과 질의: {changed}",
               f"m 분포(통과): {dict(sorted(Counter(len(v) for v in gold.values()).items()))}", ""]
        print(f"[{pop}] {n} -> {len(gold)} pass; excluded {dict(why)}; gold changed {changed}", flush=True)
    (out / f"AUDIT_{a.split}.md").write_text("\n".join(md) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
