# SPDX-License-Identifier: MIT
"""기계로 확인되는 두 오류 유형을 v2 와 새 규칙에서 같은 기준으로 센다 — 모델 없음.

  E  잘못된 구획 종료. 표 본문의 구획 표제 행 h 아래, 그 구획 자신의 합계 행(표제 칸 낱말이
     'total'·'subtotal' 뒤에 h 의 낱말 그대로) 위에 있는 데이터 행인데 행 경로에 h 가 없다.
     표제 행과 합계 행 사이에 같은 표제가 다시 나오면 그 쌍은 판정에 쓰지 않는다.
  N  숫자 표제 부착. 행 경로에 표 본문 구획 표제 행의 글이 붙었는데 그 글이 숫자다
     (`_number_like`: 네 자리 연도는 숫자로 보지 않는다). 데이터 행 표제 칸에도 쓰인 글은 세지 않는다.

(행, 표제) 쌍마다 v2 와 규칙에서 따로 판정해 가른다: 둘 다 오류(v2 부터 있음) / v2 에만 오류(규칙이 없앰) /
규칙에만 오류(규칙이 만듦). 쌍 단위라서 v2 에서 이미 한 표제를 잃은 행이 규칙 때문에 다른 표제를 더 잃으면 그 표제는
'규칙이 만듦'이다(2026-09-14 첫 집계 confirmed_errors_v3_{1,2}.json 은 행 단위라 이것을 'v2 부터 있음'에 넣었다).
판정 대상은 두 설정 모두에서 본문 데이터 행인 행이다. 규칙 전체와, 행 경로를 건드리는 규칙 하나씩(v2 위에)을 잰다.

  HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python analysis/header_confirmed_errors.py --rule v3.1 \
      --out results/mh_header_v3_2/confirmed_errors_v3_1_pairs.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import mh_arms as M                                                             # noqa: E402
from rag_agent.eval.artifacts import provenance                                # noqa: E402
from rag_agent.reconstruct.header_grid import _number_like, section_label       # noqa: E402

RULESETS = {"v3.1": "V31_RULES", "v3.2": "V32_RULES"}
SEED = 20260914
_TOTAL_HEADS = (["total"], ["totals"], ["subtotal"], ["subtotals"], ["sub", "total"], ["sub", "totals"])


def words(s):
    return re.findall(r"[a-z0-9]+", str(s).lower())


def own_total(label, heading_words) -> bool:
    w = words(label)
    return any(w[:len(h)] == h and w[len(h):] == heading_words for h in _TOTAL_HEADS)


def judge(t):
    """{raw row: (missing headings, numeric headings, judged for E, row path)} — 본문 데이터 행마다."""
    g, nhr, nhc = t.grid, t.nhr, t.nhc
    body = range(nhr, len(g))
    sec = {x: section_label(g[x]) for x in body}
    has_data = {x: any(str(v).strip() for v in g[x][nhc:]) for x in body}
    data = [r for r in body if not sec[r] and has_data[r]]
    expect = {}
    for x in body:
        hw = words(sec[x])
        if not hw:
            continue
        for y in range(x + 1, len(g)):
            if sec[y] == sec[x]:
                break
            if not sec[y] and has_data[y] and any(own_total(g[y][c], hw) for c in range(nhc)):
                for r in range(x + 1, y):
                    if not sec[r] and has_data[r]:
                        expect.setdefault(r, set()).add(sec[x])
                break
    stubs = {str(g[r][c]).strip() for r in data for c in range(nhc)}
    numeric = {h for h in sec.values() if h and _number_like(h) and h not in stubs}
    out = {}
    for r in data:
        path = tuple(s.strip() for s in t.row_path(r - nhr))
        out[r] = (frozenset(expect.get(r, set()) - set(path)), frozenset(s for s in path if s in numeric),
                  r in expect, path)
    return out


def judge_all(tables):
    return {tid: judge(d.table) for tid, d in tables.items()}


def compare(base, new):
    """(표, 행, 표제) 쌍을 v2 에도 있음 / v2 에만 / 규칙이 만듦으로 가른다."""
    res = {"rows_data_in_both": 0, "rows_data_in_one_only": 0}
    keys = {k: {"v2_and_rule": [], "v2_only": [], "rule_made": []} for k in ("E", "N")}
    judged = 0
    for tid in base:
        jb, jn = base[tid], new[tid]
        both = set(jb) & set(jn)
        res["rows_data_in_both"] += len(both)
        res["rows_data_in_one_only"] += len(set(jb) ^ set(jn))
        for r in both:
            judged += jb[r][2] and jn[r][2]
            for k, i in (("E", 0), ("N", 1)):
                b, x = jb[r][i], jn[r][i]
                for bucket, labels in (("v2_and_rule", b & x), ("v2_only", b - x), ("rule_made", x - b)):
                    keys[k][bucket] += [(tid, r, lab) for lab in labels]
    rng = random.Random(SEED)
    for k, i in (("E", 0), ("N", 1)):
        block = {"rows_judged_in_both": judged} if k == "E" else {}
        for bucket, ks in keys[k].items():
            ks.sort()
            rows = {(t, r) for t, r, _ in ks}
            block[bucket] = {"pairs": len(ks), "rows": len(rows), "tables": len({t for t, _ in rows}), "examples": [
                {"table": t, "raw_row": r, "heading": lab, "flag_v2": sorted(base[t][r][i]),
                 "flag_rule": sorted(new[t][r][i]), "v2_row_path": list(base[t][r][3]),
                 "rule_row_path": list(new[t][r][3])}
                for t, r, lab in rng.sample(ks, min(4, len(ks)))]}
        res[k] = block
    return res, keys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rule", required=True, choices=sorted(RULESETS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out_path = Path(a.out)
    if out_path.exists():
        raise SystemExit(f"{out_path} exists — 덮어쓰지 않는다")
    rules = getattr(M, RULESETS[a.rule])
    _, docs, _ = M.load_population("train")
    v2 = M.build_tables(docs, "v2")[0]
    grids = {tid: d.table.grid for tid, d in v2.items()}
    j2 = judge_all(v2)
    del v2
    flags = [v for j in j2.values() for v in j.values()]
    report = {"rule": a.rule, "models_run": False, "unit": "(표, 행, 표제) 쌍", "definition": __doc__.split("\n\n  HF_")[0],
              "v2_alone": {"E_pairs": sum(len(v[0]) for v in flags), "E_rows": sum(bool(v[0]) for v in flags),
                           "N_pairs": sum(len(v[1]) for v in flags), "N_rows": sum(bool(v[1]) for v in flags),
                           "E_rows_judged": sum(v[2] for v in flags)},
              "configs": {}}
    for name, rs in [(a.rule, None), *((f"v2+{r}", {r}) for r in rules if r[:2] in ("S2", "S3", "S4", "S5"))]:
        tabs = M.build_tables(docs, a.rule if rs is None else "v2", rules=rs)[0]
        assert all(tabs[t].table.grid == grids[t] for t in grids)
        jr = judge_all(tabs)
        del tabs
        report["configs"][name], keys = compare(j2, jr)
        print(name, {k: {b: (v["pairs"], v["rows"], v["tables"]) for b, v in report["configs"][name][k].items()
                         if isinstance(v, dict)} for k in ("E", "N")}, flush=True)
        if rs and next(iter(rs)).startswith("S4"):
            # 이전 기준(header_v3_side_effects.py): S4 가 뗀 마지막 표제의 'Total <표제>' 행이 표 아래 어디든
            # 있으면 셌다. 같은 표제가 다시 나온 뒤의 합계(다음 구획의 것)까지 세므로 E 보다 넓다.
            loose = set()
            for tid, jn in jr.items():
                g = grids[tid]
                for r in set(jn) & set(j2[tid]):
                    gone = [s for s in j2[tid][r][3] if s not in jn[r][3]]
                    if gone and any(words(g[x][0]) == ["total", *words(gone[-1])] for x in range(r + 1, len(g))):
                        loose.add((tid, r))
            tight = {(t, r) for t, r, _ in keys["E"]["rule_made"]}
            report["configs"][name]["previous_criterion_rows"] = {
                "rows": len(loose), "tables": len({t for t, _ in loose}), "also_E_rule_made": len(loose & tight),
                "E_rule_made_not_in_previous": len(tight - loose)}
        del jr
    report["provenance"] = provenance(ROOT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k not in ("provenance", "configs", "definition")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
