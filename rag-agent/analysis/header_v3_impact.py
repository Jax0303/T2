# SPDX-License-Identifier: MIT
"""헤더 규칙 v3 / v3.1 의 영향 — 인코더·리더를 돌리지 않는다 (PREREG-2026-09-14-header-v3.md).

train 전체 표에서 v2, 새 규칙 전체, 그리고 v2 + 규칙 하나씩의 문장을 만들어 비교한다.
셀은 원표 좌표(raw 행, raw 열)로 짝짓는다 — 헤더 행 수가 바뀌면 데이터 좌표가 밀리기 때문이다.

  HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python analysis/header_v3_impact.py        # v3 → results/mh_header_v3
  HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python analysis/header_v3_impact.py --rule v3.1 \
      --out results/mh_header_v3_1/impact
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from mh_arms import (V3_RULES, V31_RULES, V32_RULES, V33_RULES, _DESC, build_tables,  # noqa: E402
                     load_population, resolve_gold)
from retrieval_accuracy import build_corpus                                       # noqa: E402
from rag_agent.eval.artifacts import digest, provenance                           # noqa: E402

OUT = ROOT / "results/mh_header_v3"
RULES = {"v3": V3_RULES, "v3.1": V31_RULES, "v3.2": V32_RULES, "v3.3": V33_RULES}
ARMS = ("cell", "chunk", "huawei", "rowcol", "tablerag_allobj_path")
YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
WORD_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
SEED = 20260914
# 보고용 사례 목록(사후 분석에서 읽은 문서). 규칙·테스트는 이 목록을 쓰지 않는다.
CASES = {"ce87645c": "헤더 복원", "d4944be6": "헤더 복원", "8018bbdd": "헤더 복원",
         "04e25e10": "헤더 복원", "bd3f443d": "표 구분", "b8fe82b0": "표 구분",
         "b2585208": "표 구분(+범위)", "60a5af4a": "변환 누락", "9807e7e7": "헤더 복원(추가)"}


def corpus(tables, arm):
    a = json.loads((ROOT / f"results/mh_arms/mh_{arm}_hv2.json").read_text(encoding="utf-8"))["arguments"]
    return build_corpus("", sorted(tables), a["template"], a["unit"], {}, a["chunk_chars"],
                        a["tablerag_colmode"], a["row_text"], a["chunk_overlap"], None,
                        load=lambda tid, _d: tables.get(tid), trag_dtype=a["tablerag_dtype"])


def cells(tables):
    """{(tid, raw r, raw c): (row path, col path, value, sentence)} — 본 방법 색인 문장 그대로."""
    texts, covers, _, _, _ = corpus(tables, "cell")
    out = {}
    for text, cov in zip(texts, covers):
        tid, i, j = next(iter(cov))
        t = tables[tid].table
        out[(tid, i + t.nhr, j + t.nhc)] = (tuple(t.row_path(i)), tuple(t.col_path(j)), t.data[i][j], text)
    return out


def within_table_collisions(cm):
    vals = defaultdict(set)
    for (tid, _r, _c), (rp, cp, v, _s) in cm.items():
        vals[(tid, rp, cp)].add(v)
    return sum(1 for (tid, _r, _c), (rp, cp, _v, _s) in cm.items() if len(vals[(tid, rp, cp)]) > 1)


def cross_table_collisions(cm, tables):
    """같은 문서의 다른 표에 같은 경로·다른 값이 있는 셀 수, 그런 표 쌍, 그중 헤더 블록이 같은 쌍."""
    by = defaultdict(lambda: defaultdict(set))
    for (tid, _r, _c), (rp, cp, v, _s) in cm.items():
        by[(tid.split("::")[0], rp, cp)][tid].add(v)
    n, pairs = 0, set()
    for (tid, _r, _c), (rp, cp, v, _s) in cm.items():
        group = by[(tid.split("::")[0], rp, cp)]
        hit = [t2 for t2, vs in group.items() if t2 != tid and vs - {v}]
        if hit:
            n += 1
            pairs.update(tuple(sorted((tid, t2))) for t2 in hit)

    def block(tid):
        t = tables[tid].table
        return tuple(tuple(x.strip() for x in row) for row in t.grid[:t.nhr])
    same = sum(1 for a, b in pairs if block(a) == block(b))
    return {"cells": n, "table_pairs": len(pairs), "pairs_identical_header_block": same}


def dataset_metrics(docs, tables, cm):
    """M2(데이터셋 헤더의 연도가 우리 경로에 있는 셀 비율), M3(데이터셋이 문장을 준 셀 중 데이터 영역 비율).

    M2 는 두 정의를 싣는다. ``M2_year_all``/``_any`` 는 문장이 없는(헤더로 빠진) 셀도 분모에 넣는다.
    ``M2_year_any_in_our_cells`` 는 분모를 우리 셀 문장이 있는 셀로 좁힌 정의로, v1·v2 등록값에 가장
    가깝다(results/mh_header_v3_1/m2_definition.json).
    """
    m2_all = m2_any = n_year = inside = n = m2_ours = n_ours = 0
    for uid, doc in docs.items():
        desc = json.loads(doc[1]) if isinstance(doc[1], str) else doc[1]
        for key, sentence in desc.items():
            parts = key.split("-")
            if len(parts) != 3:
                continue
            t_idx, r, c = (int(x) for x in parts)
            tid = f"{uid}::{t_idx}"
            if tid not in tables:
                continue
            t = tables[tid].table
            if not (0 <= r < len(t.grid) and 0 <= c < len(t.grid[0])):
                continue
            n += 1
            inside += r >= t.nhr and c >= t.nhc
            m = _DESC.match(sentence)
            cell = cm.get((tid, r, c))
            path = " > ".join((*cell[0], *cell[1])) if cell else ""
            years = set(YEAR.findall(m.group(2))) if m else set()
            if years:
                n_year += 1
                m2_all += all(y in path for y in years)
                m2_any += any(y in path for y in years)
            wy = set(WORD_YEAR.findall(m.group(2))) if m else set()
            if wy and cell is not None:
                n_ours += 1
                m2_ours += any(y in path for y in wy)
    return {"M2_year_all": round(m2_all / n_year, 4), "M2_year_any": round(m2_any / n_year, 4),
            "n_year_cells": n_year, "M2_year_any_in_our_cells": round(m2_ours / n_ours, 4),
            "n_year_cells_in_our_cells": n_ours,
            "M3_data_cell": round(inside / n, 4), "n_described_cells": n}


def population(queries, tables, hdr):
    live = {(tid, i, j) for tid, tab in tables.items() for i, row in enumerate(tab.table.data)
            for j, v in enumerate(row) if str(v).strip()}
    qs = resolve_gold(copy.deepcopy(queries), tables, hdr, live)
    gold = {q["uid"]: {(tid, i + tables[tid].table.nhr, j + tables[tid].table.nhc) for tid, i, j in q["gold"]}
            for q in qs if not q["excluded"]}
    return {"n_scored": len(gold), "excluded_by_reason": dict(Counter(q["excluded"] for q in qs if q["excluded"]))}, gold


def diff(base, new, tb, tn):
    both = set(base) & set(new)
    changed = sorted(k for k in both if base[k][3] != new[k][3])
    added = removed = replaced = row_changed = col_changed = 0
    seg_added, seg_removed, dchars = Counter(), Counter(), []
    for k in changed:
        b, x = base[k], new[k]
        plus = Counter((*x[0], *x[1])) - Counter((*b[0], *b[1]))
        minus = Counter((*b[0], *b[1])) - Counter((*x[0], *x[1]))
        added += bool(plus) and not minus
        removed += bool(minus) and not plus
        replaced += bool(plus) and bool(minus)
        row_changed += b[0] != x[0]
        col_changed += b[1] != x[1]
        seg_added.update(plus)
        seg_removed.update(minus)
        dchars.append(len(x[3]) - len(b[3]))
    left, entered = set(base) - set(new), set(new) - set(base)
    tids = {k[0] for k in changed} | {k[0] for k in left | entered}
    shape = sum(1 for tid in tb if (tb[tid].table.nhr, tb[tid].table.nhc) != (tn[tid].table.nhr, tn[tid].table.nhc))
    return changed, {
        "tables_changed": len(tids), "tables_header_shape_changed": shape,
        "cells_sentence_changed": len(changed), "cells_left_data_region": len(left),
        "cells_entered_data_region": len(entered),
        "change_type": {"segments_added_only": added, "segments_removed_only": removed,
                        "segments_replaced": replaced, "row_path_changed": row_changed,
                        "col_path_changed": col_changed},
        "chars_delta_mean": round(sum(dchars) / len(dchars), 1) if dchars else 0.0,
        "top_segments_added": seg_added.most_common(8), "top_segments_removed": seg_removed.most_common(8),
        "examples": [{"cell": list(k), "v2": base[k][3], "new": new[k][3]}
                     for k in random.Random(SEED).sample(changed, min(5, len(changed)))]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rule", default="v3", choices=sorted(RULES))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out_dir, rules, name = Path(a.out), RULES[a.rule], a.rule
    if (out_dir / "impact.json").exists():
        raise SystemExit(f"{out_dir / 'impact.json'} exists — 결과를 덮어쓰지 않는다")
    queries, docs, _ = load_population("train")

    v2 = build_tables(docs, "v2")[0]
    hdr = lambda ts: {tid: (d.table.nhr, d.table.nhc) for tid, d in ts.items()}  # noqa: E731
    preserved = {}
    for arm in ARMS:
        s = json.loads((ROOT / f"results/mh_arms/mh_{arm}_hv2.json").read_text(encoding="utf-8"))
        texts = corpus(v2, arm)[0]
        preserved[arm] = digest(texts) == s["corpus_text_sha256"] and len(texts) == s["n_units"]
    print("[preserve v2]", preserved, flush=True)
    if not all(preserved.values()):
        raise SystemExit("v2 문장이 기록과 다르다 — 코드 변경이 v2 를 바꿨다")

    c2 = cells(v2)
    pop2, gold2 = population(queries, v2, hdr(v2))
    report = {"prereg": "PREREG-2026-09-14-header-v3.md", "rule": name, "models_run": False,
              "v2_preserved_corpus_sha256": preserved, "n_tables": len(v2), "n_cells_v2": len(c2),
              "configs": {}}

    def measure(tables, cm, pop):
        return {"population": pop, "M1_within_table_collision": round(within_table_collisions(cm) / len(cm), 4),
                "cross_table": cross_table_collisions(cm, tables), **dataset_metrics(docs, tables, cm),
                "n_cells": len(cm)}

    report["configs"]["v2"] = measure(v2, c2, pop2)
    per_rule = {}
    for rule in rules:
        tr = build_tables(docs, "v2", rules={rule})[0]
        assert set(tr) == set(v2) and all(tr[t].table.grid == v2[t].table.grid for t in v2)
        cr = cells(tr)
        changed, d = diff(c2, cr, v2, tr)
        pop, _ = population(queries, tr, hdr(tr))
        report["configs"][f"v2+{rule}"] = {**measure(tr, cr, pop), "vs_v2": d}
        per_rule[rule] = set(changed)
        print(f"[{rule}]", {k: d[k] for k in ("tables_changed", "cells_sentence_changed",
                                            "tables_header_shape_changed")}, flush=True)
        del tr, cr

    v3 = build_tables(docs, name)[0]
    assert set(v3) == set(v2) and all(v3[t].table.grid == v2[t].table.grid for t in v2)
    c3 = cells(v3)
    pop3, gold3 = population(queries, v3, hdr(v3))
    changed3, d3 = diff(c2, c3, v2, v3)
    report["configs"][name] = {**measure(v3, c3, pop3), "vs_v2": d3}
    report["configs"][name]["vs_v2"]["cells_changed_by_rule_alone"] = {
        r: len(per_rule[r] & set(changed3)) for r in rules}
    report["configs"][name]["vs_v2"]["cells_changed_only_in_combination"] = len(
        set(changed3) - set().union(*per_rule.values()))

    # 다른 arm 의 색인 단위(같은 표 파서를 쓴다)
    arms = {}
    for arm in ARMS:
        t2, _cv2, _, u2, _ = corpus(v2, arm)
        t3, _cv3, _, u3, _ = corpus(v3, arm)
        g2, g3 = defaultdict(Counter), defaultdict(Counter)
        for text, tid in zip(t2, u2):
            g2[tid][text] += 1
        for text, tid in zip(t3, u3):
            g3[tid][text] += 1
        new_units = sum(sum((g3[t] - g2[t]).values()) for t in set(g2) | set(g3))
        arms[arm] = {"units_v2": len(t2), "units_new": len(t3), "units_not_in_v2": new_units,
                     "tables_with_changed_units": sum(1 for t in set(g2) | set(g3) if g2[t] != g3[t])}
    report["arms"] = arms

    # 평가 문항
    ids200 = json.loads((ROOT / "results/mh_interim200/ids_200.json").read_text(encoding="utf-8"))["ids"]
    ids1047 = [json.loads(line)["query_id"]
               for line in open(ROOT / "results/mh_arms/mh_cell_hv2_answer_doc.jsonl", encoding="utf-8")]
    changed_uids = {k[0].split("::")[0] for k in changed3}
    report["evaluation_sets"] = {
        label: {"n": len(ids), "excluded": sorted(q for q in ids if q not in gold3),
                "gold_cell_set_changed": sorted(q for q in ids if q in gold3 and gold2.get(q) != gold3[q]),
                "gold_sentence_changed": sum(1 for q in ids if q in gold3 and any(
                    c2[k][3] != c3.get(k, (0, 0, 0, None))[3] for k in gold2.get(q, ()))),
                "doc_has_changed_sentence": sum(1 for q in ids if q in changed_uids)}
        for label, ids in (("mh_cell_hv2_answer_doc_1047", ids1047), ("interim200", ids200))}

    # 결함 사례
    cases = {}
    for prefix, kind in CASES.items():
        uid = next(u for u in docs if u.startswith(prefix))
        ks = [k for k in changed3 if k[0].split("::")[0] == uid]
        cm2 = {k: v for k, v in c2.items() if k[0].split("::")[0] == uid}
        cm3 = {k: v for k, v in c3.items() if k[0].split("::")[0] == uid}
        docs_tabs = {t: v3[t] for t in v3 if t.split("::")[0] == uid}
        cases[prefix] = {"kind": kind, "cells_changed": len(ks),
                         "header_shape": {t: [list(hdr(v2)[t]), list(hdr(v3)[t])] for t in docs_tabs},
                         "cross_table_v2": cross_table_collisions(cm2, {t: v2[t] for t in docs_tabs}),
                         "cross_table_new": cross_table_collisions(cm3, docs_tabs),
                         "within_table_collisions_v2": within_table_collisions(cm2),
                         "within_table_collisions_new": within_table_collisions(cm3),
                         "examples": [{"cell": list(k), "v2": c2[k][3], "new": c3[k][3],
                                       "rules_alone": [r for r in rules if k in per_rule[r]]}
                                      for k in ks[:8]]}
    report["cases"] = cases
    report["provenance"] = provenance(ROOT)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"changed_cells_{name}.jsonl").open("x", encoding="utf-8") as f:
        for k in changed3:
            f.write(json.dumps({"cell": list(k), "v2": c2[k][3], "new": c3[k][3],
                                "rules_alone": [r for r in rules if k in per_rule[r]]},
                               ensure_ascii=False) + "\n")
    with (out_dir / "impact.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k not in ("cases", "provenance")},
                     ensure_ascii=False, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
