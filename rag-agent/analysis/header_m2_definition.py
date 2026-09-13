# SPDX-License-Identifier: MIT
"""M2 연도 보존률이 v2 등록값과 다른 이유 — 모델을 돌리지 않는다.

PREREG-2026-09-13-header-units-note.md 는 M1/M2/M3 를 v1·v2 두 값씩 싣지만, 그 값을 낸 코드는 저장소·
git 기록·세션 scratchpad 어디에도 없다. 그래서 문서의 문장 정의("데이터셋 헤더에 연도가 있는 셀 중
우리 경로에 그 연도가 있는 비율")가 열어 둔 선택지를 모두 조합해, v1·v2 기록값 **둘 다**를 재현하는
정의만 남긴다. M1·M3 는 같은 표·셀 집합을 쓰는지 확인하는 대조군이다.

  HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python analysis/header_m2_definition.py
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from analysis.header_v3_impact import cells, within_table_collisions  # noqa: E402
from mh_arms import _DESC, build_tables, load_population             # noqa: E402

OUT = ROOT / "results/mh_header_v3_1/m2_definition.json"
RECORDED = {"v1": {"M1": .2190, "M2": .8998, "M3": .9826}, "v2": {"M1": .1340, "M2": .9673, "M3": .9836}}
REGEX = {"digit_bounded": re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)"),
         "word_bounded": re.compile(r"\b(?:19|20)\d{2}\b")}
PHRASE = ("header_phrase", "col_part", "row_part", "whole_sentence")
MATCH = ("all", "any")
PATH = ("path", "sentence")
DENOM = ("described_in_grid", "in_our_cells", "in_data_region")
UNIT = ("table_instance", "unique_table")


def records(docs, tables, cm):
    rows, seen = [], set()
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
            m = _DESC.match(sentence)
            head = m.group(2) if m else ""
            col = head.rsplit(" of ", 1)[1] if " of " in head else head
            row = head.rsplit(" of ", 1)[0] if " of " in head else ""
            texts = {"header_phrase": head, "col_part": col, "row_part": row, "whole_sentence": sentence}
            cell = cm.get((tid, r, c))
            ukey = (hashlib.sha1(doc[0][t_idx].encode()).hexdigest(), key)
            first = ukey not in seen
            seen.add(ukey)
            rows.append({"texts": texts, "cell": cell, "data": r >= t.nhr and c >= t.nhc, "first": first})
    return rows


def m2_variants(rows):
    acc = {k: [0, 0] for k in itertools.product(PHRASE, REGEX, MATCH, PATH, DENOM, UNIT)}
    for x in rows:
        masks = [(de, un) for de in DENOM for un in UNIT
                 if not (un == "unique_table" and not x["first"])
                 and not (de == "in_our_cells" and x["cell"] is None)
                 and not (de == "in_data_region" and not x["data"])]
        if not masks:
            continue
        cell = x["cell"]
        hays = {"path": " > ".join((*cell[0], *cell[1])) if cell else "", "sentence": cell[3] if cell else ""}
        for ph in PHRASE:
            for rx, pat in REGEX.items():
                ys = set(pat.findall(x["texts"][ph]))
                if not ys:
                    continue
                for pa, hay in hays.items():
                    hit = [y in hay for y in ys]
                    res = {"all": all(hit), "any": any(hit)}
                    for mt in MATCH:
                        for de, un in masks:
                            a = acc[(ph, rx, mt, pa, de, un)]
                            a[0] += res[mt]
                            a[1] += 1
    return {"|".join(k): (round(n / d, 4) if d else None) for k, (n, d) in acc.items()}


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists — 덮어쓰지 않는다")
    _, docs, _ = load_population("train")
    res = {}
    for rule in ("v1", "v2"):
        tables = build_tables(docs, rule)[0]
        cm = cells(tables)
        rows = records(docs, tables, cm)
        m3 = round(sum(x["data"] for x in rows) / len(rows), 4)
        res[rule] = {"M1": round(within_table_collisions(cm) / len(cm), 4), "M3": m3, "M2": m2_variants(rows),
                     "n_described": len(rows)}
        print(rule, "M1", res[rule]["M1"], "M3", m3, flush=True)
    keys = res["v1"]["M2"]
    match = sorted(k for k in keys if res["v1"]["M2"][k] == RECORDED["v1"]["M2"]
                   and res["v2"]["M2"][k] == RECORDED["v2"]["M2"])
    near = sorted(keys, key=lambda k: abs((res["v1"]["M2"][k] or 0) - RECORDED["v1"]["M2"])
                  + abs((res["v2"]["M2"][k] or 0) - RECORDED["v2"]["M2"]))[:10]
    summary = {"recorded": RECORDED,
               "controls": {r: {"M1": res[r]["M1"], "M3": res[r]["M3"]} for r in res},
               "definitions_reproducing_both": match,
               "nearest_10": [{"definition": k, "v1": res["v1"]["M2"][k], "v2": res["v2"]["M2"][k]} for k in near],
               "v3_impact_definition": {"definition": "header_phrase|digit_bounded|all|path|described_in_grid|table_instance",
                                        "v1": res["v1"]["M2"]["header_phrase|digit_bounded|all|path|described_in_grid|table_instance"],
                                        "v2": res["v2"]["M2"]["header_phrase|digit_bounded|all|path|described_in_grid|table_instance"]},
               "all_variants": {r: res[r]["M2"] for r in res}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "all_variants"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
