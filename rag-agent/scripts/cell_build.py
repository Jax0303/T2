#!/usr/bin/env python3
"""CELL_RETRIEVAL GATE-1 + GATE-2. 셀좌표 샘플링 + 셀단위 A/B 표현.

hier(HiTab): linked_cells.quantity_link 의 데이터영역 좌표 = 정답셀(ground-truth).
flat(WTQ): 셀좌표 라벨 없음 → 정답문자열과 정규화 매칭되는 셀(complete-match).
           매칭 0개/>2개 질문은 제외(애매). 제외율 보고.

각 표를 데이터영역 셀로 분해:
  A(raw)  = 값 단독
  B(path) = "{row_path} | {top_path} | {value}"  (flat은 "{col_header} | {value}")

산출: results/cell_sample_hier.json, results/cell_sample_flat.json
"""
from __future__ import annotations
import csv, json, random, re
from pathlib import Path

HITAB_RAW = Path("data/hitab/data/tables/raw")
HITAB_DEV = Path("data/hitab/data/dev_samples.jsonl")
WTQ = Path("data/wtq")
RES = Path("results"); RES.mkdir(exist_ok=True)
N = 100
SEED = 42


def norm(s):
    s = str(s).strip().lower().replace(",", "").replace("%", "").replace("$", "")
    try:
        return str(round(float(s), 4))
    except ValueError:
        return s.strip()


# ---------- HIER (HiTab) — 전부 texts 프레임에서 처리 ----------
def header_band(texts, data):
    """texts 안에서 데이터행렬을 정렬해 헤더밴드 크기(hr,hc) 추정.
    raw 파일의 header_rows/cols 카운트가 불안정하므로 값매칭으로 고정."""
    R = len(texts); C = max(len(x) for x in texts)
    if not data:
        return 1, 1
    nR = len(data); nC = len(data[0])
    best = (-1, 1, 1)
    for hr in range(0, max(1, R - nR) + 1):
        for hc in range(0, max(1, C - nC) + 1):
            m = 0
            for j in range(nR):
                row = texts[hr + j] if hr + j < R else []
                for i in range(nC):
                    c = hc + i
                    if c < len(row) and str(row[c]).strip() == str(data[j][i].get("value")).strip():
                        m += 1
            if m > best[0]:
                best = (m, hr, hc)
    return best[1], best[2]


def fill_paths(texts, hr, hc):
    """병합셀 처리: 상단헤더는 좌→우, 좌측헤더는 상→하 forward-fill."""
    C = max(len(x) for x in texts)
    top = []   # top[r][c]
    for r in range(hr):
        row, last = [], ""
        for c in range(C):
            v = str(texts[r][c]).strip() if c < len(texts[r]) else ""
            if v:
                last = v
            row.append(last)
        top.append(row)
    left = [["" for _ in range(hc)] for _ in range(len(texts))]
    for c in range(hc):
        last = ""
        for r in range(len(texts)):
            v = str(texts[r][c]).strip() if c < len(texts[r]) else ""
            if v:
                last = v
            left[r][c] = last

    def dedup(seq):
        out = []
        for x in seq:
            if x and (not out or out[-1] != x):
                out.append(x)
        return out

    def col_path(c):
        return " > ".join(dedup([top[r][c] for r in range(hr)]))

    def row_path(r):
        return " > ".join(dedup([left[r][c] for c in range(hc)]))
    return col_path, row_path


def build_hier():
    corp = {}
    for l in open("corpus/tables.jsonl", encoding="utf-8"):
        r = json.loads(l); corp[r["table_id"]] = r
    samples = [json.loads(l) for l in open(HITAB_DEV, encoding="utf-8")]
    rng = random.Random(SEED)
    rng.shuffle(samples)
    out, skipped = [], 0
    for s in samples:
        tid = s["table_id"]
        tf = HITAB_RAW / f"{tid}.json"
        if not tf.exists():
            skipped += 1; continue
        raw = json.load(open(tf, encoding="utf-8"))
        texts = raw["texts"]
        data = corp[tid]["raw_cells"]["data"] if tid in corp else []
        hr, hc = header_band(texts, data)
        col_path, row_path = fill_paths(texts, hr, hc)
        # 후보 셀 = 데이터영역 비어있지 않은 셀 (texts 프레임 좌표)
        cells = []
        for r in range(hr, len(texts)):
            for c in range(hc, len(texts[r])):
                val = str(texts[r][c]).replace("\n", " ").strip()
                if not val:
                    continue
                cells.append({"coord": [r, c], "A": val,
                              "B": f"{row_path(r)} | {col_path(c)} | {val}"})
        # gold = quantity_link 데이터영역 좌표 (texts 프레임 그대로)
        gold = []
        ql = sample_ql(s)
        for r, c in ql:
            if r >= hr and c >= hc and str(texts[r][c]).strip():
                gold.append([r, c])
        gold = [list(x) for x in {tuple(g) for g in gold}]
        if not gold or not cells:
            skipped += 1; continue
        out.append({"id": s["id"], "table_id": tid, "question": s["question"],
                    "answer": s["answer"], "gold_cells": gold, "cells": cells,
                    "hr": hr, "hc": hc})
        if len(out) >= N:
            break
    json.dump(out, open(RES / "cell_sample_hier.json", "w"), ensure_ascii=False)
    return out, skipped


def sample_ql(s):
    coords = []
    for _v, cd in s["linked_cells"].get("quantity_link", {}).items():
        for rc in cd:
            coords.append(tuple(eval(rc)))
    return coords


# ---------- FLAT (WTQ) ----------
def load_wtq_table(rel):
    with open(WTQ / rel, newline="", encoding="utf-8") as f:
        return [row for row in csv.reader(f)]


def build_flat():
    tsv = WTQ / "data" / "pristine-unseen-tables.tsv"
    rows = list(csv.DictReader(open(tsv, newline="", encoding="utf-8"), delimiter="\t"))
    rng = random.Random(SEED)
    rng.shuffle(rows)
    out, excl_zero, excl_multi, considered = [], 0, 0, 0
    for row in rows:
        considered += 1
        grid = load_wtq_table(row["context"])
        if len(grid) < 2:
            continue
        header = [str(c).replace("\n", " ").strip() for c in grid[0]]
        gold_ans = norm(row["targetValue"])
        cells, gold = [], []
        for r in range(1, len(grid)):
            for c in range(len(grid[r])):
                val = str(grid[r][c]).replace("\n", " ").strip()
                colh = header[c] if c < len(header) else f"col{c}"
                cells.append({"coord": [r, c], "A": val, "B": f"{colh} | {val}"})
                if norm(val) == gold_ans:
                    gold.append([r, c])
        if len(gold) == 0:
            excl_zero += 1; continue
        if len(gold) > 2:
            excl_multi += 1; continue
        out.append({"id": row["id"], "table_id": row["context"], "question": row["utterance"],
                    "answer": row["targetValue"], "gold_cells": gold, "cells": cells})
        if len(out) >= N:
            break
    json.dump(out, open(RES / "cell_sample_flat.json", "w"), ensure_ascii=False)
    excl_total = excl_zero + excl_multi
    denom = len(out) + excl_total
    return out, {"excl_zero": excl_zero, "excl_multi": excl_multi,
                 "excl_rate": round(excl_total / denom, 3) if denom else 0,
                 "considered_until_100": considered}


def main():
    hier, h_skip = build_hier()
    flat, f_stat = build_flat()

    print("=" * 60); print("GATE-1")
    print(f"  hier(HiTab): {len(hier)}쌍 로드 (skip={h_skip})")
    print(f"  flat(WTQ):   {len(flat)}쌍 로드")
    print(f"  flat 정답셀 약식매칭 제외: zero={f_stat['excl_zero']} multi={f_stat['excl_multi']} "
          f"제외율={f_stat['excl_rate']}")
    # hier 표 1개 헤더 구조
    h0 = hier[0]; raw = json.load(open(HITAB_RAW / f"{h0['table_id']}.json"))
    print(f"\n  [hier 표 헤더 구조 — 추정 header_rows={h0['hr']} "
          f"header_cols={h0['hc']}] (계층 육안확인)")
    for r in raw["texts"][:h0["hr"] + 1]:
        print("   | " + " | ".join(str(x) for x in r) + " |")

    print("\n" + "=" * 60); print("GATE-2  (동일 정답셀의 A vs B)")
    gc = tuple(h0["gold_cells"][0])
    cell = next(c for c in h0["cells"] if tuple(c["coord"]) == gc)
    print(f"  hier id={h0['id']} 정답셀{gc}")
    print(f"    [A] {cell['A']!r}")
    print(f"    [B] {cell['B']!r}")
    f0 = flat[0]; fgc = tuple(f0["gold_cells"][0])
    fcell = next(c for c in f0["cells"] if tuple(c["coord"]) == fgc)
    print(f"  flat id={f0['id']} 정답셀{fgc}  (A≈B 대조군)")
    print(f"    [A] {fcell['A']!r}")
    print(f"    [B] {fcell['B']!r}")


if __name__ == "__main__":
    main()
