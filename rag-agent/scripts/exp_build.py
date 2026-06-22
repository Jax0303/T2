#!/usr/bin/env python3
"""EXPERIMENT_PROMPT GATE-1 + GATE-2.

- flat: WikiTableQuestions(원본) test에서 seed=42로 30개 (질문,표) 추출.
- hier: HiTab dev에서 seed=42로 30개 추출.
- 조건 A(raw 직렬화) / B(header-path 펼치기) 표현 생성.
산출: results/sample_flat_ids.json, results/sample_hier_ids.json, results/exp_inputs.jsonl
"""
from __future__ import annotations
import csv, json, random
from pathlib import Path

WTQ = Path("data/wtq")
HITAB_RAW = Path("data/hitab/data/tables/raw")
RES = Path("results"); RES.mkdir(exist_ok=True)
N = 30
SEED = 42
MAX_BODY_ROWS = 20  # A/B 동일 적용(같은 정보). 경로펼침 B가 길어 요청한도(413) 방지.


# ---------- 직렬화 헬퍼 ----------
def md_grid(rows):
    """rows: list[list[str]] (헤더행 포함된 그대로). 첫 행을 헤더로 마크다운."""
    rows = [[str(c).replace("\n", " ").strip() for c in r] for r in rows]
    if not rows:
        return ""
    w = max(len(r) for r in rows)
    rows = [r + [""] * (w - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |",
           "| " + " | ".join("---" for _ in range(w)) + " |"]
    for r in rows[1:]:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


# ---------- FLAT (WTQ) ----------
def load_wtq_table(rel_path):
    with open(WTQ / rel_path, newline="", encoding="utf-8") as f:
        return [row for row in csv.reader(f)]


def uniqify(header):
    seen, out = {}, []
    for h in header:
        if h in seen:
            seen[h] += 1; out.append(f"{h}_{seen[h]}")
        else:
            seen[h] = 1; out.append(h)
    return out


def rows_to_lines(grid):
    """grid(list of lists) → 행별 파이프 결합 문자열 리스트. A의 `table`/표시 공통."""
    return [" | ".join(str(c).replace("\n", " ").strip() for c in r) for r in grid]


def flat_A_B(grid):
    header = uniqify([str(c).replace("\n", " ").strip() for c in grid[0]])
    body = grid[1:1 + MAX_BODY_ROWS]
    capped_grid = [grid[0]] + body
    # A(raw): grid의 자연 형태 = list of lists. 표시는 마크다운 grid.
    A_table = [[str(c).replace("\n", " ").strip() for c in r] for r in capped_grid]
    A_text = md_grid(capped_grid)
    # B(flat): 행별 "header = value" 줄. 한 겹이라 A와 정보 동일(의도된 A≈B).
    # `table`은 보여준 표현과 동일한 "행들의 리스트"(문자열 줄)로 제공.
    lines = []
    for ri, row in enumerate(body):
        parts = [f"{h} = {row[ci].replace(chr(10),' ').strip() if ci < len(row) else ''}"
                 for ci, h in enumerate(header)]
        lines.append(f"row {ri+1}: " + " | ".join(parts))
    return A_text, "\n".join(lines), A_table, lines


# ---------- HIER (HiTab) ----------
def leaves(node, path=None):
    val = node.get("value")
    path = (path or []) + ([str(val)] if val not in ("<TOP>", "<LEFT>", None) else [])
    ch = node.get("children_dict") or []
    if not ch:
        return [path]
    out = []
    for c in ch:
        out += leaves(c, path)
    return out


BUDGET = 12000  # A/B 직렬화 문자 상한(요청한도 안전). 같은 행수로 양쪽 동시 트림.


def hier_A_B(raw, parsed):
    rc = parsed["raw_cells"]
    top = leaves(rc["top_root"])
    left = leaves(rc["left_root"])
    full_data = rc.get("data") or []
    hdr_rows = raw.get("top_header_rows_num", 1)
    top_s = [" > ".join(p) for p in top]
    left_s = [" > ".join(p) for p in left]

    def build(nrows):
        data = full_data[:nrows]
        texts_c = raw["texts"][:hdr_rows + nrows]
        A_tab = [[str(c).replace("\n", " ").strip() for c in r] for r in texts_c]
        lines = []
        for ri, drow in enumerate(data):
            rlab = left_s[ri] if ri < len(left_s) else f"row{ri}"
            for ci, cell in enumerate(drow):
                v = cell.get("value") if isinstance(cell, dict) else cell
                clab = top_s[ci] if ci < len(top_s) else f"col{ci}"
                lines.append(f"[{rlab}] {clab} = {v}")
        return md_grid(texts_c), "\n".join(lines), A_tab, lines

    nrows = min(MAX_BODY_ROWS, len(full_data))
    while nrows > 1:
        A_text, B_text, A_table, B_table = build(nrows)
        if max(len(A_text), len(B_text)) <= BUDGET:
            break
        nrows -= 2
    else:
        A_text, B_text, A_table, B_table = build(1)
    return A_text, B_text, A_table, B_table


def main():
    rng = random.Random(SEED)
    inputs = []

    # FLAT
    tsv = WTQ / "data" / "pristine-unseen-tables.tsv"
    wtq_rows = []
    with open(tsv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            wtq_rows.append(row)
    flat_idx = sorted(rng.sample(range(len(wtq_rows)), N))
    flat_ids = []
    for i in flat_idx:
        row = wtq_rows[i]
        grid = load_wtq_table(row["context"])
        A, B, At, Bt = flat_A_B(grid)
        flat_ids.append({"idx": i, "id": row["id"], "question": row["utterance"],
                         "table_path": row["context"], "answer": row["targetValue"]})
        inputs.append({"id": row["id"], "complexity": "flat", "question": row["utterance"],
                       "gold_answer": row["targetValue"], "A_text": A, "B_text": B,
                       "A_table": At, "B_table": Bt})
    json.dump(flat_ids, open(RES / "sample_flat_ids.json", "w"), indent=2, ensure_ascii=False)

    # HIER
    queries = [json.loads(l) for l in open("queries.jsonl", encoding="utf-8")]
    queries = [q for q in queries if q["split"] == "dev"]
    hier_idx = sorted(rng.sample(range(len(queries)), N))
    hier_ids = []
    for i in hier_idx:
        q = queries[i]
        tid = q["gold_table_id"]
        raw = json.load(open(HITAB_RAW / f"{tid}.json", encoding="utf-8"))
        parsed = None
        for line in open("corpus/tables.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r["table_id"] == tid:
                parsed = r; break
        A, B, At, Bt = hier_A_B(raw, parsed)
        ans = q["answer"]
        hier_ids.append({"idx": i, "id": q["query_id"], "table_id": tid,
                         "question": q["question"], "answer": ans})
        inputs.append({"id": q["query_id"], "complexity": "hier", "question": q["question"],
                       "gold_answer": ans, "A_text": A, "B_text": B,
                       "A_table": At, "B_table": Bt})
    json.dump(hier_ids, open(RES / "sample_hier_ids.json", "w"), indent=2, ensure_ascii=False)

    with open(RES / "exp_inputs.jsonl", "w", encoding="utf-8") as f:
        for r in inputs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- GATE-1 ----
    print("=" * 60)
    print("GATE-1")
    print(f"  flat loaded: {len([x for x in inputs if x['complexity']=='flat'])}  -> results/sample_flat_ids.json")
    print(f"  hier loaded: {len([x for x in inputs if x['complexity']=='hier'])}  -> results/sample_hier_ids.json")
    fe = next(x for x in inputs if x["complexity"] == "flat")
    he = next(x for x in inputs if x["complexity"] == "hier")
    print("\n  [flat sample table — single header row]")
    print("   " + "\n   ".join(fe["A_text"].splitlines()[:4]))
    print("\n  [hier sample table — multi-row header]")
    print("   " + "\n   ".join(he["A_text"].splitlines()[:5]))

    # ---- GATE-2 ----
    print("\n" + "=" * 60)
    print("GATE-2  (A vs B)")
    print("\n--- FLAT id=%s ---" % fe["id"])
    print("[A raw]\n" + "\n".join(fe["A_text"].splitlines()[:4]))
    print("[B header-path]\n" + "\n".join(fe["B_text"].splitlines()[:3]))
    print("\n--- HIER id=%s ---" % he["id"])
    print("[A raw — visible multi-header]\n" + "\n".join(he["A_text"].splitlines()[:5]))
    print("[B header-path — flattened]\n" + "\n".join(he["B_text"].splitlines()[:5]))


if __name__ == "__main__":
    main()
