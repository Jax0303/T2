#!/usr/bin/env python3
"""진단 Phase 1 — C0~C2 전처리 조건 직렬화 (flat + hierarchical 공통).

독립변수 = *정보 증강* (직렬화 포맷이 아니라). artifact §3:
직렬화 포맷은 BGE에서 병목이 아니므로, 변수를 metadata/schema 증강에 둔다.

  C0 (raw)        : 표 본문(헤더+행)만. 제목/메타데이터 없음.
  C1 (+metadata)  : C0 + 제목/섹션/캡션.
  C2 (+schema)    : C1 + 스키마 서술.
                    flat → 컬럼명+추론타입.
                    hier → root-to-leaf 헤더 경로(계층 평탄화; HiTab/codegen에서 표준).
  C3 (+synthetic Q): 별도 스크립트(diag_synthetic_q.py)에서 C2 위에 합성질문 추가.

산출: diag/{flat,hier}/serialized/{C0,C1,C2}.records.jsonl  {table_id, text}

사용: python scripts/diag_serialize.py --dataset flat
      python scripts/diag_serialize.py --dataset hier
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

MAX_ROWS = 40  # 토큰 예산: 본문 행 상한 (flat 표 일부는 수백 행)


def md_table(header, rows, max_rows=MAX_ROWS):
    header = [str(h) for h in header]
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join("---" for _ in header) + " |"]
    for r in rows[:max_rows]:
        cells = [str(c) for c in r]
        if len(cells) < len(header):
            cells += [""] * (len(header) - len(cells))
        out.append("| " + " | ".join(cells[:len(header)]) + " |")
    return "\n".join(out)


def infer_type(rows, col_idx):
    for r in rows:
        if col_idx < len(r):
            v = str(r[col_idx]).strip().replace(",", "").replace("%", "")
            if v:
                try:
                    float(v.split()[0]); return "number"
                except ValueError:
                    return "text"
    return "text"


# ---------- FLAT (OpenWikiTable) ----------
def flat_records(tables_path):
    for line in open(tables_path, encoding="utf-8"):
        t = json.loads(line)
        header, rows = t["header"], t["rows"]
        body = md_table(header, rows)
        meta = " — ".join(x for x in (t["page_title"], t["section_title"], t["caption"]) if x)
        schema = "Columns: " + ", ".join(
            f"{h} ({infer_type(rows, i)})" for i, h in enumerate(header))
        yield t["table_id"], {
            "C0": body,
            "C1": (meta + "\n\n" + body) if meta else body,
            "C2": (meta + "\n\n" + body + "\n\n" + schema) if meta else (body + "\n\n" + schema),
        }


# ---------- HIERARCHICAL (HiTab) ----------
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


def hier_records(tables_path):
    for line in open(tables_path, encoding="utf-8"):
        t = json.loads(line)
        rc = t["raw_cells"]
        top = leaves(rc["top_root"])      # 컬럼 경로
        left = leaves(rc["left_root"])    # 행 경로
        data = rc.get("data") or []
        col_names = [" > ".join(p) for p in top]
        row_labels = [" > ".join(p) for p in left]
        # 본문 grid: 첫 컬럼 = 행 헤더(leaf), 이후 = 데이터
        header = ["row"] + [p[-1] if p else "" for p in top]
        rows = []
        for i, drow in enumerate(data):
            label = row_labels[i] if i < len(row_labels) else ""
            vals = [d.get("value") if isinstance(d, dict) else d for d in drow]
            rows.append([label] + vals)
        body = md_table(header, rows)
        title = t.get("title") or ""
        # C2-hier schema = root-to-leaf 헤더 경로 전체(계층 평탄화)
        schema = ("Column header paths: " + "; ".join(col_names)
                  + "\nRow header paths: " + "; ".join(row_labels[:MAX_ROWS]))
        yield t["table_id"], {
            "C0": body,
            "C1": (title + "\n\n" + body) if title else body,
            "C2": (title + "\n\n" + body + "\n\n" + schema) if title else (body + "\n\n" + schema),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["flat", "hier"], required=True)
    args = ap.parse_args()

    if args.dataset == "flat":
        tables_path = "diag/flat/tables.jsonl"
        gen = flat_records(tables_path)
        out_dir = Path("diag/flat/serialized")
    else:
        tables_path = "corpus/tables.jsonl"
        gen = hier_records(tables_path)
        out_dir = Path("diag/hier/serialized")
    out_dir.mkdir(parents=True, exist_ok=True)

    writers = {c: open(out_dir / f"{c}.records.jsonl", "w", encoding="utf-8") for c in ("C0", "C1", "C2")}
    n = 0
    for tid, conds in gen:
        for c, w in writers.items():
            w.write(json.dumps({"table_id": tid, "text": conds[c]}, ensure_ascii=False) + "\n")
        n += 1
    for w in writers.values():
        w.close()
    print(f"[{args.dataset}] serialized {n} tables × C0/C1/C2 → {out_dir}")


if __name__ == "__main__":
    main()
