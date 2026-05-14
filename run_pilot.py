"""
HiTab Serialization Format Pilot Experiment
============================================

Measures retrieval accuracy across 6 table serialization formats on HiTab.
Each table -> embedded once per format. Each QA -> embedded as query.
Gold label = QA's table_id. Metrics: Recall@K, MRR.
"""

import json
import os
import random
import sys
import time
import html as html_lib
from pathlib import Path
from typing import Any

# Force UTF-8 stdout on Windows so em-dash and other unicode print without errors.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
from sentence_transformers import SentenceTransformer
from tabulate import tabulate
from tqdm import tqdm

# ----------------------------- config -----------------------------
SEED = 42
N_TABLES = 100
HITAB_ROOT = Path(r"C:\Users\user\hitab_pilot\HiTab")
TABLES_DIR = HITAB_ROOT / "data" / "tables" / "raw"
DEV_PATH = HITAB_ROOT / "data" / "dev_samples.jsonl"
OUTPUT_DIR = Path(r"C:\Users\user\hitab_pilot\results")
MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
KS = [1, 3, 5, 10]


# --------------------------- table parser -------------------------
def _node_name(node: dict, texts: list[list[str]]) -> str:
    """Resolve a tree node's display name. Raw tables only carry row/col indices; hmt carries name/value."""
    n = node.get("name") or node.get("value")
    if n:
        return str(n).strip()
    r, c = node.get("row_index"), node.get("column_index")
    if r is not None and c is not None and 0 <= r < len(texts) and 0 <= c < len(texts[r]):
        return str(texts[r][c]).strip()
    return ""


def _collect(node: dict, path: list[str], texts: list[list[str]],
             out_paths: dict[int, list[str]], out_max_perp: list[int],
             axis: str, depth: int = 0):
    """Walk a header tree.

    axis='top'  -> leaf maps from data column index (column_index) to header path; perp axis is row_index.
    axis='left' -> leaf maps from data row index (row_index) to header path; perp axis is column_index.

    Sentinel root has row_index/column_index = -1 and is skipped from the path.
    """
    is_sentinel = depth == 0 or (
        node.get("row_index") in (None, -1) and node.get("column_index") in (None, -1)
        and not node.get("name") and not node.get("value")
    )
    name = _node_name(node, texts) if not is_sentinel else ""
    new_path = path + [name] if name else path

    children = node.get("children", []) or node.get("children_dict", []) or []
    if not children:
        # leaf: record the path and the data row/col it covers
        if "line_idx" in node and node.get("line_idx") is not None:
            # hmt format: line_idx = data axis index (0-based among data cells)
            out_paths[int(node["line_idx"])] = new_path
            return
        r = node.get("row_index")
        c = node.get("column_index")
        if axis == "top":
            if c is not None:
                out_paths[int(c)] = new_path
            if r is not None:
                out_max_perp[0] = max(out_max_perp[0], int(r))
        else:
            if r is not None:
                out_paths[int(r)] = new_path
            if c is not None:
                out_max_perp[0] = max(out_max_perp[0], int(c))
        return
    for ch in children:
        _collect(ch, new_path, texts, out_paths, out_max_perp, axis, depth + 1)


def load_table(path: Path) -> dict:
    """Load and normalize a HiTab raw table."""
    with open(path, "r", encoding="utf-8") as f:
        t = json.load(f)
    title = t.get("title", "") or ""
    texts_raw = t.get("texts") or t.get("data") or []
    texts = [[(str(c.get("value", "")) if isinstance(c, dict) else str(c)) for c in row] for row in texts_raw]

    n_rows = len(texts)
    n_cols = max((len(r) for r in texts), default=0)
    for r in texts:
        if len(r) < n_cols:
            r.extend([""] * (n_cols - len(r)))

    top_paths: dict[int, list[str]] = {}
    left_paths: dict[int, list[str]] = {}
    top_max_row = [-1]
    left_max_col = [-1]
    _collect(t["top_root"], [], texts, top_paths, top_max_row, axis="top")
    _collect(t["left_root"], [], texts, left_paths, left_max_col, axis="left")

    # Header extents: trust the tree, then take the max with the dataset field if present.
    # Dataset's top_header_rows_num is sometimes (tree_max + 2); we want first_data_row_index.
    tree_first_data_row = top_max_row[0] + 1 if top_max_row[0] >= 0 else 0
    tree_first_data_col = left_max_col[0] + 1 if left_max_col[0] >= 0 else 0

    ds_top = t.get("top_header_rows_num")
    ds_left = t.get("left_header_columns_num")
    # dataset values appear to overcount by 1 in many tables (off-by-one). Take min so we don't lose data rows.
    if ds_top is not None and ds_top - 1 > tree_first_data_row:
        first_data_row = ds_top - 1
    else:
        first_data_row = tree_first_data_row
    if ds_left is not None and ds_left - 1 > tree_first_data_col:
        first_data_col = ds_left - 1
    else:
        first_data_col = tree_first_data_col

    data_rows = list(range(first_data_row, n_rows))
    data_cols = list(range(first_data_col, n_cols))

    return {
        "title": title,
        "texts": texts,
        "top_paths": top_paths,
        "left_paths": left_paths,
        "top_header_rows": first_data_row,
        "left_header_cols": first_data_col,
        "data_rows": data_rows,
        "data_cols": data_cols,
        "merged_regions": t.get("merged_regions", []),
    }


def get_header_path(parsed: dict, axis: str, idx: int) -> list[str]:
    """Get header path for a given column (axis='top') or row (axis='left')."""
    if axis == "top":
        return parsed["top_paths"].get(idx, [])
    else:
        return parsed["left_paths"].get(idx, [])


# ----------------------- merged-cell resolver ---------------------
def build_merge_map(parsed: dict) -> dict[tuple[int, int], tuple[int, int]]:
    """For each merged cell (r, c), map it to the start cell (first_row, first_col)."""
    m = {}
    for region in parsed.get("merged_regions", []):
        fr, lr = region["first_row"], region["last_row"]
        fc, lc = region["first_column"], region["last_column"]
        for r in range(fr, lr + 1):
            for c in range(fc, lc + 1):
                m[(r, c)] = (fr, fc)
    return m


def cell_value(parsed: dict, merge_map: dict, r: int, c: int) -> str:
    """Get cell value, resolving through merge map (so merged cells get the start-cell value)."""
    sr, sc = merge_map.get((r, c), (r, c))
    texts = parsed["texts"]
    if 0 <= sr < len(texts) and 0 <= sc < len(texts[sr]):
        return texts[sr][sc]
    return ""


# ------------------------- 6 SERIALIZERS --------------------------
def serialize_markdown(parsed: dict) -> str:
    """Hierarchical headers flattened by ' - '."""
    title = parsed["title"]
    texts = parsed["texts"]
    data_rows = parsed["data_rows"]
    data_cols = parsed["data_cols"]
    left_cols = parsed["left_header_cols"]
    merge_map = build_merge_map(parsed)

    # Build flattened header for each data column
    flat_headers = []
    for c in data_cols:
        path = get_header_path(parsed, "top", c)
        flat_headers.append(" - ".join(p for p in path if p) or f"col{c}")

    # Header for left columns
    left_label = "Row Header"
    out = []
    if title:
        out.append(f"# {title}")
        out.append("")
    header_line = "| " + left_label + " | " + " | ".join(flat_headers) + " |"
    sep_line = "| " + " | ".join(["---"] * (1 + len(flat_headers))) + " |"
    out.append(header_line)
    out.append(sep_line)
    for r in data_rows:
        # left header
        left_path = get_header_path(parsed, "left", r)
        left_label_val = " - ".join(p for p in left_path if p)
        if not left_label_val:
            left_label_val = " / ".join(cell_value(parsed, merge_map, r, lc) for lc in range(left_cols)).strip(" /")
        row_vals = [cell_value(parsed, merge_map, r, c) for c in data_cols]
        out.append("| " + left_label_val + " | " + " | ".join(row_vals) + " |")
    return "\n".join(out)


def serialize_html(parsed: dict) -> str:
    """HTML with rowspan/colspan for merged cells. Preserves hierarchy."""
    title = parsed["title"]
    texts = parsed["texts"]
    n_rows = len(texts)
    n_cols = max((len(r) for r in texts), default=0)

    # Determine which (r, c) to skip due to merge (only render the start cell)
    skip = set()
    span_info = {}  # (r, c) -> (rowspan, colspan)
    for region in parsed.get("merged_regions", []):
        fr, lr, fc, lc = region["first_row"], region["last_row"], region["first_column"], region["last_column"]
        span_info[(fr, fc)] = (lr - fr + 1, lc - fc + 1)
        for r in range(fr, lr + 1):
            for c in range(fc, lc + 1):
                if (r, c) != (fr, fc):
                    skip.add((r, c))

    top_h = parsed["top_header_rows"]
    left_w = parsed["left_header_cols"]

    out = ["<table>"]
    if title:
        out.append(f"  <caption>{html_lib.escape(title)}</caption>")
    for r in range(n_rows):
        out.append("  <tr>")
        for c in range(n_cols):
            if (r, c) in skip:
                continue
            val = texts[r][c] if c < len(texts[r]) else ""
            tag = "th" if (r < top_h or c < left_w) else "td"
            attrs = ""
            if (r, c) in span_info:
                rs, cs = span_info[(r, c)]
                if rs > 1:
                    attrs += f' rowspan="{rs}"'
                if cs > 1:
                    attrs += f' colspan="{cs}"'
            out.append(f"    <{tag}{attrs}>{html_lib.escape(val)}</{tag}>")
        out.append("  </tr>")
    out.append("</table>")
    return "\n".join(out)


def serialize_csv(parsed: dict) -> str:
    """Plain CSV. Structure lost."""
    import csv
    import io
    title = parsed["title"]
    texts = parsed["texts"]
    buf = io.StringIO()
    if title:
        buf.write(title + "\n")
    w = csv.writer(buf)
    for row in texts:
        w.writerow(row)
    return buf.getvalue()


def serialize_json_records(parsed: dict) -> str:
    """List of {flat_header: value} dicts, one per data row."""
    title = parsed["title"]
    data_rows = parsed["data_rows"]
    data_cols = parsed["data_cols"]
    left_cols = parsed["left_header_cols"]
    merge_map = build_merge_map(parsed)

    flat_headers = []
    for c in data_cols:
        path = get_header_path(parsed, "top", c)
        flat_headers.append(" - ".join(p for p in path if p) or f"col{c}")

    records = []
    for r in data_rows:
        rec = {}
        left_path = get_header_path(parsed, "left", r)
        rec["__row__"] = " - ".join(p for p in left_path if p) or " / ".join(
            cell_value(parsed, merge_map, r, lc) for lc in range(left_cols)
        )
        for c, h in zip(data_cols, flat_headers):
            rec[h] = cell_value(parsed, merge_map, r, c)
        records.append(rec)
    obj = {"title": title, "records": records}
    return json.dumps(obj, ensure_ascii=False, indent=None)


def serialize_natural_language(parsed: dict) -> str:
    """One sentence per data cell: 'The [path] is [value]'."""
    title = parsed["title"]
    data_rows = parsed["data_rows"]
    data_cols = parsed["data_cols"]
    merge_map = build_merge_map(parsed)
    left_cols = parsed["left_header_cols"]

    sentences = []
    if title:
        sentences.append(f"Table title: {title}.")
    for r in data_rows:
        left_path = get_header_path(parsed, "left", r)
        if not left_path:
            left_path = [cell_value(parsed, merge_map, r, lc) for lc in range(left_cols) if cell_value(parsed, merge_map, r, lc)]
        for c in data_cols:
            top_path = get_header_path(parsed, "top", c)
            val = cell_value(parsed, merge_map, r, c)
            if not val:
                continue
            full_path = top_path + left_path
            path_str = " ".join(p for p in full_path if p)
            sentences.append(f"The {path_str} is {val}.")
    return " ".join(sentences)


def serialize_breadcrumb(parsed: dict) -> str:
    """Markdown but each data cell prefixed with [Top > Path > Left] breadcrumb."""
    title = parsed["title"]
    data_rows = parsed["data_rows"]
    data_cols = parsed["data_cols"]
    left_cols = parsed["left_header_cols"]
    merge_map = build_merge_map(parsed)

    flat_headers = []
    for c in data_cols:
        path = get_header_path(parsed, "top", c)
        flat_headers.append(" > ".join(p for p in path if p) or f"col{c}")

    out = []
    if title:
        out.append(f"# {title}")
        out.append("")
    out.append("| Row | " + " | ".join(flat_headers) + " |")
    out.append("| " + " | ".join(["---"] * (1 + len(flat_headers))) + " |")
    for r in data_rows:
        left_path = get_header_path(parsed, "left", r)
        left_str = " > ".join(p for p in left_path if p)
        if not left_str:
            left_str = " / ".join(cell_value(parsed, merge_map, r, lc) for lc in range(left_cols)).strip(" /")
        cells = []
        for c, top_h in zip(data_cols, flat_headers):
            val = cell_value(parsed, merge_map, r, c)
            crumb = f"[{top_h} > {left_str}]" if left_str else f"[{top_h}]"
            cells.append(f"{crumb} {val}")
        out.append("| " + left_str + " | " + " | ".join(cells) + " |")
    return "\n".join(out)


SERIALIZERS = {
    "markdown": serialize_markdown,
    "html": serialize_html,
    "csv": serialize_csv,
    "json": serialize_json_records,
    "natural_language": serialize_natural_language,
    "breadcrumb": serialize_breadcrumb,
}


# --------------------- sample tables + QAs ------------------------
def sample_tables_and_qas() -> tuple[list[str], list[dict]]:
    """Sample N_TABLES table_ids that have at least one QA in dev set; return all matching QAs."""
    table_to_qas: dict[str, list[dict]] = {}
    with open(DEV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            qa = json.loads(line)
            table_to_qas.setdefault(qa["table_id"], []).append(qa)

    # Only keep tables whose JSON file actually exists
    candidate_ids = sorted([tid for tid in table_to_qas if (TABLES_DIR / f"{tid}.json").exists()])
    print(f"[sample] dev tables with file present: {len(candidate_ids)} / {len(table_to_qas)}")

    rng = random.Random(SEED)
    sampled = rng.sample(candidate_ids, min(N_TABLES, len(candidate_ids)))
    sampled_set = set(sampled)
    qas = [qa for tid in sampled for qa in table_to_qas[tid]]
    return sampled, qas


# --------------------------- evaluate -----------------------------
def evaluate(query_embs: np.ndarray, table_embs: np.ndarray,
             qa_table_idx: list[int]) -> dict:
    """Cosine similarity (embeddings already normalized) -> Recall@K, MRR."""
    sims = query_embs @ table_embs.T  # (n_queries, n_tables)
    # rank: for each query, position of the gold table
    ranks = []
    for i, gold in enumerate(qa_table_idx):
        order = np.argsort(-sims[i])  # descending
        rank = int(np.where(order == gold)[0][0]) + 1  # 1-indexed
        ranks.append(rank)
    ranks = np.array(ranks)
    out = {}
    for k in KS:
        out[f"R@{k}"] = float((ranks <= k).mean())
    out["MRR"] = float((1.0 / ranks).mean())
    return out


# ----------------------------- main -------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] Sampling tables and QAs...")
    table_ids, qas = sample_tables_and_qas()
    tid_to_idx = {tid: i for i, tid in enumerate(table_ids)}
    print(f"  sampled {len(table_ids)} tables, {len(qas)} QAs total")

    print("[2/5] Parsing tables...")
    parsed_tables = []
    skipped = []
    for tid in tqdm(table_ids):
        try:
            parsed_tables.append(load_table(TABLES_DIR / f"{tid}.json"))
        except Exception as e:
            skipped.append((tid, str(e)))
            parsed_tables.append(None)
    if skipped:
        print(f"  WARNING: {len(skipped)} tables failed to parse:")
        for s in skipped[:5]:
            print("   ", s)

    print("[3/5] Loading model...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    print("[4/5] Serializing + embedding tables (6 formats)...")
    fmt_results: dict[str, dict] = {}
    serialized_first: dict[str, str] = {}
    for fmt_name, fn in SERIALIZERS.items():
        print(f"  -- {fmt_name} --")
        serialized = []
        for parsed in parsed_tables:
            if parsed is None:
                serialized.append("")
                continue
            try:
                s = fn(parsed)
            except Exception as e:
                s = ""
                print(f"    serialize error in {fmt_name}: {e}")
            serialized.append(s)
        if serialized_first.get(fmt_name) is None and serialized:
            serialized_first[fmt_name] = serialized[0]
        avg_chars = float(np.mean([len(s) for s in serialized]))
        t0 = time.time()
        embs = model.encode(serialized, normalize_embeddings=True, show_progress_bar=False, batch_size=8)
        emb_time = time.time() - t0
        fmt_results[fmt_name] = {
            "embeddings": embs,
            "avg_chars": avg_chars,
            "embed_time_sec": emb_time,
        }
        print(f"    avg chars: {avg_chars:.0f} | embed time: {emb_time:.1f}s")

    print("[5/5] Embedding queries + evaluating...")
    queries = [BGE_QUERY_PREFIX + qa["question"] for qa in qas]
    qa_table_idx = [tid_to_idx[qa["table_id"]] for qa in qas]
    q_embs = model.encode(queries, normalize_embeddings=True, show_progress_bar=False, batch_size=16)

    final = {}
    for fmt_name, info in fmt_results.items():
        metrics = evaluate(q_embs, info["embeddings"], qa_table_idx)
        final[fmt_name] = {
            **metrics,
            "avg_chars": round(info["avg_chars"], 1),
            "embed_time_sec": round(info["embed_time_sec"], 2),
        }

    # ---- Print results table ----
    print("\n" + "=" * 80)
    print("RESULTS — HiTab serialization format vs retrieval accuracy")
    print(f"  N_tables = {len(table_ids)}, N_queries = {len(qas)}, model = {MODEL_NAME}")
    print("=" * 80)
    headers = ["Format"] + [f"R@{k}" for k in KS] + ["MRR", "AvgChars", "EmbedSec"]
    rows = []
    for fmt in SERIALIZERS:
        m = final[fmt]
        rows.append([
            fmt,
            *[f"{m[f'R@{k}']:.3f}" for k in KS],
            f"{m['MRR']:.3f}",
            f"{m['avg_chars']:.0f}",
            f"{m['embed_time_sec']:.1f}",
        ])
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # KEY FINDINGS
    print("\nKEY FINDINGS (R@10):")
    md_r10 = final["markdown"]["R@10"] * 100
    html_r10 = final["html"]["R@10"] * 100
    bc_r10 = final["breadcrumb"]["R@10"] * 100
    print(f"  Markdown   R@10 = {md_r10:.1f}%")
    print(f"  HTML       R@10 = {html_r10:.1f}%")
    print(f"  Breadcrumb R@10 = {bc_r10:.1f}%")
    print(f"  Breadcrumb − Markdown = {bc_r10 - md_r10:+.1f}pp")
    print(f"  Breadcrumb − HTML     = {bc_r10 - html_r10:+.1f}pp")
    print(f"  HTML       − Markdown = {html_r10 - md_r10:+.1f}pp")

    # ---- Save outputs ----
    out_json = {
        "config": {
            "n_tables": len(table_ids),
            "n_queries": len(qas),
            "seed": SEED,
            "model": MODEL_NAME,
            "ks": KS,
        },
        "table_ids": table_ids,
        "results_per_format": final,
    }
    json_path = OUTPUT_DIR / "pilot_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {json_path}")

    # Examples file
    examples_path = OUTPUT_DIR / "serialization_examples.txt"
    with open(examples_path, "w", encoding="utf-8") as f:
        f.write(f"First sampled table_id: {table_ids[0]}\n")
        f.write(f"Title: {parsed_tables[0]['title'] if parsed_tables[0] else '(parse failed)'}\n")
        f.write("=" * 80 + "\n\n")
        for fmt_name in SERIALIZERS:
            f.write(f"### {fmt_name.upper()}\n")
            s = serialized_first.get(fmt_name, "")
            f.write(s[:2000])
            if len(s) > 2000:
                f.write("\n... [truncated]")
            f.write("\n\n" + "-" * 80 + "\n\n")
    print(f"Saved -> {examples_path}")


if __name__ == "__main__":
    main()
