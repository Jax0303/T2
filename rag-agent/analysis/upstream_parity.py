# SPDX-License-Identifier: MIT
"""이식한 비교군 단위가 **원본 코드의 출력과 같은가** — 문자열로 대조한다.

원본을 읽고 옮겼다는 말 대신, 원본 함수를 그대로 실행해서 우리 포트의 출력과
맞춰 본다. 다르면 어디가 다른지 인쇄한다. 원본은 클론해 둔 저장소에서 읽는다:

  google-research/table_rag/agent/retriever.py   (Chen et al., NeurIPS 2024)
  yxh-y/TableRAG online_inference/…              (Huawei, arXiv 2506.10380)

  PYTHONPATH=. .venv/bin/python analysis/upstream_parity.py <clone_dir>
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rag_agent.bench import hitab_grid as hg                          # noqa: E402
from rag_agent.serialization import tablerag_unit as trag             # noqa: E402


# ---- 원본 코드 (google-research/table_rag/agent/retriever.py) --------------
# Document 는 page_content 만 쓰므로 문자열로 대신한다. 본문은 손대지 않는다.
def up_build_schema_corpus(df):
    docs = []
    for col_name, col in df.items():
        if col.dtype != 'object' and col.dtype != str:
            result_text = f'{{"column_name": "{col_name}", "dtype": "{col.dtype}", "min": {col.min()}, "max": {col.max()}}}'
        else:
            most_freq_vals = col.value_counts().index.tolist()
            example_cells = most_freq_vals[:min(3, len(most_freq_vals))]
            result_text = f'{{"column_name": "{col_name}", "dtype": "{col.dtype}", "cell_examples": {example_cells}}}'
        docs.append(result_text)
    return docs


def up_build_cell_corpus(df, max_encode_cell=10000):
    docs = []
    categorical_columns = df.columns[(df.dtypes == 'object') | (df.dtypes == str)]
    other_columns = df.columns[~(df.dtypes == 'object') | (df.dtypes == str)]
    if len(other_columns) > 0:
        for col_name in other_columns:
            col = df[col_name]
            docs.append(f'{{"column_name": "{col_name}", "dtype": "{col.dtype}", "min": {col.min()}, "max": {col.max()}}}')
    if len(categorical_columns) > 0:
        cell_cnt = Counter(df[categorical_columns].apply(
            lambda x: '{"column_name": "' + x.name + '", "cell_value": "' + x.astype(str) + '"}').values.flatten())
        docs += [cell for cell, _ in cell_cnt.most_common(max_encode_cell - len(docs))]
    return docs


def up_build_row_corpus(df):
    return ['|'.join(str(cell) for cell in row) for _, row in df.iterrows()]


def up_build_column_corpus(df):
    return ['|'.join(str(cell) for cell in column) for _, column in df.items()]


# ---- HiTab 표를 "평평하게 읽은" 데이터프레임 -------------------------------
def infer_dtype(df):
    """원본 `utils/utils.py: infer_dtype` — 이것을 빼면 원본과 다른 분기를 탄다.

    `table_text_to_df` 가 항상 부르므로 TableRAG 의 데이터프레임은 숫자열이
    숫자 dtype 이다. pandas 2.2 에서 `errors='ignore'` 가 사라졌으므로 같은
    의미(변환되면 쓰고 아니면 원래 열을 둔다)로 쓴다.
    """
    for col in df.columns:
        try:
            conv = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            continue
        df[col] = conv
    return df


def flat_frame(tab, colmode="leaf"):
    """TableRAG 이 받는 모양: 행 라벨이 보통 열 하나, 열 이름은 잎 라벨."""
    t = tab.table
    # dict 로 모으면 잎 라벨이 겹치는 열이 서로를 덮어써서 열이 사라진다 —
    # 원본은 평평한 프레임이라 같은 이름의 열이 그대로 둘 남는다.
    data = [[str(t.cell(r, c)) for r in range(t.n_rows)] for c in range(t.n_cols)]
    names = [trag._col_name(t, c, colmode) for c in range(t.n_cols)]
    df = pd.DataFrame({i: v for i, v in enumerate(data)})
    df.columns = names
    return infer_dtype(df)


def main() -> int:
    clone = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    tids = hg.table_ids("data/hitab")[:40]
    stats = Counter()
    diffs = []
    for tid in tids:
        tab = hg.load_table(tid, "data/hitab")
        if tab is None:
            continue
        t = tab.table
        df = flat_frame(tab)

        # --- TableRAG 셀/스키마 코퍼스 ---
        ours = {c.text for c in trag.serialize(t, column_name_mode="leaf")}
        theirs = set(up_build_cell_corpus(df))
        stats["trag_cells_ours"] += len(ours)
        stats["trag_cells_theirs"] += len(theirs)
        stats["trag_cells_shared"] += len(ours & theirs)
        if ours != theirs and len(diffs) < 3:
            diffs.append(("tablerag", tid, sorted(theirs - ours)[:2], sorted(ours - theirs)[:2]))

        # --- 행 단위 / 열 단위 (RowCol) ---
        # 우리 쪽 행 텍스트: line_text(..., row_text="values") 에서 제목 접두만 뺀다.
        from retrieval_accuracy import line_text
        live = [(i, j) for i in range(t.n_rows) for j in range(t.n_cols)
                if str(t.data[i][j]).strip()]
        for i in range(t.n_rows):
            cs = [(i, j) for (a, j) in live if a == i]
            if not cs:
                continue
            ours_row = line_text(t, cs, "", "s3c", "values")
            # 원본은 빈 칸도 'nan'/'' 로 싣는다. 우리는 빈 칸을 색인에서 빼므로
            # 값이 모두 차 있는 행에서만 대조한다.
            if len(cs) == t.n_cols:
                theirs_row = up_build_row_corpus(df)[i]
                lab = trag.fmt_value(t.row_path(i)[-1]) if t.row_path(i) else ""
                want = "|".join(([lab] if lab else []) + [str(t.data[i][j]) for i, j in cs])
                stats["row_full"] += 1
                stats["row_match_upstream_without_label"] += int(theirs_row == "|".join(
                    str(t.data[i][j]) for i, j in cs))
                stats["row_ours_equals_label_plus_values"] += int(ours_row == want)
    print("== TableRAG (Chen et al., NeurIPS 2024) 셀/스키마 코퍼스 ==")
    print(f"  우리 문서 {stats['trag_cells_ours']} / 원본 {stats['trag_cells_theirs']} / "
          f"공통 {stats['trag_cells_shared']}")
    print(f"  일치율(우리 기준) = {stats['trag_cells_shared'] / max(stats['trag_cells_ours'], 1):.4f}")
    print("== build_row_corpus ==")
    print(f"  값이 모두 찬 행 {stats['row_full']} 중 원본이 '값만 |조인' {stats['row_match_upstream_without_label']}, "
          f"우리가 '행라벨+값' {stats['row_ours_equals_label_plus_values']}")
    for kind, tid, only_theirs, only_ours in diffs:
        print(f"\n-- {kind} {tid}\n   원본만: {only_theirs}\n   우리만: {only_ours}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
