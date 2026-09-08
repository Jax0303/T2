# SPDX-License-Identifier: MIT
"""기준선 색인 단위들의 계약.

이 단위들은 셀을 1:1 로 담지 않으므로 "이 단위가 어떤 셀을 배달하는가"(covers)가
정확해야 채점이 성립한다. 그리고 셋 다 **평탄한 읽기**를 전제한 방법이라, 우리
파서가 헤더와 데이터를 갈라 둔 것을 그대로 쓰면 baseline 이 부당하게 깎인다 —
아래 테스트가 그 세 지점을 잡는다: 청킹은 원본 격자의 헤더 블록을 담고, TableRAG
는 스텁 열을 범주 열로 갖고, rowcol 은 합집합이 아니라 교집합이다.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("ra", ROOT / "scripts/retrieval_accuracy.py")
ra = importlib.util.module_from_spec(_s)
_s.loader.exec_module(ra)


class T:
    """파싱된 3x3 — 숫자 열 둘, 범주 열 하나(값 하나가 중복)."""
    table_id = "A"
    data = [["10", "1.5", "kept"],
            ["30", "2.5", "kept"],
            ["20", "3.5", "other"]]
    n_rows, n_cols = 3, 3

    def cell(self, r, c):
        return self.data[r][c]

    def col_path(self, c):
        return ["top", ["a", "b", "label"][c]]

    def row_path(self, r):
        return ["side", f"r{r}"]


class Tab:
    """`hitab_grid.HitabTable` 대역 — raw 격자와 raw→parsed 축 매핑."""
    title = "T"
    table = T()
    raw = {"texts": [["stub", "a", "b", "label"],
                     ["r0", "10", "1.5", "kept"],
                     ["r1", "30", "2.5", "kept"],
                     ["r2", "20", "3.5", "other"]]}
    row_map = {1: 0, 2: 1, 3: 2}          # raw 행 → 데이터 행 (0 은 헤더)
    col_map = {1: 0, 2: 1, 3: 2}          # raw 열 → 데이터 열 (0 은 라벨)


def test_raw_lines_keeps_the_header_block_and_maps_cells_exactly():
    lines = ra.raw_lines(Tab(), T())
    assert len(lines) == 4
    head, head_cells = lines[0]
    assert "stub" in head and head_cells == [], "헤더 줄은 데이터 셀을 배달하지 않는다"
    for k, (txt, cs) in enumerate(lines[1:]):
        assert cs == [(k, 0), (k, 1), (k, 2)], (k, cs)
        assert f"r{k}" in txt, "행 라벨이 줄에 남아 있어야 한다"


def test_chunk_repeats_the_header_block_and_partitions_cells():
    out = ra.markdown_chunks(Tab(), T(), "T", chunk_chars=1)   # 1 = 행마다 끊김
    assert len(out) == 3, "데이터 행마다 끊겼어야 한다"
    seen = [c for _t, cs in out for c in cs]
    assert sorted(seen) == sorted((i, j) for i in range(3) for j in range(3))
    for txt, _cs in out:
        assert "stub" in txt and "# T" in txt, "청크마다 제목과 헤더 블록이 반복된다"


def test_tablerag_folds_numeric_and_keeps_the_stub_column():
    out = ra.tablerag_units(Tab(), T(), "leaf")
    docs = dict(out)
    schema = [d for d in docs if '"dtype"' in d]
    assert len(schema) == 2, "숫자 열 둘이 요약 하나씩으로 접힌다"
    a = next(d for d in schema if '"column_name": "a"' in d)
    assert '"min": 10.0' in a and '"max": 30.0' in a
    assert docs[a] == [(0, 0), (1, 0)], "min 행과 max 행만 배달한다"
    kept = next(d for d in docs if '"cell_value": "kept"' in d)
    assert docs[kept] == [(0, 2), (1, 2)], "중복 제거된 범주 문서는 같은 값의 셀 전부"
    stub = [d for d in docs if '"column_name": "stub"' in d]
    assert len(stub) == 3, "평탄한 읽기의 스텁 열도 범주 열이다"
    assert all(docs[d] == [] for d in stub), "행 라벨 문서는 데이터 셀을 배달하지 않는다"
    delivered = {c for cs in docs.values() for c in cs}
    assert (2, 0) not in delivered and (1, 1) not in delivered, \
        "숫자 열의 min/max 가 아닌 셀은 어떤 질의로도 닿을 수 없다"


def test_rowcol_is_an_intersection_not_a_union():
    rowsA = [frozenset(("A", i, j) for j in range(3)) for i in range(3)]
    colsA = [frozenset(("A", i, j) for i in range(3)) for j in range(3)]
    rowsB = [frozenset(("B", i, j) for j in range(3)) for i in range(3)]
    covers = rowsA + rowsB + colsA
    is_row = [True] * 6 + [False] * 3
    texts = [f"u{i}" for i in range(len(covers))]

    got, n, _ = ra.rowcol_select([0, 3, 6], covers, texts, is_row, budget=1, dump=0)
    assert got == {("A", 0, 0)}, got          # B 행은 A 열과 교차하지 않는다
    assert n == 1
    got, n, _ = ra.rowcol_select([0, 1, 2], covers[:6] + colsA, texts,
                                 [True] * 6 + [False] * 3, budget=5, dump=0)
    assert got == set() and n == 0, got       # 열이 없으면 배달 0 (합집합이면 아님)
    got, n, _ = ra.rowcol_select([0, 1, 2, 6, 7, 8], covers, texts, is_row,
                                 budget=9, dump=0)
    assert n == 9, n                          # 3x3 전체


def test_published_row_unit_is_values_not_our_sentence():
    t = T()
    cells = [(0, 0), (0, 1), (0, 2)]
    v = ra.line_text(t, cells, "T", "s3c", "values")
    assert v == "T | r0|10|1.5|kept", v        # 행 라벨 + 값, 헤더 경로 없음
    sent = ra.line_text(t, cells, "T", "s3c", "sentence")
    assert "side" in sent and "top" in sent, "sentence 모드는 우리 헤더 경로를 담는다"
