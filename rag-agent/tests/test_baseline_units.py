# SPDX-License-Identifier: MIT
"""기준선 색인 단위 둘의 계약 — `chunk` 와 `tablerag`.

둘 다 셀을 1:1 로 담지 않으므로 "이 단위가 어떤 셀을 배달하는가"(covers)가
정확해야 채점이 성립한다. chunk 는 살아있는 셀을 하나도 잃지 않고 두 번 세지도
않아야 하고, tablerag 는 숫자 열을 요약 하나로 접으면서 그 요약이 배달하는 셀을
min/max 두 개로만 주장해야 한다.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("ra", ROOT / "scripts/retrieval_accuracy.py")
ra = importlib.util.module_from_spec(_s)
_s.loader.exec_module(ra)


class T:
    """헤더 경로가 있는 3x3 — 숫자 열 둘, 범주 열 하나(값 하나가 중복)."""
    table_id = "t"
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


def test_chunk_partitions_cells_and_repeats_header():
    t = T()
    out = ra.markdown_chunks(None, t, "T", chunk_chars=1)   # 1 = 행마다 잘림
    assert len(out) == 3, "행마다 끊겼어야 한다"
    seen = [c for _txt, cs in out for c in cs]
    assert sorted(seen) == sorted((i, j) for i in range(3) for j in range(3)), "셀 손실/중복"
    for txt, _cs in out:
        assert "| a | b | label |" in txt, "청크마다 헤더가 반복돼야 한다"


def test_tablerag_folds_numeric_and_delivers_only_min_max():
    t = T()
    out = ra.tablerag_units(None, t, "leaf")
    docs = dict(out)
    assert len(out) == 4, "숫자 열 2 + 범주값 2 = 4 (셀 9개가 아니다)"
    schema = [d for d in docs if '"dtype"' in d]
    assert len(schema) == 2
    a = next(d for d in schema if '"column_name": "a"' in d)
    assert '"min": 10.0' in a and '"max": 30.0' in a
    assert docs[a] == [(0, 0), (1, 0)], "min 행과 max 행만 배달한다"
    kept = next(d for d in docs if '"cell_value": "kept"' in d)
    assert docs[kept] == [(0, 2), (1, 2)], "중복 제거된 범주 문서는 같은 값의 셀 전부를 배달"
    delivered = {c for cs in docs.values() for c in cs}
    # 닿지 않는 셀: a 열의 중간값 (2,0) 과 b 열의 중간값 (1,1)
    assert delivered == {(0, 0), (1, 0), (0, 1), (2, 1), (0, 2), (1, 2), (2, 2)}
    assert (2, 0) not in delivered and (1, 1) not in delivered, \
        "숫자 열의 min/max 가 아닌 셀은 어떤 질의로도 닿을 수 없다"


def test_rowcol_is_an_intersection_not_a_union():
    """RowColRetrieval 은 `df.iloc[row_ids, col_ids]` 다 — 합집합이 아니라 교집합.

    행만 뽑히고 그 표의 열이 안 뽑히면 아무 셀도 배달되지 않아야 하고, 다른 표의
    행과 열이 섞여도 교차 셀이 생기면 안 된다.
    """
    # 표 A 3x3, 표 B 3x3. covers 는 (tid, i, j).
    rowsA = [frozenset(("A", i, j) for j in range(3)) for i in range(3)]
    colsA = [frozenset(("A", i, j) for i in range(3)) for j in range(3)]
    rowsB = [frozenset(("B", i, j) for j in range(3)) for i in range(3)]
    covers = rowsA + rowsB + colsA
    is_row = [True] * 6 + [False] * 3
    texts = [f"u{i}" for i in range(len(covers))]

    # A 행 1개 + B 행 1개 만 상위, 열은 A 것만 -> 교차는 A 안에서만 생긴다
    order = [0, 3, 6]                      # A행0, B행0, A열0
    got, n, _ = ra.rowcol_select(order, covers, texts, is_row, budget=1, dump=0)
    assert got == {("A", 0, 0)}, got       # B 행은 A 열과 교차하지 않는다
    assert n == 1

    # 열이 하나도 안 뽑히면 배달 0
    got, n, _ = ra.rowcol_select([0, 1, 2], covers[:6] + colsA, texts,
                                 [True] * 6 + [False] * 3, budget=5, dump=0)
    assert got == set() and n == 0, got

    # k 가 자라면 k x k 부분표
    order = [0, 1, 2, 6, 7, 8]             # A행 3개, A열 3개
    got, n, _ = ra.rowcol_select(order, covers, texts, is_row, budget=9, dump=0)
    assert n == 9, n                       # 3x3 전체
