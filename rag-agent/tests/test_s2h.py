"""S2h는 표 단위 구별자를 구조에서 뽑는다 -- 그 세 규칙만 지키면 된다.

1. 그 표의 최상위 축 레이블 중 이 셀의 경로에 없는 것을 붙인다
   (이미 문장에 있는 토큰을 반복하면 새 정보가 0이다)
2. 여럿이면 코퍼스에서 드문 것부터 -- 표를 특정하는 것은 흔한 라벨이 아니다
3. 붙일 것이 없으면 S2 그대로 둔다
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from corpus_dump_vs_cell import label_doc_freq, s2h_prefixes, table_top_labels


class _C:
    """Corpus 중 S2h가 읽는 두 필드만."""
    def __init__(self, paths, owner):
        self.cell_paths, self.cell_owner = paths, owner


def _corpus():
    # 표 A의 최상위 라벨 = {province, mining, year}, 표 B = {year, revenue}.
    # year만 두 표에 있으므로 흔하고, 나머지는 각 표를 특정한다.
    return _C(
        paths=[(["province", "ontario"], ["year", "2016"], 1),
               (["mining", "gold"], ["year", "2016"], 2),
               (["year", "1999"], ["revenue", "total"], 3)],
        owner=[("A", 0, 0), ("A", 1, 0), ("B", 0, 0)])


def test_경로에_없는_최상위_라벨만_붙는다():
    pre = s2h_prefixes(_corpus())
    assert pre[0] == "[mining] "      # A의 최상위 중 이 셀 경로에 없는 것
    assert pre[1] == "[province] "


def test_최상위가_전부_경로에_있으면_그대로():
    # 표 B의 셀은 year와 revenue를 이미 경로에 갖고 있다
    assert s2h_prefixes(_corpus())[2] == ""


def test_드문_라벨을_고른다():
    C = _C(paths=[(["a"], ["x"], 1), (["b"], ["x"], 2), (["c"], ["x"], 3),
                  (["b"], ["y"], 4)],
           owner=[("T", 0, 0), ("T", 1, 0), ("T", 2, 0), ("U", 0, 0)])
    assert label_doc_freq(C)["b"] == 2 and label_doc_freq(C)["c"] == 1
    # 표 T의 첫 셀에는 {b, c}가 남는데, U에도 있는 b가 아니라 c를 골라야 한다
    assert s2h_prefixes(C)[0] == "[c] "
    assert s2h_prefixes(C, k=2)[0] == "[b | c] "


def test_최상위_라벨은_두_축의_뿌리다():
    C = _corpus()
    assert table_top_labels(C)["A"] == {"province", "mining", "year"}
    assert table_top_labels(C)["B"] == {"year", "revenue"}


if __name__ == "__main__":
    test_경로에_없는_최상위_라벨만_붙는다()
    test_최상위가_전부_경로에_있으면_그대로()
    test_드문_라벨을_고른다()
    test_최상위_라벨은_두_축의_뿌리다()
    print("ok")
