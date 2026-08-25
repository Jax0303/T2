"""S2h는 표 단위 구별자를 그 표의 헤더에서 뽑는다 -- 규칙 셋만 지키면 된다.

1. 이 셀의 경로에 이미 있는 라벨은 후보에서 뺀다 (반복은 새 정보가 0이다)
2. 남은 것 중 코퍼스에서 가장 드문 것을 고른다 -- 표를 특정하는 것은 흔한 라벨이 아니다
3. 후보가 없으면 S2 그대로 둔다
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from corpus_dump_vs_cell import label_doc_freq, s2h_prefixes, table_labels


class _C:
    """Corpus 중 S2h가 읽는 두 필드만."""
    def __init__(self, paths, owner):
        self.cell_paths, self.cell_owner = paths, owner


def _corpus():
    # 두 항공사 표가 revenue/2016을 공유한다 -- 그것만으로는 표가 구별되지 않는다.
    # T에만 있는 'aircraft fuel expense'가 T를 코퍼스에서 특정하는 라벨이다.
    return _C(
        paths=[(["revenue"], ["2016"], 1),
               (["aircraft fuel expense"], ["2016"], 2),
               (["revenue"], ["2016"], 3)],
        owner=[("T", 0, 0), ("T", 1, 0), ("U", 0, 0)])


def test_드문_라벨이_붙는다():
    C = _corpus()
    assert label_doc_freq(C)["revenue"] == 2                  # 두 표에 있다
    assert label_doc_freq(C)["aircraft fuel expense"] == 1     # T에만 있다
    assert s2h_prefixes(C)[0] == "[aircraft fuel expense] "


def test_경로에_있는_라벨은_후보가_아니다():
    # 이 셀은 이미 aircraft fuel expense를 경로에 갖고 있으므로 남는 것은 revenue뿐
    assert s2h_prefixes(_corpus())[1] == "[revenue] "


def test_붙일게_없으면_그대로():
    # 표 U의 라벨은 {revenue, 2016}이고 둘 다 이 셀의 경로에 있다
    C = _corpus()
    assert table_labels(C)["U"] == {"revenue", "2016"}
    assert s2h_prefixes(C)[2] == ""


def test_후보집합은_깊이를_가리지_않는다():
    # 최상위(depth-0)만 보면 'fuel'이 후보에서 빠져 표를 특정할 수 없다
    C = _C(paths=[(["expense", "fuel"], ["2016"], 1), (["expense", "labor"], ["2016"], 2)],
           owner=[("T", 0, 0), ("T", 1, 0)])
    assert table_labels(C)["T"] == {"expense", "fuel", "labor", "2016"}
    assert s2h_prefixes(C)[0] == "[labor] "


def test_k를_늘리면_더_붙는다():
    C = _C(paths=[(["a"], ["x"], 1), (["b"], ["y"], 2)], owner=[("T", 0, 0), ("T", 1, 0)])
    assert s2h_prefixes(C, k=2)[0] == "[b | y] "


if __name__ == "__main__":
    test_드문_라벨이_붙는다()
    test_경로에_있는_라벨은_후보가_아니다()
    test_붙일게_없으면_그대로()
    test_후보집합은_깊이를_가리지_않는다()
    test_k를_늘리면_더_붙는다()
    print("ok")
