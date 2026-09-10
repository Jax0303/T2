# SPDX-License-Identifier: MIT
"""MultiHiertt 계측기가 기대는 두 가지: gold 좌표 규약과 층 이름.

MultiHiertt 의 ``table_evidence`` 는 ``"{표}-{행}-{열}"`` 문자열이고, 그 좌표는
**colspan/rowspan 을 펼친 격자** 위의 위치다. ``parse_html_table`` 도 같은 규약으로
펼친다 — 두 규약이 어긋나면 gold 가 엉뚱한 칸을 가리키고 정확도 수치 전체가
쓰레기가 되는데, 그 사고는 조용히 난다(값이 그럴듯하게 나온다). 그래서 데이터셋
자신의 셀 렌더링(``table_description``)과 대조하는 검사를 남긴다.

데이터셋 캐시가 없는 환경에서는 HTML 을 직접 써서 규약만 검사한다.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rag_agent.reconstruct import parse_html_table                    # noqa: E402

_VAL = re.compile(r"\bis\s+(.+?)\s*\.\s*$")
_norm = lambda s: re.sub(r"[,\s]", "", s or "")


def test_evidence_coords_are_expanded_grid_positions():
    """colspan 이 있는 표에서 "0-2-2" 가 가리키는 칸이 펼친 격자의 (2,2) 인가."""
    html = ("<table>"
            "<tr><td></td><td colspan='2'>2007</td><td colspan='2'>2006</td></tr>"
            "<tr><td></td><td>Carrying</td><td>Fair</td><td>Carrying</td><td>Fair</td></tr>"
            "<tr><td>Mortgages</td><td>584,795</td><td>603,200</td>"
            "<td>420,061</td><td>449,130</td></tr>"
            "</table>")
    grid = parse_html_table(html)
    # 헤더 첫 행은 blank-after-first 로 펼쳐진다: 스팬 원점에만 텍스트가 남는다
    assert grid[0][1] == "2007" and grid[0][2] == ""
    # 데이터 행은 열이 밀리지 않는다 — 이것이 gold 좌표가 기대는 정렬이다
    assert grid[2][2] == "603,200", grid[2]
    assert grid[2][4] == "449,130", grid[2]


def test_gold_coords_match_dataset_own_rendering():
    """실측: gold 좌표에서 우리가 읽은 값이 데이터셋 문장의 값과 같은가.

    캐시가 없으면 건너뛴다 — 이 검사는 데이터가 있는 환경에서만 의미가 있다
    (``CLAUDE.md`` §0.6: 있으면 추론으로 대신하지 않고 실측한다).
    """
    try:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from datasets import load_dataset
        rows = load_dataset("bevaya/MultiHiertt", split="train")
    except Exception as e:                                  # pragma: no cover
        import pytest
        pytest.skip(f"MultiHiertt cache unavailable: {e}")

    checked = matched = 0
    for row in rows:
        ev = row["table_evidence"] or []
        if not ev or row["text_evidence"]:
            continue
        desc = json.loads(row["table_description"])
        grids = [parse_html_table(h) for h in row["tables"]]
        for e in ev:
            t, i, j = (int(x) for x in e.split("-"))
            m = _VAL.search(desc.get(e, ""))
            if m is None:
                continue
            g = grids[t] if t < len(grids) else []
            got = g[i][j].strip() if i < len(g) and j < len(g[i]) else None
            checked += 1
            matched += int(got is not None and _norm(m.group(1)) == _norm(got))
        if checked >= 300:
            break
    assert checked >= 300, checked
    assert matched == checked, f"{checked - matched}/{checked} gold 좌표가 어긋난다"


def test_layer_names_come_from_dataset_fields_only():
    """층은 `program` 의 유무와 gold 셀 개수로만 갈린다 — 우리가 붙인 분류가 없다."""
    from retrieval_accuracy_mh import layer
    assert layer({"kind": "lookup", "gold": {1}}) == "lookup_m1"
    assert layer({"kind": "lookup", "gold": {1, 2}}) == "lookup_m2+"
    assert layer({"kind": "arith", "gold": {1, 2, 3}}) == "arith_m2+"


if __name__ == "__main__":
    test_evidence_coords_are_expanded_grid_positions()
    test_layer_names_come_from_dataset_fields_only()
    test_gold_coords_match_dataset_own_rendering()
    print("ok")
