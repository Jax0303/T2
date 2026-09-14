# SPDX-License-Identifier: MIT
"""표 고유 라벨 추출 (PREREG-2026-09-13-table-label.md)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from mh_arms import table_labels  # noqa: E402


def test_label_is_the_paragraph_right_before_the_marker():
    paras = ["Intro text.", "The following table presents sales by segment (in millions):",
             "## Table 0", "", "Cash Flows", "## Table 1",
             "n/a = not applicable " + "word " * 30, "## Table 2"]
    assert table_labels(paras, "none") == {}
    l1 = table_labels(paras, "L1")
    assert l1[0].startswith("The following table presents")
    assert l1[1] == "Cash Flows"                      # 빈 문단은 건너뛴다
    assert l1[2].startswith("n/a = not applicable")
    l2 = table_labels(paras, "L2")
    assert l2[0] and l2[1] == "Cash Flows"
    assert l2[2] == ""                                # ':' 도 없고 25단어 초과
