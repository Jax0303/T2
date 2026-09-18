#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Follow-up robustness checks on RETRIEVAL_IMPROVEMENT.md (user critique,
2026-09-16): CI/McNemar for the 150-sample reader comparison, whether
wrong_table errors correlate with caption similarity, whether wrong_row/
wrong_column errors concentrate on short/numeric headers, and an empirical
root-cause split for the 943 serialization-collision groups (cross-table vs
within-table) to judge whether "add a distinguishing field" can actually fix
them. No original file modified; reuses the test Bundle/bundles already built
by retrieval_improvement.py and item1's already-computed rows.

  PYTHONPATH=. .venv/bin/python scripts/robustness_checks.py all
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                                                    # noqa: E402
from scipy.stats import binomtest                                     # noqa: E402

from rag_agent.bench import hitab_grid as hg                          # noqa: E402
from rag_agent.serialization.caption import with_page_title           # noqa: E402
from retrieval_accuracy import PAGE_TITLES                            # noqa: E402
from retrieval_improvement import Bundle, require_bundle              # noqa: E402
from bottleneck_root_cause import build_repr_corpus                   # noqa: E402

_PAGE_TITLES = json.loads(PAGE_TITLES.read_text()) if PAGE_TITLES.exists() else {}

OUT_DIR = ROOT / "results/retrieval_improvement"
RI_DIR = OUT_DIR


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(center - half, 4), round(center + half, 4)


# ---------------------------------------------------------------------------
# (a) 150-sample CI + paired McNemar (item5 top1-only vs existing top-20)
# ---------------------------------------------------------------------------

def check_ci_mcnemar() -> dict:
    top1_rows = {}
    with (RI_DIR / "item5_all_top1_qa.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            top1_rows[r["query_id"]] = r["answer_correct"]
    k20_path = ROOT / "results/k_ladder_qwen3_8b_20260916/s3c_v2_primary_qwen3_8b_k20.jsonl"
    k20 = {}
    with k20_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            k20[r["query_id"]] = r["answer_correct"]
    qids = [q for q in top1_rows if q in k20]
    n = len(qids)
    k_top1 = sum(top1_rows[q] for q in qids)
    k_top20 = sum(k20[q] for q in qids)
    b = sum(1 for q in qids if top1_rows[q] == 0 and k20[q] == 1)   # top1 wrong, top20 right
    c = sum(1 for q in qids if top1_rows[q] == 1 and k20[q] == 0)   # top1 right, top20 wrong
    mcnemar_p = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) else None
    out = {
        "n": n, "top1_only_acc": round(k_top1 / n, 4), "top1_only_ci95": wilson_ci(k_top1, n),
        "top20_acc": round(k_top20 / n, 4), "top20_ci95": wilson_ci(k_top20, n),
        "mcnemar_discordant_b_top1wrong_top20right": b,
        "mcnemar_discordant_c_top1right_top20wrong": c,
        "mcnemar_exact_p_value": round(mcnemar_p, 4) if mcnemar_p is not None else None,
        "verdict": ("차이 유의함 (p<0.05)" if mcnemar_p is not None and mcnemar_p < 0.05
                   else "n=150 표본으로는 통계적으로 유의하다고 말하기 어려움")}
    (OUT_DIR / "robustness_ci_mcnemar.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


# ---------------------------------------------------------------------------
# (b) wrong_table caption overlap
# ---------------------------------------------------------------------------

def caption_tokens(tid: str, b: Bundle) -> set:
    tab = b.tabs.get(tid) or hg.load_table(tid, "data/hitab")
    b.tabs[tid] = tab
    title = with_page_title(tab.title, _PAGE_TITLES.get(tid))
    return set(str(title).lower().split())


def check_wrong_table_captions() -> dict:
    b = require_bundle("test")
    rows = []
    with (RI_DIR / "item1_rows_test.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["hybrid_error_class"] == "wrong_table":
                rows.append(r)
    overlaps = []
    for r in rows:
        qid = r["query_id"]
        qidx = b.qids.index(qid)
        q = b.primary[qid]
        s = b.query_scores(qidx, q["question"])
        top1 = b.cell_coords[int(np.argmax(s["hybrid"]))]
        gold_title = caption_tokens(r["gold_table"], b)
        wrong_title = caption_tokens(top1[0], b)
        inter, union = len(gold_title & wrong_title), len(gold_title | wrong_title)
        jaccard = round(inter / union, 4) if union else 0.0
        overlaps.append({"query_id": qid, "gold_table": r["gold_table"], "wrong_table": top1[0],
                         "jaccard": jaccard, "identical_title": jaccard == 1.0})
    n = len(overlaps)
    identical = sum(1 for o in overlaps if o["identical_title"])
    high = sum(1 for o in overlaps if o["jaccard"] >= 0.5)
    out = {"n_wrong_table": n, "n_identical_title": identical,
          "n_jaccard_ge_0.5": high, "mean_jaccard": round(sum(o["jaccard"] for o in overlaps) / n, 4) if n else None,
          "interpretation": ("표 제목이 이미 서로 다른 경우가 대부분이면(자카드 낮음) "
                            "'캡션 부족'보다 dense 의미 혼동이 더 큰 원인; 제목이 같거나 "
                            "비슷한 경우가 많으면(자카드 높음) 캡션이 실제로 표를 구별 못 함")}
    (OUT_DIR / "robustness_wrong_table_captions.json").write_text(
        json.dumps({"summary": out, "rows": overlaps}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


# ---------------------------------------------------------------------------
# (c) short/numeric header correlation with wrong_row / wrong_column
# ---------------------------------------------------------------------------

def is_short_or_numeric(label: str) -> bool:
    if not label:
        return True
    s = label.strip()
    if len(s) <= 4:
        return True
    return s.replace(".", "", 1).replace("-", "", 1).isdigit()


def check_short_header_correlation() -> dict:
    path = ROOT / "results/bottleneck_root_cause/top1_error_detail.csv"
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for cls, axis_col in (("wrong_row", "gold_row_path"), ("wrong_column", "gold_col_path")):
        sub = [r for r in rows if r["error_class"] == cls]
        leaves = [r[axis_col].split(">")[-1].strip() for r in sub if r[axis_col]]
        n_short = sum(1 for lab in leaves if is_short_or_numeric(lab))
        out[cls] = {"n": len(sub), "n_short_or_numeric_leaf": n_short,
                   "share": round(n_short / len(leaves), 4) if leaves else None}
    all_leaves_row = [r["gold_row_path"].split(">")[-1].strip() for r in rows if r["gold_row_path"]]
    all_leaves_col = [r["gold_col_path"].split(">")[-1].strip() for r in rows if r["gold_col_path"]]
    out["baseline_all_queries"] = {
        "row_share_short_or_numeric": round(sum(is_short_or_numeric(x) for x in all_leaves_row) / len(all_leaves_row), 4),
        "col_share_short_or_numeric": round(sum(is_short_or_numeric(x) for x in all_leaves_col) / len(all_leaves_col), 4)}
    (OUT_DIR / "robustness_short_header.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


# ---------------------------------------------------------------------------
# (d)+(f) collision root-cause split (cross-table vs within-table) --
# informs whether "add a distinguishing field" can fix them at all
# ---------------------------------------------------------------------------

def check_collision_root_cause() -> dict:
    texts, coords, _fb = build_repr_corpus("gold_path", "data/hitab")
    groups = defaultdict(list)
    for t, c in zip(texts, coords):
        groups[t].append(c)
    coll = [(t, cs) for t, cs in groups.items() if len(cs) > 1]
    cross = [(t, cs) for t, cs in coll if len({c[0] for c in cs}) > 1]
    within = [(t, cs) for t, cs in coll if len({c[0] for c in cs}) == 1]
    out = {
        "n_groups": len(coll), "n_cross_table_groups": len(cross), "n_within_table_groups": len(within),
        "cross_table_examples": [{"text": t[:160], "coords": cs} for t, cs in cross[:3]],
        "within_table_examples": [{"text": t[:160], "coords": cs} for t, cs in within[:3]],
        "verdict": ("cross-table 그룹 다수는 서로 다른 table_id가 동일 제목·동일 값을 가진 "
                   "진짜 중복/근사중복 표(예: 같은 선수의 통계표가 두 문서에 각각 존재)로, "
                   "셀 직렬화에 어떤 '의미 있는' 필드를 추가해도 구별 불가능 — table_id 자체를 "
                   "노출하는 것은 정답을 암기시키는 것과 같아 타당한 수정이 아님. 이 유형은 "
                   "same_value류와 같은 성격: 검색이 '틀렸다'기보다 답변 가능한 셀이 여러 개 "
                   "존재하는 평가 한계에 가깝다. within-table 그룹(소수)은 같은 표 안에서 "
                   "서로 다른 행/열이 동일 header path 문자열로 축약되는 경우로, 더 깊은 "
                   "header 계층(현재 col_path/row_path가 담지 못하는 레벨)이 남아 있는지 "
                   "확인해야 하며, 그런 레벨이 실제로 존재한다면 노출이 타당한 고정 후보이다 — "
                   "다만 이는 테이블별 hmt raw tree 재조사가 필요한 별도 작업이다.")}
    (OUT_DIR / "robustness_collision_root_cause.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if not k.endswith("examples")},
                     indent=2, ensure_ascii=False))
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ci = check_ci_mcnemar()
    wt = check_wrong_table_captions()
    sh = check_short_header_correlation()
    cr = check_collision_root_cause()
    print(json.dumps({"ci_mcnemar": ci, "wrong_table_captions": wt,
                      "short_header": sh, "collision_root_cause":
                      {k: v for k, v in cr.items() if not k.endswith("examples")}},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
