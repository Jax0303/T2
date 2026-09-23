# SPDX-License-Identifier: MIT
"""interim200 다섯 arm 답변 EM — 본 방법 대 네 비교군 (탐색용, 사전등록 아님). PLAN_ARMS.md.

검사: 여섯 파일 모두 ids_200.json 의 200 질의를 담고, 질의마다 문맥 해시가 기준 파일과 같고, 요약의
records_sha256 이 파일과 같으며, 생성 설정·모델 revision 이 본 방법과 같아야 한다. 어긋나면 멈춘다.

칸: 전체 / 다중 조회 / 다중 산술. 추론: 본 방법 대 다섯 arm × 세 칸 = 15개 비교를 한 묶음으로,
정확 McNemar p 에 Holm 보정, EM 차이에는 Bonferroni 동시 수준 짝지음 부트스트랩 CI(층 안 복원추출).

  PYTHONPATH=. .venv/bin/python results/mh_interim200/analyze_arms.py \
      --out results/mh_interim200/interim200_arms_qwen3_8b_cot.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from analysis.mh_final_report import holm                    # noqa: E402
from rag_agent.eval.artifacts import file_digest, write_pair  # noqa: E402

ARMS = {"ours": "mh_cell_hv2", "chunk": "mh_chunk_hv2",
        "huawei": "mh_huawei_hv2", "rowcol": "mh_rowcol_hv2", "tablerag": "mh_tablerag_allobj_path_hv2"}
CELLS = ("ALL", "lookup_m2+", "arith_m2+")
SAME = ("reader", "prompt_sha256", "max_new_tokens", "seed", "batch_size", "scope",
        "condition", "header_rule")
SEED, REPS, ALPHA = 20260913, 20_000, 0.05


def load(p):
    return {r["query_id"]: r for r in map(json.loads, Path(p).open(encoding="utf-8"))}


def setting(s):
    return {**{f: s.get(f) for f in SAME},
            "revision_resolved": s["reader_details"].get("revision_resolved")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(HERE), help="답변 파일·기준 문맥 해시·ids_200.json 폴더")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    H = Path(a.dir)
    spec = json.loads((H / "ids_200.json").read_text(encoding="utf-8"))
    ids = spec["ids"]
    paths = {k: H / f"{t}_answer_doc_qwen3_8b_cot_200.jsonl" for k, t in ARMS.items()}
    refs = {k: H / f"ref_contexts_{k}_200.jsonl" for k in ARMS}
    rows, summ = {}, {}
    for k, p in paths.items():
        rows[k], ref = load(p), load(refs[k])
        miss = [i for i in ids if i not in rows[k]]
        bad = [i for i in ids if i in rows[k] and rows[k][i]["context_sha256"] != ref[i]["context_sha256"]]
        if miss or bad:
            raise SystemExit(f"{k}: 없는 질의 {len(miss)}, 문맥 해시 불일치 {len(bad)}")
        summ[k] = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        if file_digest(p) != summ[k]["records_sha256"]:
            raise SystemExit(f"{k}: 요약의 records_sha256 이 파일과 다르다")
    diff = {k: setting(s) for k, s in summ.items() if setting(s) != setting(summ["ours"])}
    if diff:
        raise SystemExit(f"생성 설정이 본 방법과 다르다: {diff} (본 방법 {setting(summ['ours'])})")

    layer = {i: rows["ours"][i]["layer"] for i in ids}
    em = {k: {i: int(rows[k][i]["answer_correct"]) for i in ids} for k in ARMS}
    cell_ids = {c: [i for i in ids if c == "ALL" or layer[i] == c] for c in CELLS}
    table = {c: {k: {"n": len(q), "em": round(sum(em[k][i] for i in q) / len(q), 4),
                     "retrieval_accuracy": round(sum(int(rows[k][i]["retrieval_correct"]) for i in q) / len(q), 4),
                     "marker_missing": sum(not rows[k][i].get("marker_found", True) for i in q)}
                 for k in ARMS} for c, q in cell_ids.items()}

    rng = np.random.default_rng(SEED)
    n_tests = (len(ARMS) - 1) * len(CELLS)
    level = 1 - ALPHA / n_tests
    tests = []
    for k in list(ARMS)[1:]:
        for c in CELLS:
            q = cell_ids[c]
            w = sum(em["ours"][i] > em[k][i] for i in q)
            l = sum(em["ours"][i] < em[k][i] for i in q)
            strata = {}
            for i in q:
                strata.setdefault(layer[i], []).append(em["ours"][i] - em[k][i])
            total = sum(rng.choice(np.array(d), size=(REPS, len(d))).sum(axis=1) for d in strata.values())
            lo, hi = np.quantile(total / len(q), [(1 - level) / 2, 1 - (1 - level) / 2])
            tests.append({"arm": k, "cell": c, "n": len(q), "ours_only_correct": w, "other_only_correct": l,
                          "em_diff": round((w - l) / len(q), 4),
                          "diff_ci_bonferroni": [round(float(lo), 4), round(float(hi), 4)],
                          "mcnemar_exact_p": binomtest(w, w + l).pvalue if w + l else 1.0})
    for t, p in zip(tests, holm([t["mcnemar_exact_p"] for t in tests])):
        t["p_holm"] = p

    out_rows = [{"query_id": i, "layer": layer[i], "em": {k: em[k][i] for k in ARMS},
                 "retrieval": {k: int(rows[k][i]["retrieval_correct"]) for k in ARMS}} for i in ids]
    summary = {"exploratory": True, "preregistered": False,
               "note": "탐색용. 전체 1,047 본 실행(PREREG-2026-09-13-reader-qwen3.md)의 판정에 쓰지 않는다.",
               "plan": "results/mh_interim200/PLAN_ARMS.md", "ids_sha256": spec["ids_sha256"], "arms": ARMS,
               "inputs_sha256": {p.name: file_digest(p) for p in [*paths.values(), *refs.values()]},
               "generation_settings": setting(summ["ours"]),
               "code": {k: {"git_commit": s["provenance"]["git_commit"],
                            "source_sha256": s["provenance"]["source_sha256"]} for k, s in summ.items()},
               "inference": {"family": f"본 방법 대 {len(ARMS) - 1} arm × {len(CELLS)} 칸 = {n_tests}",
                             "p": "정확 McNemar, Holm 보정",
                             "ci": f"짝지음 부트스트랩 {REPS}회, 층 안 복원추출, Bonferroni 동시 수준 {level:.5f}",
                             "seed": SEED},
               "table": table, "tests": tests}
    write_pair(Path(a.out), out_rows, summary)
    for c in CELLS:
        print(c, {k: v["em"] for k, v in table[c].items()})
    for t in tests:
        print(t["arm"], t["cell"], f"{t['ours_only_correct']}:{t['other_only_correct']}", t["em_diff"],
              t["diff_ci_bonferroni"], f"p_holm={t['p_holm']:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
