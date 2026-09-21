# SPDX-License-Identifier: MIT
"""중간 보고용 200문항 탐색 비교 — 본 방법 대 MT2Net (탐색용, 사전등록 아님).

ids_200.json 의 질의에서 두 답변 파일을 짝짓는다. 파일은 200건을 모두 담기만 하면 된다(1,047건
전체 출력도 받는다). 질의마다 문맥 해시가 기준 파일과 같고, 두 파일의 생성 설정과
모델 revision 이 같아야 한다. 하나라도 어긋나면 멈춘다.

보고: 층별·전체 표본 EM(Wilson 95% CI), 문항별 승패, EM 차이(본 방법 − MT2Net)의 짝지음
부트스트랩 95% CI(층 안에서 복원추출), 정확 McNemar p(보정 없음).

  PYTHONPATH=. .venv/bin/python results/mh_interim200/analyze.py \
      --out results/mh_interim200/interim200_qwen3_8b_cot.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from rag_agent.eval.artifacts import file_digest, write_pair  # noqa: E402

SEED, REPS, Z = 20260913, 10_000, 1.959964
SAME = ("reader", "prompt_sha256", "max_new_tokens", "seed", "batch_size", "scope",
        "condition", "header_rule")


def load(p):
    return {r["query_id"]: r for r in map(json.loads, Path(p).open(encoding="utf-8"))}


def wilson(k, n):
    p, d = k / n, 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def boot_ci(groups, rng):
    n = sum(len(v) for v in groups.values())
    stats = sorted(sum(a - b for v in groups.values() for a, b in (v[rng.randrange(len(v))] for _ in v)) / n
                   for _ in range(REPS))
    return [round(stats[int(.025 * REPS)], 4), round(stats[int(.975 * REPS) - 1], 4)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", default=str(HERE / "ids_200.json"))
    ap.add_argument("--ours", default=str(HERE / "mh_cell_hv2_answer_doc_qwen3_8b_cot_200.jsonl"))
    ap.add_argument("--mt2net", default=str(HERE / "mh_mt2net_answer_doc_qwen3_8b_cot_200.jsonl"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    spec = json.loads(Path(a.ids).read_text(encoding="utf-8"))
    ids = spec["ids"]
    paths = {"ours": a.ours, "mt2net": a.mt2net}
    arms = {k: load(p) for k, p in paths.items()}
    refs = {k: load(Path(a.ids).parent / f"ref_contexts_{k}_200.jsonl") for k in paths}
    for k, rows in arms.items():
        miss = [i for i in ids if i not in rows]
        bad = [i for i in ids if i in rows and rows[i]["context_sha256"] != refs[k][i]["context_sha256"]]
        if miss or bad:
            raise SystemExit(f"{k}: 없는 질의 {len(miss)}, 문맥 해시 불일치 {len(bad)}")
    summ = {k: json.loads(Path(p).with_suffix(".json").read_text(encoding="utf-8")) for k, p in paths.items()}
    diff = {f: [s.get(f) for s in summ.values()] for f in SAME if summ["ours"].get(f) != summ["mt2net"].get(f)}
    rev = [s["reader_details"].get("revision_resolved") for s in summ.values()]
    if diff or rev[0] != rev[1]:
        raise SystemExit(f"생성 설정이 다르다: {diff}, revision {rev}")

    rows = [{"query_id": i, "layer": arms["ours"][i]["layer"],
             "ours_em": int(arms["ours"][i]["answer_correct"]),
             "mt2net_em": int(arms["mt2net"][i]["answer_correct"]),
             "ours_retrieval": int(arms["ours"][i]["retrieval_correct"]),
             "mt2net_retrieval": int(arms["mt2net"][i]["retrieval_correct"]),
             "answer": arms["ours"][i]["answer"],
             "ours_pred": arms["ours"][i]["pred"], "mt2net_pred": arms["mt2net"][i]["pred"]} for i in ids]
    by = {}
    for r in rows:
        by.setdefault(r["layer"], []).append((r["ours_em"], r["mt2net_em"]))
    rng = random.Random(SEED)
    table = {}
    for name, groups in [*((L, {L: by[L]}) for L in sorted(by)), ("ALL", by)]:
        pairs = [p for v in groups.values() for p in v]
        n, ko, km = len(pairs), sum(x for x, _ in pairs), sum(y for _, y in pairs)
        w, l = sum(x > y for x, y in pairs), sum(x < y for x, y in pairs)
        table[name] = {"n": n, "em_ours": round(ko / n, 4), "em_ours_wilson95": wilson(ko, n),
                       "em_mt2net": round(km / n, 4), "em_mt2net_wilson95": wilson(km, n),
                       "ours_only_correct": w, "mt2net_only_correct": l,
                       "both_correct": sum(x and y for x, y in pairs),
                       "both_wrong": sum(not (x or y) for x, y in pairs),
                       "em_diff": round((w - l) / n, 4), "em_diff_bootstrap95": boot_ci(groups, rng),
                       "mcnemar_exact_p": binomtest(w, w + l).pvalue if w + l else 1.0}

    summary = {"exploratory": True, "preregistered": False,
               "note": "PREREG-2026-09-13-reader-qwen3.md 본 실행의 판정에 쓰지 않는다. p 보정 없음.",
               "ids_sha256": spec["ids_sha256"],
               "inputs_sha256": {str(p): file_digest(p) for p in
                                 (a.ids, a.ours, a.mt2net, *(Path(a.ids).parent / f"ref_contexts_{k}_200.jsonl"
                                                            for k in paths))},
               "generation_settings": {**{f: summ["ours"].get(f) for f in SAME},
                                       "revision_resolved": rev[0]},
               "source_sha256": {k: s["provenance"]["source_sha256"] for k, s in summ.items()},
               "git_commit": {k: s["provenance"]["git_commit"] for k, s in summ.items()},
               "bootstrap": {"reps": REPS, "seed": SEED, "resample": "층 안에서 짝 단위 복원추출"},
               "table": table}
    write_pair(Path(a.out), rows, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "inputs_sha256"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
