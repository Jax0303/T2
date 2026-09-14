# SPDX-License-Identifier: MIT
"""interim200 — 헤더 규칙 v2 대 v3.3 답변 EM, 본 방법·chunk. PLAN.md (생성 전 고정, 탐색용).

interim200(train)은 헤더 규칙 v3 의 결함 사례를 찾은 **사후 분석·개발에 사용된 표본**이다. 여기서 나온 차이는 규칙이
이 표본에 맞춰진 몫을 포함하므로 독립적인 미관측 평가가 아니다.

검사(어기면 멈춘다): 네 답변 파일이 ids_200.json 의 200 질의를 모두 담고, 요약의 records_sha256 과
retrieval_records_sha256 이 파일과 같고, 생성 설정·모델 revision 이 넷 다 같다. 새 검색 요약의 헤더 규칙이 기대값이고
라벨이 none 이며, 검색 인자가 header_rule·tag·out_dir 밖에서 v2 요약과 같다. 질의마다 layer·m 이 v2 와 같다.

묶음 A: arm 둘 × 칸 셋(전체 / 다중 조회 / 다중 산술), 새 규칙 대 v2. 정확 McNemar p 에 Holm(6),
EM 차이에 Bonferroni 동시 수준 짝지음 부트스트랩 CI(층 안 복원추출).
묶음 B: 새 규칙에서 본 방법 대 chunk, 칸 셋. 정확 McNemar p 에 Holm(3). v2 의 같은 칸을 나란히 싣는다(보정 없음).
기술: 문맥이 바뀐 질의 수, 검색 성공 변화, v2 불일치(본 방법 대 chunk)의 새 규칙 결과, 결함 사례 문서의 결과.

  PYTHONPATH=. .venv/bin/python results/mh_interim200_v33/analyze_v33.py \
      --out results/mh_interim200_v33/interim200_v33_qwen3_8b_cot.jsonl
  # 배관 점검(생성 전): 새 규칙 자리에 v2 파일을 넣는다
  PYTHONPATH=. .venv/bin/python results/mh_interim200_v33/analyze_v33.py --new-dir results/mh_interim200 \
      --new-retrieval-dir results/mh_arms --new-tag hv2 --expect-header-rule v2 --out <scratch>.jsonl
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

H2 = ROOT / "results/mh_interim200"
R2 = ROOT / "results/mh_arms"
ARMS = {"ours": "mh_cell", "chunk": "mh_chunk"}
CELLS = ("ALL", "lookup_m2+", "arith_m2+")
SAME = ("reader", "prompt_sha256", "max_new_tokens", "seed", "batch_size", "scope", "condition")
FREE_ARGS = ("header_rule", "tag", "out_dir")
# 보고용 결함 사례(문서 uid 앞 8자, PREREG-2026-09-14-header-v3.md). 판정·검사에는 쓰지 않는다.
CASES = ("ce87645c", "d4944be6", "8018bbdd", "04e25e10", "bd3f443d", "b8fe82b0", "b2585208", "60a5af4a", "9807e7e7")
SEED, REPS, ALPHA = 20260914, 20_000, 0.05


def load(p):
    return {r["query_id"]: r for r in map(json.loads, Path(p).open(encoding="utf-8"))}


def setting(s):
    return {**{f: s.get(f) for f in SAME}, "revision_resolved": s["reader_details"].get("revision_resolved")}


def paired(a, b, q, layer, rng, level):
    w = sum(a[i] > b[i] for i in q)
    l = sum(a[i] < b[i] for i in q)
    strata = {}
    for i in q:
        strata.setdefault(layer[i], []).append(a[i] - b[i])
    total = sum(rng.choice(np.array(d), size=(REPS, len(d))).sum(axis=1) for d in strata.values())
    lo, hi = np.quantile(total / len(q), [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"n": len(q), "first_only_correct": w, "second_only_correct": l, "em_diff": round((w - l) / len(q), 4),
            "diff_ci": [round(float(lo), 4), round(float(hi), 4)],
            "mcnemar_exact_p": binomtest(w, w + l).pvalue if w + l else 1.0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-dir", default=str(HERE), help="새 규칙 답변 파일 폴더")
    ap.add_argument("--new-retrieval-dir", default=str(HERE), help="새 규칙 검색 요약·레코드 폴더")
    ap.add_argument("--new-tag", default="hv33")
    ap.add_argument("--expect-header-rule", default="v3.3")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    spec = json.loads((H2 / "ids_200.json").read_text(encoding="utf-8"))
    ids = spec["ids"]
    rows, summ, retr, inputs = {}, {}, {}, {}
    for arm, base in ARMS.items():
        for ver, adir, rdir, tag in (("v2", H2, R2, f"{base}_hv2"),
                                     ("new", Path(a.new_dir), Path(a.new_retrieval_dir), f"{base}_{a.new_tag}")):
            p = adir / f"{tag}_answer_doc_qwen3_8b_cot_200.jsonl"
            rec = rdir / f"{tag}_records.jsonl"
            r, s = load(p), json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
            if file_digest(p) != s["records_sha256"]:
                raise SystemExit(f"{arm}/{ver}: 요약의 records_sha256 이 파일과 다르다")
            if file_digest(rec) != s["retrieval_records_sha256"]:
                raise SystemExit(f"{arm}/{ver}: 답변 요약이 가리키는 검색 레코드가 {rec} 와 다르다")
            miss = [i for i in ids if i not in r]
            if miss:
                raise SystemExit(f"{arm}/{ver}: ids_200 중 없는 질의 {len(miss)} (예: {miss[:3]})")
            rows[(arm, ver)], summ[(arm, ver)] = r, s
            retr[(arm, ver)] = json.loads((rdir / f"{tag}.json").read_text(encoding="utf-8"))
            inputs[f"{arm}/{ver}: {p.name}"] = s["records_sha256"]
        old, new = retr[(arm, "v2")]["arguments"], retr[(arm, "new")]["arguments"]
        if new.get("header_rule") != a.expect_header_rule or new.get("label_rule") != "none":
            raise SystemExit(f"{arm}: 새 검색의 헤더 규칙 {new.get('header_rule')}·라벨 {new.get('label_rule')}")
        diff = sorted(k for k in set(old) | set(new) if k not in FREE_ARGS and old.get(k) != new.get(k))
        if diff:
            raise SystemExit(f"{arm}: 검색 인자가 v2 와 다르다 {diff}")
        bad = [i for i in ids if (rows[(arm, "v2")][i]["layer"], rows[(arm, "v2")][i]["m"])
               != (rows[(arm, "new")][i]["layer"], rows[(arm, "new")][i]["m"])]
        if bad:
            raise SystemExit(f"{arm}: layer·m 이 v2 와 다른 질의 {len(bad)} (예: {bad[:3]})")
    settings = {k: setting(s) for k, s in summ.items()}
    if len({json.dumps(v, sort_keys=True) for v in settings.values()}) != 1:
        raise SystemExit(f"생성 설정이 다르다: {settings}")

    layer = {i: rows[("ours", "v2")][i]["layer"] for i in ids}
    em = {k: {i: int(r[i]["answer_correct"]) for i in ids} for k, r in rows.items()}
    ret = {k: {i: int(r[i]["retrieval_correct"]) for i in ids} for k, r in rows.items()}
    cell_ids = {c: [i for i in ids if c == "ALL" or layer[i] == c] for c in CELLS}
    table = {c: {f"{arm}/{ver}": {"n": len(q), "em": round(sum(em[(arm, ver)][i] for i in q) / len(q), 4),
                                  "retrieval_accuracy": round(sum(ret[(arm, ver)][i] for i in q) / len(q), 4),
                                  "marker_missing": sum(not rows[(arm, ver)][i].get("marker_found", True) for i in q),
                                  "mean_n_tok": round(float(np.mean([rows[(arm, ver)][i]["n_tok"] for i in q])), 1)}
                 for arm in ARMS for ver in ("v2", "new")} for c, q in cell_ids.items()}

    rng = np.random.default_rng(SEED)
    level_a = 1 - ALPHA / (len(ARMS) * len(CELLS))
    family_a = [{"arm": arm, "cell": c, "compare": "new vs v2",
                 **paired(em[(arm, "new")], em[(arm, "v2")], cell_ids[c], layer, rng, level_a)}
                for arm in ARMS for c in CELLS]
    for t, p in zip(family_a, holm([t["mcnemar_exact_p"] for t in family_a])):
        t["p_holm"] = p
    level_b = 1 - ALPHA / len(CELLS)
    family_b = [{"cell": c, "compare": f"ours vs chunk ({ver})",
                 **paired(em[("ours", ver)], em[("chunk", ver)], cell_ids[c], layer, rng, level_b)}
                for ver in ("new", "v2") for c in CELLS]
    for t, p in zip(family_b[:len(CELLS)], holm([t["mcnemar_exact_p"] for t in family_b[:len(CELLS)]])):
        t["p_holm"] = p

    changed = {arm: sum(rows[(arm, "v2")][i]["context_sha256"] != rows[(arm, "new")][i]["context_sha256"]
                        for i in ids) for arm in ARMS}
    retrieval_change = {arm: {"new_only_correct": sum(ret[(arm, "new")][i] > ret[(arm, "v2")][i] for i in ids),
                              "v2_only_correct": sum(ret[(arm, "new")][i] < ret[(arm, "v2")][i] for i in ids)}
                        for arm in ARMS}
    discord = {}
    for name, cond in (("v2_ours_only_correct", lambda i: em[("ours", "v2")][i] > em[("chunk", "v2")][i]),
                       ("v2_chunk_only_correct", lambda i: em[("ours", "v2")][i] < em[("chunk", "v2")][i])):
        q = [i for i in ids if cond(i)]
        cross = {}
        for i in q:
            key = f"ours_new={em[('ours', 'new')][i]},chunk_new={em[('chunk', 'new')][i]}"
            cross[key] = cross.get(key, 0) + 1
        discord[name] = {"n": len(q), "new_outcomes": dict(sorted(cross.items()))}
    cases = {c: {f"{arm}/{ver}": em[(arm, ver)][i] for arm in ARMS for ver in ("v2", "new")}
             | {"query_id": i} for c in CASES for i in ids if i.startswith(c)}

    out_rows = [{"query_id": i, "layer": layer[i],
                 "em": {f"{arm}/{ver}": em[(arm, ver)][i] for arm in ARMS for ver in ("v2", "new")},
                 "retrieval": {f"{arm}/{ver}": ret[(arm, ver)][i] for arm in ARMS for ver in ("v2", "new")},
                 "context_changed": {arm: int(rows[(arm, "v2")][i]["context_sha256"]
                                              != rows[(arm, "new")][i]["context_sha256"]) for arm in ARMS}}
                for i in ids]
    summary = {"exploratory": True, "plan": "results/mh_interim200_v33/PLAN.md",
               "sample_status": "사후 분석·개발에 사용된 표본(train interim200). 독립적인 미관측 test 평가가 아니다.",
               "header_rules": {"v2": "v2", "new": a.expect_header_rule}, "ids_sha256": spec["ids_sha256"],
               "inputs_sha256": inputs, "generation_settings": settings[("ours", "v2")],
               "inference": {"family_a": f"arm {len(ARMS)} × 칸 {len(CELLS)}, 정확 McNemar Holm, "
                                         f"부트스트랩 {REPS}회 층 안, Bonferroni 동시 수준 {level_a:.5f}",
                             "family_b": f"새 규칙 본 방법 대 chunk 칸 {len(CELLS)}, Holm; v2 칸은 보정 없이 병기",
                             "seed": SEED},
               "table": table, "family_a": family_a, "family_b": family_b,
               "contexts_changed": changed, "retrieval_change_200": retrieval_change,
               "v2_discordant_ours_vs_chunk": discord, "defect_case_documents": cases}
    write_pair(Path(a.out), out_rows, summary)
    for c in CELLS:
        print(c, {k: v["em"] for k, v in table[c].items()})
    for t in family_a + family_b:
        print(t.get("arm", ""), t["cell"], t["compare"], f"{t['first_only_correct']}:{t['second_only_correct']}",
              t["em_diff"], t["diff_ci"], f"p={t['mcnemar_exact_p']:.3g}", f"holm={t.get('p_holm', '-')}")
    print("contexts_changed", changed, "discord", discord)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
