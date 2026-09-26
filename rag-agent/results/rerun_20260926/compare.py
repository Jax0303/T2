"""2026-09-26 PREREG-2026-09-26-rerun-dev-alpha.md 집계 (리더 없음, 재검색 없음 — run.sh 출력만 읽는다).
항목 1: 비교군별 정확도, 기존 records 대비 불일치 문항 수, s3c 대비 McNemar(정확 이항, b = s3c만 맞힘, c = 비교군만) + Holm
        (가족 = 한 데이터셋·한 범위 안의 비교군 전부).
        HiTab = test 단일 셀 조회 991(mode all, m=1, aggregation none), MH = train 문서 안 채점 문항 전부.
항목 2: dev 선택(selection.json)과 test/train 적용 결과.
실행: .venv/bin/python results/rerun_20260926/compare.py
"""
import json
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
R1, R2 = RES / "rerun_20260926", RES / "dev_alpha_20260926"
ARMS = ["s3c", "sleaf", "table", "row", "chunk", "trag_hetero", "tablerag_leaf", "tablerag_path", "rowcol", "randrow"]
OLD_HITAB = {
    "gold": {"s3c": "retrieval_accuracy/t_s3c_gold_labelabl", "sleaf": "retrieval_accuracy/t_sleaf_gold",
             "chunk": "retrieval_accuracy/t_chunk_s3c_gold_v2", "trag_hetero": "retrieval_accuracy/t_trag_hetero_gold_v2",
             "tablerag_leaf": "retrieval_accuracy/t_tablerag_leaf_v2_gold",
             "tablerag_path": "retrieval_accuracy/t_tablerag_path_v2_gold",
             "rowcol": "retrieval_accuracy/t_rowcol_s3c_gold_v2", "randrow": "retrieval_accuracy/t_randrow_s3c_gold"},
    "split": {"s3c": "retrieval_accuracy/t_s3c_split_labelabl", "sleaf": "retrieval_accuracy/t_sleaf_split",
              "table": "problem_def_audit_20260925/table_split_truncate", "row": "retrieval_accuracy/t_row_values",
              "chunk": "evaluation_v2/chunk1000_v2", "trag_hetero": "evaluation_v2/huawei_char_v2",
              "tablerag_leaf": "retrieval_accuracy/t_tablerag_leaf_v2_split",
              "tablerag_path": "retrieval_accuracy/t_tablerag_path_v2_split", "rowcol": "evaluation_v2/rowcol_v2"}}
OLD_MH = {"s3c": "mh_arms/mh_train_cell_hv3.3u_none_doc"}


def hitab_sc(stem):
    out = {}
    for r in map(json.loads, open(f"{stem}_records.jsonl")):
        if "correct" in r and r["mode"] == "all" and r.get("m") == 1 and (r.get("aggregation") or "none") == "none":
            out[r["query_id"]] = int(r["correct"])
    return out


def mh_doc(stem):
    return {r["query_id"]: int(r["doc"]["correct"]) for r in map(json.loads, open(f"{stem}_records.jsonl"))
            if "doc" in r}


def mcnemar(ref, x):
    qs = sorted(set(ref) & set(x))
    b = sum(ref[q] and not x[q] for q in qs)
    c = sum(x[q] and not ref[q] for q in qs)
    return {"n": len(qs), "b": b, "c": c, "p": binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0}


def holm(rows):
    """rows 의 'p' 로 Holm 보정 p 를 'p_holm' 에 넣는다."""
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p"])
    m, run = len(rows), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * rows[i]["p"]))
        rows[i]["p_holm"] = run
    return rows


def prov(path):
    d = json.loads(Path(f"{path}.json").read_text())
    p = d.get("provenance") or {}
    return {"git_commit": (p.get("git_commit") or "")[:7], "git_dirty": p.get("git_dirty")}


def family(dataset, scope, new_stem, old_map, load):
    res = {}
    for arm in ARMS:
        stem = new_stem(arm)
        if not Path(f"{stem}.json").exists():
            continue
        new = load(stem)
        row = {"arm": arm, "n": len(new), "correct": sum(new.values()), "accuracy": round(sum(new.values()) / len(new), 4),
               "result": str(Path(f"{stem}.json").relative_to(ROOT)), "log": str(Path(f"{stem}.log").relative_to(ROOT)),
               **prov(stem)}
        if arm in old_map:
            old = load(str(RES / old_map[arm]))
            assert set(old) == set(new), (dataset, scope, arm)
            row.update(old_result=f"results/{old_map[arm]}.json", old_correct=sum(old.values()),
                       disagree=sum(old[q] != new[q] for q in new),
                       old_only=sum(old[q] and not new[q] for q in new), new_only=sum(new[q] and not old[q] for q in new))
        res[arm] = (row, new)
    ref = res["s3c"][1]
    tests = holm([{"arm": a, **mcnemar(ref, v[1])} for a, v in res.items() if a != "s3c"])
    for t in tests:
        res[t["arm"]][0]["vs_s3c"] = {k: t[k] for k in ("n", "b", "c", "p", "p_holm")}
    return [r for r, _ in res.values()]


out = {"item1": {}, "item2": {}}
for scope in ("gold", "split"):
    out["item1"][f"hitab_{scope}"] = family("hitab", scope, lambda a, s=scope: str(R1 / f"hitab/hitab_test_{s}_{a}"),
                                            OLD_HITAB[scope], hitab_sc)
out["item1"]["mh_doc"] = family("mh", "doc", lambda a: str(R1 / f"mh/mh_train_{a}"), OLD_MH, mh_doc)
for r in out["item1"]["mh_doc"]:
    d = json.loads((ROOT / r["result"]).read_text())
    r["by_layer_doc"] = {k: v["doc"]["accuracy_all"] for k, v in d["by_layer"].items() if isinstance(v, dict) and "doc" in v}

sel = json.loads((R2 / "selection.json").read_text()) if (R2 / "selection.json").exists() else None
if sel:
    out["item2"]["selection"] = sel
    applied = []
    for line in sel["apply"]:
        ds, scope, c = line.split()
        nm = "prefix" if c == "prefix" else f"a{c}"
        stem = R2 / ("apply/" + (f"hitab_test_{scope}_{nm}" if ds == "hitab" else f"mh_train_{nm}"))
        if not Path(f"{stem}.json").exists():
            continue
        x = hitab_sc(str(stem)) if ds == "hitab" else mh_doc(str(stem))
        applied.append({"dataset": ds, "scope": scope, "candidate": c, "n": len(x), "correct": sum(x.values()),
                        "accuracy": round(sum(x.values()) / len(x), 4),
                        "result": str(Path(f"{stem}.json").relative_to(ROOT)), **prov(str(stem))})
    out["item2"]["applied"] = applied
(R1 / "compare.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps(out, indent=1, ensure_ascii=False))
