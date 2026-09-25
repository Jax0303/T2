"""2026-09-26 MultiHiertt c / e 검색 정확도 층별 짝 비교(결과 파일 읽기 + 정확 McNemar 만).
c = results/mh_arms/mh_cell_hv2 (라벨 없음), e = results/mh_arms/mh_cell_hv2_L1 (라벨 L1). 두 실행의 arguments 는
label_rule·tag 만 다르다. Holm 은 4층×2범위 8개 묶음에만 건다(ALL 은 따로).
실행: .venv/bin/python results/components_20260925/mh_layers.py
"""
import json
from pathlib import Path
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
load = lambda s: {r["query_id"]: r for r in map(json.loads, open(ROOT / f"results/mh_arms/{s}_records.jsonl"))
                  if "excluded" not in r}
c, e = load("mh_cell_hv2"), load("mh_cell_hv2_L1")
assert set(c) == set(e) and len(c) == 2878
rows = []
for scope in ("doc", "corpus"):
    for layer in ("ALL", "lookup_m1", "lookup_m2+", "arith_m1", "arith_m2+"):
        q = [k for k in c if layer == "ALL" or c[k]["layer"] == layer]
        b = sum(c[k][scope]["correct"] and not e[k][scope]["correct"] for k in q)
        d = sum(e[k][scope]["correct"] and not c[k][scope]["correct"] for k in q)
        rows.append({"scope": scope, "layer": layer, "n": len(q),
                     "c_correct": sum(c[k][scope]["correct"] for k in q),
                     "e_correct": sum(e[k][scope]["correct"] for k in q),
                     "c_only": b, "e_only": d, "p_exact": binomtest(b, b + d).pvalue if b + d else 1.0})
fam = sorted((r for r in rows if r["layer"] != "ALL"), key=lambda r: r["p_exact"])
run = 0.0
for i, r in enumerate(fam):
    run = max(run, min(1.0, r["p_exact"] * (len(fam) - i)))
    r["p_holm8"] = run
(Path(__file__).parent / "mh_layers.json").write_text(json.dumps(rows, indent=1))
for r in rows:
    print(r)
