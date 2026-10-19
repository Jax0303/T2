"""HiTab 새 설정 사후 분석(기술 통계) — 다수결 기준 D 대 A 를 표 특성별로 나눠 본다: 표가 4청크를 넘는지, 표 셀 수, 머리글 경로 깊이. 불일치 문항 덤프.
  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/hitab_newregime_20261019/analyze.py run1 run2 run3
"""
import ast, json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "results/format_dev_20261017"), str(ROOT), str(ROOT / "scripts"), str(ROOT / "results/cellform_20261012"), str(ROOT / "results/score_fix_20261014")]
import score_formats2 as sf
from score_v2 import included_v2
from rag_agent.bench.hitab_grid import load_table
V = "R0X1F1B2D1C1E0"
BASE = Path(os.environ.get("HITAB_DIR", HERE))
REC = {arm: {r["query_id"]: r for r in map(json.loads, open(BASE / f"{arm}_records.jsonl", encoding="utf-8"))} for arm in ("A", "C", "D")}
N = len(REC["A"])
def gold_list(ans):
    if isinstance(ans, list): return [str(x) for x in ans]
    try:
        v = ast.literal_eval(str(ans)); return [str(x) for x in v] if isinstance(v, list) else [str(v)]
    except Exception: return [str(ans)]
import re
def norm_fl(fl):  # score_hitab.py 와 같은 정규화(앞에 0 없는 소수, 부동소수 표기 잡음)
    if fl is None: return None
    fl = re.sub(r"(?<![\d.])\.(\d+)", r"0.\1", fl)
    return re.sub(r"-?\d+\.\d{7,}", lambda m: format(float(m.group(0)), ".10g"), fl)
def fin(r, q):
    fl = norm_fl(sf.final_line(r["raw"])); return int(fl is not None and all(included_v2(fl, x, False) for x in gold_list(REC["A"][q]["answer"]))), fl
runs = [d for d in sys.argv[1:]]
S = {arm: [] for arm in ("A", "C", "D")}; FL = {arm: {} for arm in S}
for d in runs:
    for arm in S:
        p = BASE / d / arm / f"{V}.jsonl"
        if p.exists():
            rows = {r["query_id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}
            if len(rows) == N:
                S[arm].append({q: fin(r, q)[0] for q, r in rows.items()})
                if not FL[arm]: FL[arm] = {q: fin(r, q)[1] for q, r in rows.items()}
qs = sorted(REC["A"])
maj = {arm: {q: int(sum(s[q] for s in S[arm]) * 2 > len(S[arm])) for q in qs} for arm in S if S[arm]}
print("실행 수:", {arm: len(S[arm]) for arm in S}, "| 다수결 맞힘:", {arm: sum(m.values()) for arm, m in maj.items()})
# 표 특성
feat = {}
for q in qs:
    tab = load_table(REC["A"][q]["table_id"], "data/hitab"); t = tab.table
    feat[q] = {"chunks": REC["A"][q]["n_chunks_in_table"], "cells": t.n_rows * t.n_cols, "rdepth": max(len(t.row_path(i)) for i in range(t.n_rows)), "cdepth": max(len(t.col_path(j)) for j in range(t.n_cols)), "n_rows": t.n_rows}
def split(name, f):
    groups = {}
    for q in qs: groups.setdefault(f(q), []).append(q)
    print(f"\n== {name} ==\n| 묶음 | 문항 | A | C | D | D−A |\n|---|---|---|---|---|---|")
    for k in sorted(groups):
        g = groups[k]; a = sum(maj["A"][q] for q in g) if "A" in maj else 0; c = sum(maj["C"][q] for q in g) if "C" in maj else 0; d = sum(maj["D"][q] for q in g) if "D" in maj else 0
        print(f"| {k} | {len(g)} | {a} | {c} | {d} | {d-a:+d} |")
split("표가 4청크 안에 다 들어가는지", lambda q: "≤4청크(표 전체)" if feat[q]["chunks"] <= 4 else ">4청크(일부)")
split("표 셀 수", lambda q: "≤50" if feat[q]["cells"] <= 50 else ("51-150" if feat[q]["cells"] <= 150 else ("151-400" if feat[q]["cells"] <= 400 else ">400")))
split("행 머리글 깊이", lambda q: str(feat[q]["rdepth"]))
split("열 머리글 깊이", lambda q: str(feat[q]["cdepth"]))
split("정답 셀이 C 입력에 있는지", lambda q: "있음" if REC["C"][q]["correct"] else "없음")
if "A" in maj and "D" in maj:
    with open(BASE / "disagree_majority.txt", "w", encoding="utf-8") as f:
        for tag, cond in (("D 맞음·A 틀림", lambda q: maj["D"][q] and not maj["A"][q]), ("A 맞음·D 틀림", lambda q: maj["A"][q] and not maj["D"][q])):
            L = [q for q in qs if cond(q)]; f.write(f"##### {tag}: {len(L)}\n")
            for q in L:
                f.write(f"[{q[:8]}] 표 {feat[q]} 청크 {REC['A'][q]['n_chunks_in_table']}\nQ: {REC['A'][q]['question']}\ngold={REC['A'][q]['answer']!r}\nA 답={FL['A'].get(q)!r}\nD 답={FL['D'].get(q)!r}\n\n")
    print("\n불일치 덤프 → disagree_majority.txt")
# 사전등록 H2(PREREG-2026-10-19-hitab-confirm.md): 행 머리글 깊이 ≥3 대 ≤2 에서 D 대 A 다수결 McNemar + 상호작용(불일치 방향 비율 Fisher)
if "A" in maj and "D" in maj:
    from scipy.stats import binomtest, fisher_exact
    rows = []
    for name, cond in (("깊이 ≥3", lambda q: feat[q]["rdepth"] >= 3), ("깊이 ≤2", lambda q: feat[q]["rdepth"] <= 2)):
        g = [q for q in qs if cond(q)]; b = sum(maj["D"][q] and not maj["A"][q] for q in g); c = sum(maj["A"][q] and not maj["D"][q] for q in g)
        d = sum(maj["D"][q] for q in g); a = sum(maj["A"][q] for q in g); rows.append((b, c))
        print(f"\n[H2] {name}: 문항 {len(g)}, D {d} 대 A {a} ({100*(d-a)/len(g):+.1f}%p), b:c {b}:{c} p={binomtest(min(b,c), b+c, 0.5).pvalue if b+c else 1.0:.4f}")
    print(f"[H2] 상호작용(불일치 방향 비율, Fisher): p={fisher_exact(rows)[1]:.4f}")
