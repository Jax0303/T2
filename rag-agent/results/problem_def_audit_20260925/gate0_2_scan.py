"""GATE 0-2: git 전체 기록(모든 ref)에서 표 전체 입력 계열 arm 이 있는 결과 JSON 을 읽어
arm 별 answer_em 을 뽑는다. 추론 없음 — git show 로 파일만 읽는다.
후보 경로 = gate0_2_candidate_paths.txt (git log --all -G 'whole_table|whole table|full[ _-]?table|table_md|통째|표 전체').
각 파일은 그 경로를 마지막으로 건드린 커밋에서 읽고, 그 커밋이 삭제면 부모에서 읽는다."""
import json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
WHOLE = ("dump", "cell2dump", "table_md", "goldtable", "base", "full_table", "whole_table",
         "fulltable", "table", "P3_whole_table")

def git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout

def load(path):
    c = git("log", "--all", "-1", "--format=%h", "--", path).strip()
    if not c:
        return None, None
    blob = git("show", f"{c}:{path}")
    ref = c
    if not blob:
        ref = c + "^"
        blob = git("show", f"{ref}:{path}")
    try:
        return ref, json.loads(blob)
    except Exception:
        return ref, None

def find(d, keys):
    for k in keys:
        cur = d
        for p in k.split("."):
            cur = cur.get(p) if isinstance(cur, dict) else None
        if cur is not None:
            return cur
    return None

def arm_em(d):
    """arm -> answer_em, 여러 스키마에서."""
    out = {}
    s = d.get("summary")
    if isinstance(s, dict):
        for arm, v in s.items():
            if isinstance(v, dict) and "answer_em" in v:
                out[arm] = v["answer_em"]
    o = d.get("overall")
    if isinstance(o, dict) and isinstance(o.get("accuracy"), dict):
        out.update(o["accuracy"])
    for k in ("answer_accuracy_strict",):
        if isinstance(d.get(k), dict):
            out.update({a: d[k][a] for a in ("base", "treat") if a in d[k]})
    return out

rows = []
for path in (HERE / "gate0_2_candidate_paths.txt").read_text().split():
    if not path.endswith(".json"):
        continue
    ref, d = load(path)
    if not isinstance(d, dict):
        continue
    arms = d.get("arms") if isinstance(d.get("arms"), dict) else {}
    em = arm_em(d)
    whole = sorted(set(arms) & set(WHOLE)) or sorted(set(em) & set(WHOLE))
    if not whole:
        continue
    rows.append({
        "ref": ref, "path": path,
        "experiment": find(d, ["experiment", "leg"]),
        "dataset": find(d, ["corpus.dataset", "dataset"]),
        "split": find(d, ["corpus.split"]),
        "population": find(d, ["population.name"]),
        "query_count": find(d, ["population.n", "overall.n", "overall.n_questions", "answer_accuracy_strict.n", "n"]),
        "encoder": find(d, ["env.embed_model", "encoder", "embed_model"]),
        "reader": find(d, ["reader", "solver", "pipeline.solver", "env.reader"]),
        "budget_tokens": find(d, ["budget_tokens"]),
        "retriever": find(d, ["retriever", "pipeline.retriever"]),
        "whole_arms": {a: arms.get(a) for a in whole},
        "answer_em": em,
    })
json.dump(rows, sys.stdout, ensure_ascii=False, indent=1)
