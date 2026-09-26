"""2026-09-26 PREREG-2026-09-26-rerun-dev-alpha.md 항목 2 — dev 결과만 보고 후보를 고른다.

후보 순서 ORDER = 1.0, 0.9, …, 0.1, prefix (a<α> = 라벨 섞기 비율 α, prefix = 라벨을 문장 접두어로).
- HiTab dev: 단일 셀 조회(type_accuracy.single_cell) 맞힌 수, 범위(gold = 질문의 표 안, split = dev 표 한 색인)마다 따로.
- MH dev: 문서 안(records 의 doc.correct) 맞힌 수, 채점 문항 전부(표 근거 있는 질의, --keep-hybrid).
- 데이터셋별 최고 = 맞힌 수 최대. 공통 최고 = (HiTab 표 안 정확도 + MH 문서 안 정확도) / 2 최대.
- 동률은 ORDER 에서 앞선 후보.
출력: selection.json, apply_list.txt ("<hitab|mh> <gold|split|doc> <후보>" 한 줄씩, 중복 없음).
"""
import json
from fractions import Fraction
from pathlib import Path

D = Path(__file__).parent
ORDER = ["1.0", "0.9", "0.8", "0.7", "0.6", "0.5", "0.4", "0.3", "0.2", "0.1", "prefix"]
name = lambda c: "prefix" if c == "prefix" else f"a{c}"


def hitab(scope, c):
    d = json.loads((D / f"hitab_dev/hitab_dev_{scope}_{name(c)}.json").read_text())
    sc = d["type_accuracy"]["single_cell"]
    return sc["success"], sc["n"]


def mh(c):
    rows = [json.loads(l) for l in open(D / f"mh_dev/mh_dev_{name(c)}_records.jsonl")]
    rows = [r for r in rows if "doc" in r]
    return sum(r["doc"]["correct"] for r in rows), len(rows)


table = {"hitab_gold": {c: hitab("gold", c) for c in ORDER},
         "hitab_split": {c: hitab("split", c) for c in ORDER},
         "mh_doc": {c: mh(c) for c in ORDER}}
for k, v in table.items():
    assert len({n for _s, n in v.values()}) == 1, (k, v)        # 후보마다 같은 문항 수


def best(score):
    top = max(score(c) for c in ORDER)
    return next(c for c in ORDER if score(c) == top)            # 동률은 ORDER 앞


sel = {k: best(lambda c, v=v: v[c][0]) for k, v in table.items()}
acc = lambda k, c: Fraction(*table[k][c])
sel["common"] = best(lambda c: (acc("hitab_gold", c) + acc("mh_doc", c)) / 2)
apply = []
for line in (f"hitab gold {sel['hitab_gold']}", f"hitab split {sel['hitab_split']}", f"mh doc {sel['mh_doc']}",
             f"hitab gold {sel['common']}", f"hitab split {sel['common']}", f"mh doc {sel['common']}"):
    if line not in apply:
        apply.append(line)
out = {"order": ORDER, "dev": {k: {c: {"success": s, "n": n, "accuracy": round(s / n, 4)} for c, (s, n) in v.items()}
                               for k, v in table.items()},
       "common_score": {c: round(float((acc("hitab_gold", c) + acc("mh_doc", c)) / 2), 4) for c in ORDER},
       "selected": sel, "apply": apply}
(D / "selection.json").write_text(json.dumps(out, indent=1))
(D / "apply_list.txt").write_text("\n".join(apply) + "\n")
print(json.dumps(out, indent=1))
