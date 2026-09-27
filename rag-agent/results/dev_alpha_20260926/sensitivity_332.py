"""α 선택 민감도(2026-09-27): MultiHiertt dev 를 표 근거만 필요한 332건으로 좁혀 후보 11개를 다시 센다. 새 실행 없음.

실행:  python3 results/dev_alpha_20260926/sensitivity_332.py  -> sensitivity_332.json
- 후보·맞힘 정의·동률 규칙은 select_alpha.py 와 같다(맞힌 수 최대, 동률은 ORDER 에서 앞선 후보).
- 순위 두 가지: rank = 위 동률 규칙으로 끊은 순위(선택에 쓰는 순위), rank_tied = 맞힌 수가 같으면 같은 순위(1, 2, 2, 2, 5 …).
- 332건 = results/reader_format_20260927/mh_dev_pop332.jsonl (911건 중 본문 문장 근거가 필요한 579건을 뺀 것).
- 선택은 바꾸지 않는다(PREREG-2026-09-26-rerun-dev-alpha.md 의 선택 = 911건 기준 α=1.0). 결과만 보고한다.
"""
import json
from pathlib import Path

D = Path(__file__).parent
ROOT = D.parents[1]
ORDER = ["1.0", "0.9", "0.8", "0.7", "0.6", "0.5", "0.4", "0.3", "0.2", "0.1", "prefix"]
name = lambda c: "prefix" if c == "prefix" else f"a{c}"
keep = {json.loads(l)["query_id"] for l in open(ROOT / "results/reader_format_20260927/mh_dev_pop332.jsonl")}
assert len(keep) == 332


def tied_ranks(scores):
    """맞힌 수가 같으면 같은 순위. 순위 = 1 + 자기보다 많이 맞힌 후보 수."""
    return {c: 1 + sum(1 for d in ORDER if scores[d] > scores[c]) for c in ORDER}


def ranks(scores):
    """맞힌 수 내림차순, 동률은 ORDER 순서. 후보 -> 순위(1부터)."""
    order = sorted(ORDER, key=lambda c: (-scores[c], ORDER.index(c)))
    return {c: i + 1 for i, c in enumerate(order)}


res = {"tie_rule": "rank = 맞힌 수 내림차순, 동률은 ORDER(1.0, 0.9, …, 0.1, 접두어)에서 앞선 후보(select_alpha.py 와 같음). "
                   "rank_tied = 맞힌 수가 같으면 같은 순위.", "candidates": {}}
s911, s332 = {}, {}
for c in ORDER:
    rows = [json.loads(l) for l in open(D / f"mh_dev/mh_dev_{name(c)}_records.jsonl")]
    rows = [r for r in rows if "doc" in r]
    sub = [r for r in rows if r["query_id"] in keep]
    assert len(rows) == 911 and len(sub) == 332, c
    s911[c] = sum(r["doc"]["correct"] for r in rows)
    s332[c] = sum(r["doc"]["correct"] for r in sub)
sel = json.loads((D / "selection.json").read_text())
assert all(sel["dev"]["mh_doc"][c]["success"] == s911[c] for c in ORDER)   # 911건 값이 선택 파일과 같은지
r911, r332 = ranks(s911), ranks(s332)
t911, t332 = tied_ranks(s911), tied_ranks(s332)
for c in ORDER:
    res["candidates"][c] = {"n911_correct": s911[c], "n911_accuracy": round(s911[c] / 911, 4), "rank911": r911[c], "rank911_tied": t911[c],
                            "n332_correct": s332[c], "n332_accuracy": round(s332[c] / 332, 4), "rank332": r332[c], "rank332_tied": t332[c]}
res["best911"] = min(ORDER, key=lambda c: r911[c])
res["best332"] = min(ORDER, key=lambda c: r332[c])
res["selection_kept"] = sel["selected"]["mh_doc"]
(D / "sensitivity_332.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
