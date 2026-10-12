"""42문항(정답 셀 다 들어갔는데 LLM 답 = 정답 아닌 셀 값) 수동 분류 — 사례별 판정은 아래 표, 2026-09-29 Claude 판독."""
import json
from collections import Counter
from pathlib import Path
D = Path(__file__).parent
CAT = {
 "답이 이름·연도인데 숫자로 답함": [6, 7, 12, 13, 19, 27, 28, 30],
 "셀 값과 우연히 같음(개수·연도 목록 답)": [14, 20, 37],
 "표에 있는 변화율 줄을 그대로 답함": [1, 24, 33, 35, 41],
 "다른 표의 같은 이름 줄": [4, 26, 31, 32],
 "합계(Total) 줄을 고름": [10, 11, 21, 23, 40, 42],
 "같은 항목의 다른 측정 열·다른 구역 줄": [9, 29, 36, 39],
 "입력 값으로는 LLM 답이 조건에 맞음(데이터셋 정답과 다름)": [3, 15, 16, 22],
 "질문에 없는 구분의 다른 줄도 조건에 맞음": [25, 34, 38],
 "기타(질문 글자대로 2·경로에 날짜 없음 5·최댓값 오판 8·비슷한 이름 17·질문 오타 18)": [2, 5, 8, 17, 18],
}
ex = json.load(open(D / "other42.json"))
ids = sorted(i for v in CAT.values() for i in v)
assert ids == list(range(1, len(ex) + 1)), "분류 누락·중복"
above = sum(min(k for k, _ in e["chosen"]) < min(k for k, _ in e["gold"]) for e in ex)
out = {"n": len(ex), "categories": {k: {"count": len(v), "cases": v} for k, v in CAT.items()},
       "chosen_ranked_above_all_gold": above,
       "by_layer": Counter(e["layer"] for e in ex)}
json.dump(out, open(D / "classify.json", "w"), ensure_ascii=False, indent=1)
for k, v in CAT.items(): print(len(v), k)
print("고른 줄이 모든 정답 줄보다 위 순위:", above, "/", len(ex))
