"""L(본 방법만 틀림 66)·W(LangChain만 틀림 54) 수동 분류 — Claude 1인 판정(2026-10-17), 번호는 L_ours_loses.txt / W_ours_wins.txt 순서."""
import csv, json
L = {
 "머리글 경로 오류(셀 문장 생성 결함)": [2,7,34,37,40,42,45,53,54,56],
 "다른 표의 같은 이름 셀과 혼동(이중 합산·엉뚱한 표)": [13,22,23,27,48,59,63,66],
 "같은 묶음 항목 열거 누락·총계 포함(합·평균·최대 문항)": [4,18,31,35,46,50,55,62,64],
 "정답 셀이 입력에 있는데 이름 비슷한 다른 셀 선택": [3,9,11,17,47,49,58],
 "시계열 순서 단서 상실(기준 100·끝 연도)": [16,26,57],
 "LLM 계산 실수": [10,20,39,41,43,61],
 "질문 해석 차이(LangChain 해석이 정답과 일치)": [14,25,28,30,36,44,52],
 "채점 운·채점 규칙(LangChain 본문에 정답 숫자 우연 포함 7, 본 방법 첫 yes/no 규칙 2)": [1,8,12,15,24,29,60,38,65],
 "검색 누락(정답 셀이 본 방법 입력에 없음)": [5,32,33],
 "LLM 이 바로 앞 셀을 무시": [6,19],
 "조건 단계 오류·기타": [21,51],
}
W = {
 "채점 운(본 방법 본문에 정답 숫자 우연 포함 16, LangChain 첫 yes/no 규칙 2)": [5,6,7,17,19,23,24,31,33,37,39,41,46,50,52,54,18,28],
 "LangChain 셀·열 선택 오류(넓은 표 열 어긋남·옆 행·소계)": [1,8,21,22,26,27,32,34,36,40,48,53],
 "LangChain 다른 표의 같은 이름과 혼동": [13,20,25,29,38,44],
 "LangChain 항목 열거 누락": [15,16,47,51],
 "LangChain 검색 누락(청크 4개에 정답 셀 없음)": [42,45,49],
 "질문 해석 차이(본 방법 해석이 정답과 일치)": [3,4,9,11,12,14,35,43],
 "LangChain 계산 실수": [2],
 "LangChain 거절·잘림": [10],
 "조건 단계 오류": [30],
}
assert sorted(sum(L.values(), [])) == list(range(1, 67)), sorted(sum(L.values(), []))
assert sorted(sum(W.values(), [])) == list(range(1, 55)), sorted(sum(W.values(), []))
sets = json.load(open("sets.json"))
with open("classification.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["set", "idx", "query_id", "category"])
    for cat, idxs in L.items():
        for i in idxs: w.writerow(["L_ours_wrong_lc_right", i, sets["L"][i-1], cat])
    for cat, idxs in W.items():
        for i in idxs: w.writerow(["W_lc_wrong_ours_right", i, sets["W"][i-1], cat])
print("L"); [print(f"  {len(v):2d}  {k}") for k, v in L.items()]
print("W"); [print(f"  {len(v):2d}  {k}") for k, v in W.items()]
