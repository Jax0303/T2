#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""colsig 가 "질문이 오답 쪽" 이라 한 105 건 전수 판정.

`results/colsig/VERDICT.md` §E 는 이 105 건을 **검색으로 불가**로 분류하고,
거기서 R@1 상한 dev .868 / test .827 이 나왔다. 그 분류는 질문과 두 칸의 헤더
경로 차집합을 기성품 bge-base 코사인으로 견준 것이다 -- **표면 비교**다.

이 파일은 105 건을 한 건씩 열어 사람이 판정한 결과다. 근거는 질문 · HiTab 원본
문장(`sub_sentence`) · 두 칸의 경로와 값이며, 원본 문장이 결정적이다: HiTab 질문은
그 문장에서 역으로 만들어지므로, 문장과 gold 가 맞는데 질문만 어긋났다면
**라벨 오류가 아니라 질문 결함**이다.

⚠️ **한 사람의 1회 판정이고 사전등록이 없다.** 판정 결과로 어떤 arm 도 고르지
않았고 어떤 점수도 바꾸지 않는다. 채점을 실제로 고치려면 **성공한 597 건에서도
표본을 뽑아 같은 기준으로 재야 한다** -- 실패만 감사하면 한쪽으로만 점수가 오른다.

출력 `results/audit/wrongside_105_verdict.json`.

  python3 analysis/wrongside_105_verdict.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "analysis")]

import pandas as pd                                                   # noqa: E402
from header_path_coverage import load_corpus                          # noqa: E402
from intable_failures_xlsx import grid, layout, md_parts              # noqa: E402

KINDS = {
    "판정기오류": "질문은 gold 칸을 가리킨다. colsig 가 틀렸다. "
                  "대개 1등 쪽 구절이 gold 쪽 구절을 부분문자열로 품어 코사인이 뒤집힌다 "
                  "('rate' ⊂ 'percent change in rate 2014 to 2015').",
    "동의어":     "질문이 gold 칸을 가리키는데 표현이 다르다 (gars = government-assisted refugees, "
                  "girls = 0 to 14 years). 어휘로는 안 잡히지만 의미로는 잡힌다.",
    "미명세":     "질문이 두 칸을 가르는 말을 아예 안 한다 (연도·단위·행렬 방향 미지정). "
                  "1등 칸이 더 나은 답도 아니다. 정답 처리할 근거가 없다.",
    "질문결함":   "gold 는 원본 문장과 맞는데 질문이 잘못 쓰였다. 1등 칸이 실제로 더 나은 답이다. "
                  "관대 채점을 논의할 수 있는 유일한 갈래.",
    "라벨오류":   "원본 문장을 봐도 gold 칸이 질문·문장과 안 맞는다. 데이터셋 결함.",
    "추론필요":   "질문이 gold 를 가리키지만 시간·부정·문맥 추론이 필요하다 "
                  "('before leaving for X' = X 이전 소속). 어휘로는 불가, 원리적으로는 가능.",
}

# 사람 판정: 케이스 번호 -> (갈래, 근거)
V = {
0:("미명세","질문이 total 인지 league 인지 안 밝힘. 둘 다 그럴듯"),
1:("라벨오류","질문·원문장 모두 '평균 자녀 수, privately sponsored' = 1등 칸. gold 는 '대졸 비율' 칸인데 값이 우연히 같음(2.2)"),
2:("라벨오류","원문장이 'third-plus generation' 이라 명시. gold 는 'all second generation'"),
3:("판정기오류","질문의 'revised down' 이 gold 'total revision' 을 가리킴. 긴 열 이름이 코사인을 먹음"),
4:("라벨오류","질문·원문장 모두 everton 1967-68 = 1등 칸. gold 는 bradford city 1971-72, 값만 우연히 2"),
5:("라벨오류","질문·원문장 모두 1996-97 league goals = 1등 칸. gold 는 other>apps, 값만 우연히 9"),
6:("라벨오류","질문·원문장 모두 stoke 1890-91 = 1등 칸. gold 는 career total, 값만 우연히 7"),
7:("판정기오류","질문의 '1920 to 1929' 는 통산을 뜻함 = gold total. 1등은 1920-21 한 시즌"),
8:("질문결함","원문장은 '174kg in his second attempt', gold 는 그 값. 질문의 'result' 가 'result' 열과 우연히 충돌"),
9:("판정기오류","질문 'only one' 이 gold 'one' 을 지목. 'five types of media' 가 'four or five' 와 잘못 매칭"),
10:("미명세","질문이 'port vale 에서 통산' 이라 두 칸을 다 가리킴. 값도 동일(499)"),
11:("라벨오류","질문·원문장 모두 1983-84 league cup = 1등 칸. gold 는 1985-86 other, 값만 우연히 1"),
12:("판정기오류","질문 'under 15' 가 gold '0 to 14' 를 지목. '2016' 이 1등 행과 잘못 매칭"),
13:("판정기오류","질문이 '차이' 를 물음 = gold 'difference'. 질문이 인용한 68.8/82.1 이 1등 칸과 매칭"),
14:("판정기오류","질문 자체가 '...a difference of 23.8 percentage points' 로 끝남 = gold"),
15:("판정기오류","차집합이 행 식별자(날짜·예산)라 비교가 무의미. 질문은 정상적으로 gold 행을 가리킴"),
16:("판정기오류","질문 'percentage of workers in class 3' = 소속 확률 = gold. 'item 3' 과 숫자만 매칭"),
17:("판정기오류","질문이 'reduced border costs' 와 'canada to the united states' 를 둘 다 명시 = gold. 오타 'rudeuced' 로 매칭 실패"),
18:("판정기오류","질문 'rates of robbery' = gold 'rate'. 1등 'percent change in rate 2014 to 2015' 가 'rate'+'2015' 를 다 품음"),
19:("미명세","'vietnamese refugees' 가 표의 두 축(출신지/이민범주)에 걸쳐 있음"),
20:("판정기오류","질문이 sunderland 를 명시 = gold. 1등은 carlisle united"),
21:("판정기오류","질문이 '소득' 을 물음 = gold 'women' 열. 1등 'female distribution' 은 분포 비율(100)"),
22:("판정기오류","질문 'physical or sexual assault' = gold 'total violent victimization'. 'sexual assault' 부분문자열이 1등과 매칭"),
23:("동의어","'all the other countries' = gold 'total, excluding syria'"),
24:("판정기오류","질문이 'not a visible minority (78.8%)' 를 값까지 명시 = gold. 'filipino' 가 1등과 매칭"),
25:("판정기오류","#18 과 동형 (quebec)"),
26:("미명세","질문이 행 백분율인지 열 백분율인지 안 밝힘. 헤더가 'row'/'column' 이라 의미도 없음"),
27:("동의어","'girls' = gold '0 to 14 years'"),
28:("판정기오류","#22 와 동형 (percent 열)"),
29:("판정기오류","질문 'non-visible minority' = gold 'not a visible minority'. 1등 'visible minority population' 과 어휘가 거의 같아 뒤집힘"),
30:("동의어","'overqualified in both 2006 and 2016' = gold 'persistent overqualification'"),
31:("판정기오류","'how many adults' = gold 'number'. 1등은 percent"),
32:("미명세","#26 과 동형 (new brunswick)"),
33:("판정기오류","#18 과 동형 (winnipeg)"),
34:("판정기오류","#18 과 동형 (saskatoon)"),
35:("판정기오류","'how many adults ... in total' = gold 'number'. 1등은 percent"),
36:("동의어","'changed by less than 10%' = gold 'changed by -9% to 9%'. 1등은 'decreased by 10% or more'"),
37:("동의어","#27 과 동형 (outside inuit nunangat)"),
38:("동의어","'which place did ... finish' = gold 'pos'"),
39:("미명세","#26 과 동형 (2010)"),
40:("판정기오류","질문이 'average ... per 100,000 per quarter' 라 명시 = gold 'average rate'"),
41:("동의어","'games did ... start' = gold 'gs'(games started)"),
42:("동의어","'nmo cannabis users' = gold 'non-medical only'. 1등은 'non-user'"),
43:("동의어","'gars' = gold 'government-assisted refugees'"),
44:("동의어","'inflation-adjusted' = gold 'constant 2009 $'. ⚠️ 다만 원문장의 0.8% 와 gold 3.2 가 안 맞음 — 별도 확인 필요"),
45:("판정기오류","'how many offences' = gold 'number'. 1등은 percent change"),
46:("동의어","#42 와 동형 (15 to 24)"),
47:("판정기오류","'how many adult offenders' = gold 'number'. 1등은 rate"),
48:("판정기오류","질문이 'time' 을 물음 = gold. 1등은 'driver' 열(값이 사람 이름)"),
49:("판정기오류","질문이 'canadian exports to the united states' 를 명시 = gold 방향"),
50:("동의어","'psrs' = gold 'privately sponsored refugees'"),
51:("판정기오류","질문이 두 하위유형을 모두 나열 = 상위 집계인 gold. gold 쪽 차집합이 비어 강제로 오답 쪽 분류됨"),
52:("판정기오류","질문 'a total goals from 1975-78' = gold total. gold 쪽 차집합이 비어 강제 분류"),
53:("질문결함","원문장은 '10.35초에 완주', gold 는 heat 칸. 질문의 'final result' 가 'final' 열과 충돌 — 질문만 보면 1등(9.93)이 맞다"),
54:("라벨오류","원문장이 '약 73%' 라 명시 = 1등 칸(73.1). gold 는 southeast asian(82.9)"),
55:("판정기오류","'how many men ... in total' = gold 'number of workers'. 1등은 55세 이상 비율"),
56:("판정기오류","원문장 'ending england's innings at 215' = gold(total, first innings). 1등은 개인 2이닝"),
57:("미명세","질문에 연도가 없음. gold 2013 / 1등 2007 어느 쪽도 근거 없음"),
58:("라벨오류","질문·원문장이 #17 과 글자까지 동일한데 gold 가 다른 칸(71.6 대 82.2). 같은 질문에 두 개의 라벨"),
59:("판정기오류","원문장 'in five seasons ... four goals' = gold(dagenham 통산 4). 1등은 career total 5"),
60:("미명세","#26 과 동형 (british columbia)"),
61:("판정기오류","#59 와 동형 (appearances)"),
62:("동의어","'according to the payq' = gold 'questionnaire-measured'. 1등은 accelerometer"),
63:("미명세","질문이 '몇 장 팔았나' 만 물음. gold 는 첫 주(90,103), 1등은 누계(115,497) — 질문만 보면 누계가 자연스럽다"),
64:("라벨오류","'how many yards did ... complete' 는 패싱 야드(4,257)를 뜻함. gold 는 sacked yds(254). 원문장도 깨져 있음"),
65:("판정기오류","질문이 'in any other neighbourhood' 를 명시 = gold 'other sidewalk or street'"),
66:("미명세","질문에 연도가 없음. gold 2015 / 1등 2017 projected"),
67:("판정기오류","#65 와 동형 (women)"),
68:("미명세","#26 과 동형 (manitoba)"),
69:("미명세","'percentage points of donation' 이 기부금 점유율인지 기부자 비율인지 불분명"),
70:("추론필요","질문의 'these two kinds' 가 앞 문장을 가리킴. 문맥 없이는 어느 집계인지 못 정함"),
71:("라벨오류","원문장이 '15%' 라 명시 = 1등 칸(total offences). gold 는 violent offences(16)"),
72:("미명세","질문이 두 데이터 출처(ceedd / ceedd-census)를 구분하지 않음"),
73:("질문결함","원문장은 's&e-related occupation', 질문은 's&e occupation' 으로 '-related' 를 빠뜨림 — 질문만 보면 1등(80,000)이 맞다"),
74:("미명세","#26 과 동형 (2014)"),
75:("미명세","#57 과 동형 (35 to 44, gold 2013 / 1등 2004)"),
76:("동의어","'after adjusting for inflation' = gold 'constant 2009 $millions'"),
77:("동의어","'constantdollar' = gold 'constant 2009 $millions'"),
78:("질문결함","질문·원문장 모두 그냥 'percentage-point difference'. gold 만 'adjusted' — 질문만 보면 1등(15.8)이 맞다"),
79:("미명세","질문이 어느 국가 상징인지 안 밝힘 (헌장/국기/국가...)"),
80:("미명세","#69 와 동형 (church)"),
81:("질문결함","#78 과 동형 (aged 55 and older). 원문장은 20.5 를 명시하나 질문은 'adjusted' 를 안 씀"),
82:("미명세","#69 와 동형 (shopping centre)"),
83:("질문결함","질문 'what was the growth rate in 2019' 만 보면 1등(revised growth 1.86)이 맞다. gold 는 개정폭(0.2)"),
84:("동의어","#77 과 동형 (2017 projected)"),
85:("질문결함","원문장은 'saint john', 질문은 'sherbrooke' — 주석자가 도시를 바꿔 씀. 검색기는 질문대로 sherbrooke 를 찾았다"),
86:("라벨오류","질문·원문장 모두 '1950-51 third division south' = 1등 경로. gold 는 통산 league apps(38)"),
87:("추론필요","'before leaving for the doncaster rovers' = 이전 소속 stoke. 어휘로는 doncaster 가 이김"),
88:("추론필요","'for stoke and then moved to oldham' = stoke. 어휘로는 oldham/1984-85 가 이김"),
89:("추론필요","#88 과 동형 (goals)"),
90:("동의어","'donny' = gold 'doncaster rovers'. 1등은 career total"),
91:("라벨오류","질문·원문장 모두 'in the united states' 인데 gold 는 saskatchewan. 표 자체가 캐나다 주별"),
92:("라벨오류","질문·원문장 모두 'inuvialuit region' 인데 gold 는 'outside inuit nunangat'"),
93:("미명세","#26 과 동형 (overall)"),
94:("판정기오류","질문이 'physically or sexually' 와 'in the 12 months' 를 둘 다 명시 = gold"),
95:("판정기오류","#94 와 동형 (number)"),
96:("판정기오류","질문 'percentage' = gold 'proportion of total retail sales'. 1등은 달러 금액"),
97:("동의어","'reallocation effect' = gold 'between plants'"),
98:("질문결함","원문장은 'women', 질문은 'men' — 주석자가 성별을 바꿔 씀"),
99:("동의어","'invisible minority' = gold 'not a visible minority'"),
100:("질문결함","원문장은 'decreases', 질문은 'increases' — 주석자가 방향을 바꿔 씀"),
101:("라벨오류","#91 과 동형 (males, northwest territories)"),
102:("판정기오류","행 헤더가 순위 숫자('1','2')뿐이라 차집합 비교가 무의미. 질문의 'largest producers' 가 1위=gold 를 지목"),
103:("판정기오류","질문 'total' = gold 'all federal obligations'. 1등은 0 인 칸"),
104:("질문결함","'what season is the 1972-73 season' 이 무의미하게 쓰임. gold 'finish'=2nd 는 순위이지 시즌 수가 아니다"),
}


def corpus():
    a = argparse.Namespace(
        dataset="hitab", cell_scheme="S3c", data_dir="data/hitab",
        population="hitab_dev_lookup_all", split="dev", title_mode="page",
        seed=42, max_queries=0, mh_queries=400,
        rhb_question_types=[], rhb_em_only=False)
    return load_corpus(a)


def main() -> int:
    C = corpus()
    x = ROOT / "results/audit/intable_failures_166.xlsx"
    d = pd.read_excel(x, sheet_name="케이스_166").fillna("")
    w = d[d["colsig판정"].str.contains("오답")].reset_index(drop=True)
    assert len(w) == 105 == len(V), (len(w), len(V))

    cases = []
    for i, r in w.iterrows():
        kind, why = V[i]
        cases.append({
            "no": int(i), "query_id": r["query_id"],
            "판정": kind, "근거": why,
            "질문": r["질문"],
            "원본_문장": r["원본 문장(질문의 출처)"],
            "정답": r["정답(gold)"],
            "표": {"id": r["표 id"], "제목": r["표 제목"], "크기": r["표 크기"]},
            "gold_셀": {"행경로": r["gold 행경로"], "열경로": r["gold 열경로"],
                        "값": r["gold 값"], "문장": r["gold 셀 문장"],
                        "행": int(r["gold 행"]), "열": int(r["gold 열"])},
            "1등_셀": {"행경로": r["1등 행경로"], "열경로": r["1등 열경로"],
                       "값": r["1등 값"], "문장": r["1등 셀 문장"],
                       "행": int(r["1등 행"]), "열": int(r["1등 열"]),
                       "관계": r["관계"]},
            "colsig": {"판정": r["colsig판정"], "margin": r["margin"],
                       "gold만": r["gold만 있는 구절"], "1등만": r["1등만 있는 구절"]},
            "리더": {c: {"예측": r[f"리더 {c} 예측"], "정답": r[f"리더 {c} 정답"]}
                     for c in ("gold", "top1", "top10")},
            "복구가능성": r["복구가능성"],
            "gold_순위": int(r["gold 순위"]),
            "표_원문": {
                "설명": "표 원문. r0 부터가 데이터 행이고 hdr 은 헤더 행이다. "
                        "gold 칸은 【】, 1등 칸은 «» 로 표시했다. "
                        "열 구분은 탭. 비어 보이는 칸은 병합 셀이며 경로는 헤더 트리에서 채워진다.",
                "행": grid(C, r["표 id"],
                           (int(r["gold 행"]), int(r["gold 열"])),
                           (int(r["1등 행"]), int(r["1등 열"]))).split("\n"),
            },
        })

    cnt = Counter(c["판정"] for c in cases)
    # colsig 가 "검색으로 불가" 로 센 것 중 실제로 그런 것
    hard = cnt["미명세"]
    out = {
        "무엇": "colsig 가 '질문이 오답 쪽' 으로 분류한 105건 전수 사람 판정",
        "출처": {"모집단": "hitab_dev_lookup_all (830)",
                 "설정": "models/bge-base-cell-ft-p0 · α=0.8 · title-mode page · S3c",
                 "colsig": "results/colsig/VERDICT.md §E",
                 "케이스표": "results/audit/intable_failures_166.xlsx"},
        "주의": ["한 사람의 1회 판정이고 사전등록이 없다.",
                 "이 판정으로 어떤 arm 도 고르지 않았고 어떤 점수도 바꾸지 않았다.",
                 "채점을 실제로 고치려면 성공한 597건에서도 표본을 뽑아 같은 기준으로 재야 한다.",
                 "판정 근거는 질문·HiTab 원본 문장(sub_sentence)·두 칸의 경로와 값이다."],
        "갈래_정의": KINDS,
        "집계": {k: {"건수": v, "105중": round(v / 105, 4), "830중": round(v / 830, 4)}
                 for k, v in cnt.most_common()},
        "판정": {
            "colsig_주장": "105건(표 안 실패의 63.3%)은 질문이 오답 칸을 가리키므로 검색으로 불가",
            "실측": f"검색으로 못 고치는 것은 미명세 {hard}건뿐이다 "
                    f"({hard/105:.1%}). 나머지 {105-hard}건은 질문이 gold 를 가리키거나"
                    f"(판정기오류 {cnt['판정기오류']} · 동의어 {cnt['동의어']} · 추론필요 {cnt['추론필요']}), "
                    f"질문/라벨 자체가 틀렸다(질문결함 {cnt['질문결함']} · 라벨오류 {cnt['라벨오류']}).",
            "함의": "R@1 상한 dev .868 / test .827 은 105건 전부를 불가로 놓고 계산됐다. "
                    "그 전제가 성립하지 않으므로 상한은 재계산해야 한다.",
        },
        "cases": cases,
    }
    p = ROOT / "results/audit/wrongside_105_verdict.json"
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[출력] {p}  {len(cases)}건")

    # 갈래별 대표 1건만 추린 파일. 대표는 그 갈래의 기전이 가장 선명한 건으로 손으로 골랐다.
    REP = {18: "판정기오류", 43: "동의어", 57: "미명세",
           85: "질문결함", 91: "라벨오류", 87: "추론필요"}
    rep = {
        "무엇": "위 105건 판정의 갈래별 대표 1건씩. 표 원문 포함",
        "고른 법": "각 갈래에서 기전이 가장 선명한 건을 손으로 골랐다. 대표성 통계는 아니다.",
        "원본": "results/audit/wrongside_105_verdict.json",
        "갈래_정의": KINDS,
        "집계": out["집계"],
        "cases": [c for c in cases if REP.get(c["no"]) == c["판정"]],
    }
    assert len(rep["cases"]) == 6, len(rep["cases"])
    q = ROOT / "results/audit/wrongside_examples.json"
    json.dump(rep, open(q, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[출력] {q}  {len(rep['cases'])}건 (갈래별 1건)")
    for k, v in cnt.most_common():
        print(f"  {k:6} {v:3}건  ({v/105:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
