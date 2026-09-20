# 사전등록: 같은 LLM 필터를 8개 arm 전부에 똑같이 적용 — **산술(다중 피연산자)** 모집단 (2026-09-21, 2단계)

## 이 문서의 위치

1단계(`PREREG-2026-09-21-fair-llm-filtering-hitab.md`, 단일조회 n=300)는 **이미 끝났고
결과를 알고 있다.** 그 사실을 숨기지 않고 적는다 — 아래 예측은 1단계 결과를 본 상태에서
세운 것이므로 "독립적인 사전등록"이 아니라 **"1단계 결과를 전제로 한 외삽의 사전등록"**이다.

1단계 결과 요약(`results/SUMMARY_TABLES-2026-09-18.md` §4):

    ours .7500 → .7667 (필터, p=.568 무의미)   1등
    mt2net .7433 → .7367 (p=.875)              ours 와 p=.211 → 무승부
    chunk .6867 → .4033 (p<1e-5)               크게 하락
    trag_hetero .6000 → .3933 (p<1e-5)         크게 하락
    rowcol .4700 → .3333 (p<1e-5)              크게 하락
    randrow/path/leaf                          바닥권, 변화 작음

1단계의 기전: 필터가 청크류의 **마크다운 헤더 줄**을 버려서 남은 값 줄이 자립하지 못한다
(무필터→필터 회귀 사례 중 1번 줄을 버린 비율 chunk 93%, trag_hetero 99%, rowcol 83%,
ours 9%). 셀 문장은 줄마다 자기 헤더를 품어 한 줄만 남아도 자립한다.

## 왜 산술을 따로 재는가

단일조회는 정답 셀이 **하나**다. 필터가 줄 2개만 남겨도 그 하나가 살아 있으면 맞는다.
산술은 **피연산자가 여럿**(m≥2)이라, 필터가 공격적으로 줄일수록 피연산자 중 일부가
사라져 계산 자체가 불가능해진다. 즉 **필터의 위험이 구조적으로 더 크다.** 1단계에서
관찰한 "ours 의 자립 문장 이점"이 이 조건에서도 유지되는지가 이 실험의 질문이다.

## 모집단

`results/fair_filter_arith_population_216.json` — **n=216 전수**(표본추출 없음).
`retrieval_accuracy.py` 의 `query_type == "arithmetic"` 분할이며, 8개 arm 전부가 216건을
모두 담고 있음을 실행 전에 확인했다(제외 0건). m 분포는 m=2 가 최빈이고 m≥3 도 포함된다.

**mt2net 만 `_arithmetic_records.jsonl` 파일이 없다.** 조용히 빼지 않는다 — 대신 같은
실행(tag `t_mt2net_gold`)의 전체 records(1,584건)를 이 216건으로 필터해 쓴다. 같은 실행의
같은 문맥이므로 다른 arm 과 조건이 어긋나지 않는다(`load_evidence` 의 sha256·context_version
무결성 검사도 통과함).

## 개입 — 1단계와 달라지는 것 하나

필터 프롬프트에 **"계산에 필요한 값 줄을 모두 포함하라"**를 추가한다. 1단계 프롬프트는
값 줄 하나와 그 헤더를 전제로 쓰였고, 산술에 그대로 쓰면 필터가 값 줄 하나만 남기도록
유도되어 **과제 자체가 불가능해진다**. 이것은 과제 성격에 맞춘 조정이며 **결과를 보고
고치는 것이 아니라 실행 전에 고정한다.** 두 프롬프트를 나란히 적어 비교 가능하게 둔다.

1단계(단일조회):

    You select which context lines are needed to answer a question about a table.
    Reply with only the line numbers, comma-separated, in increasing order.
    Include the line that holds the value AND any header or label lines needed to
    interpret that value. Do not explain, do not repeat the lines — output numbers only.

2단계(산술) — 굵은 부분만 추가:

    You select which context lines are needed to answer a question about a table.
    The question requires a CALCULATION over SEVERAL values, so you must include
    EVERY line that holds a value the calculation needs — not just one.
    Reply with only the line numbers, comma-separated, in increasing order.
    Include the lines that hold the values AND any header or label lines needed to
    interpret them. Do not explain, do not repeat the lines — output numbers only.

나머지는 1단계와 동일하게 고정한다: 줄 단위 압축(유닛 선택 아님), 필터·리더 모두
`local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit`, temperature 0, 리더 프롬프트 `neutral`,
채점 `hitab_exact_match_text`, `mode=all`(피연산자 셀이 전부 문맥에 있어야 검색 성공),
예산 20, `--corpus gold`, 같은 질의의 무필터/필터를 같은 실행에서 페어링.

## 예측 (실행 전 고정)

- **P1. 필터는 모든 arm 에서 1단계보다 더 해롭거나 같을 것이다.** 구체적으로 ours 의 Δ가
  1단계(+.0167)보다 낮아질 것으로 예측한다 — **ours Δ ≤ 0** (즉 산술에서는 필터가
  ours 에도 도움이 안 되거나 해로울 것).
- **P2. ours 가 필터 적용 후 답변정확도 1등을 유지한다.** 단, 1단계처럼 mt2net 과는
  유의차가 없을(무승부) 가능성이 높다고 예측한다.
- **P3. 청크류(chunk/trag_hetero/rowcol)의 하락폭이 1단계보다 더 클 것이다** — 헤더 줄
  손실에 피연산자 손실이 더해지므로.
- **P4. 절대 수준은 전 arm 이 1단계보다 크게 낮다.** 산술은 리더 자체의 천장이 낮다
  (`CLAUDE.md §5`: 나눗셈 몫 소수 자릿수 등 리더의 산술 정밀도 한계가 이미 기록돼 있다).
  ours 필터 후 정확도를 **0.15~0.40 구간**으로 예측한다.
- **P5. 피연산자 손실률**(필터 후 정답값들이 문맥에서 사라진 비율)이 청크류에서 ours 보다
  높을 것으로 예측한다.

## 시나리오별 해석 — 실행 전 고정 (결과를 보고 고르지 않는다)

- **(가) ours 1등 유지, mt2net 과 유의차 없음** → 1단계와 같은 결론을 산술에서도 재현한
  것으로 적는다. "ours 만의 이점"이 아니라 **"자립적 셀 문장 단위의 이점"**이라고 쓴다
  (mt2net 도 같은 셀 문장 단위이므로). 오늘 세션의 셀렉터 공정성 실험이 내린 결론과 같다.
- **(나) ours 1등 유지, mt2net 에도 유의하게 앞섬** → 그대로 적되, 1단계에서는 무승부였다는
  사실을 함께 적어 과장하지 않는다.
- **(다) 청크류 중 하나가 ours 를 역전** → 그대로 싣고, "필터링이 붙으면 ours 의 이점이
  사라진다"고 정직하게 적는다. 프롬프트를 고쳐 재실행해 뒤집으려 하지 않는다.
- **(라) 전 arm 이 바닥(예: 0.1 미만)이라 구분이 안 됨** → 산술에서는 이 비교가 판별력이
  없다고 적고, 순위를 주장하지 않는다.
- **(마) 필터가 ours 에 도움이 됨(Δ>0)** → P1 이 틀린 것이므로 P1 이 틀렸다고 적는다.

어느 경우든 **하이퍼파라미터·프롬프트를 사후에 고쳐 재실행하지 않는다**(`CLAUDE.md §7`).

## 측정

arm 별: 검색 정확도(mode=all), 무필터/필터 답변정확도, Δ, McNemar p, 검색성공시 정답률,
필터 입력 토큰 평균, 줄 수(입력→남김), 폴백 수. 필터 적용 후 **ours vs 각 arm 페어드
McNemar**로 1등·무승부 판정. 3단 분해: ① 검색 ② 필터가 근거(정답값 전부)를 남겼는가
③ 리더. 산술 특유 실패로 **피연산자 손실률**(무필터에선 정답값이 문맥에 다 있었는데
필터 후 하나라도 사라진 비율)을 arm 별로 센다.

## 산출물

`scripts/fair_filter_eval.py --task arithmetic`(1단계 스크립트 재사용, 신규 스크립트 없음),
`results/fair_filter_arith_20260921/{rows.jsonl,summary.json}`,
`results/SUMMARY_TABLES-2026-09-18.md` §5.
