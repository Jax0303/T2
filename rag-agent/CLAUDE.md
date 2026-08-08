작업 전 RESEARCH_STRUCTURE.md를 읽을 것. 연구 구조 확정본. NEXT.md와 충돌 시 이 문서 우선.

## 검증된 외부 사실

이 목록 외의 MT2Net/MultiHiertt 수치 생성 금지.

출처: Zhao, Y., Li, Y., Li, C., Zhang, R. (2022). "MultiHiertt: Numerical
Reasoning over Multi Hierarchical Tabular and Textual Data." ACL 2022,
Vol.1 Long Papers, pp.6588-6600. arXiv:2206.01347

데이터: 10,440 QA / 2,513 docs / train 7,830, dev 1,044, test 1,566.
  문서당 평균 표 3.89개, 평균 1,645.9 단어, 문장 68.06개.
  표 평균 10.78행 x 4.97열. 평균 supporting fact 7.02개. 평균 추론 스텝 2.47.
  FinQA 예제 2,119개(전체 20.3%) 혼입 (§3.3).
연산자 (Table 7): Add / Subtract / Multiply / Divide / Exp. 5종 전부 이항.
검색기 (§4): 후보 문장을 질문과 concat하여 BERT-base 이진분류, top-10 사용.
  선형화는 셀 단위 + 계층 행/열 헤더 부착. 논문 예시 문장 원문:
  "For Innovation Systems of Segment, sales of product in 2018,
   Year Ended December 31 is 2,894"
검색 recall (§5.4): top-10 = 76.4%, top-15 = 80.8%.
Test 성능 (EM / F1):
  Longformer+Reasoning 2.86/6.23 | FR+TAPAS 7.67/10.04 | FR+NumNet 10.77/12.02
  TAGOP(RoBERTa-large) 17.81/19.35 | FR+Seq2Prog 24.58/26.30
  FinQANet(RoBERTa-large) 31.72/33.60
  MT2Net BERT-base 32.07/33.67 | BERT-large 33.25/34.98
        RoBERTa-base 34.32/36.17 | RoBERTa-large 36.22/38.43
  Human expert 83.12/87.03 (표본 60개, 전문가 2명, 3시간 제한, CI 미보고)
성능 분해 (Table 5, MT2Net RoBERTa-large, EM/F1):
  text-only 49.26/53.29 | table-only 36.77/38.55 | table-only >=2tables 24.32/24.96
  table-text 33.04/35.15 | table-text >=2tables 21.04/23.36
  1step 43.62/47.80 | 2step 34.67/37.91 | 3step 22.43/24.57 | >3step 15.14/17.19
오류 분석 (100 표본, Table 6): Wrong Operand/Span 43% | Missing Operand 21%
  | Wrong Program 19% | Lack of Domain Knowledge 4%
  Missing Operand 대표 사례: gold (1203+1437+1896+1774)/4, pred (1203+1774)/2

논문에 없는 것 (본 연구가 채우는 공백. 논문이 했다고 서술 금지):
  - 선형화 방식만 통제한 ablation 없음. 셀단위+헤더의 인과 기여도 미측정
  - oracle-retrieval 상한 성능 미보고
  - 집합 단위 검색 완전성 지표 미보고 (fact-level recall만 보고)
  - seed, 실행 횟수, 표준편차 미보고 (§5.2는 batch size 32만 명시)
  - 셀 좌표 수준 gold operand 주석 없음 (§3.2는 문장 단위 표시만 요구)
  - MultiHiertt 현재 SOTA 수치는 미검증 상태. 인용 금지
