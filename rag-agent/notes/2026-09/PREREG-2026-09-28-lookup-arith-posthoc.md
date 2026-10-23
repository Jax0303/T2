결과 확인 후 설계한 탐색적 분석

# 조회 대 산술 1위 적중률 차이 사후 분석 — 정의 고정 (2026-09-28)

원고 5.7절(05_results.md:190)의 조회 대 산술 1위 적중률 차이를 기존 records 로만 다시 센다. 모델·임베딩·검색은 다시 돌리지 않는다.
이 문서는 아래 계산(GATE 2)을 실행하기 전에 커밋한다. 다만 1위 적중률, 머리글 답 여부, 비교 단어 분류별 1위 적중 수는
이미 본 뒤에 이 설계를 만들었다(`results/lookup_vs_arith_20260928/rank1.json`, `why_rank1.json`, `posthoc/wordclass.json`).
구별 후보 수 m, 무작위 순서 기준값, T5 위치, 표준화·부트스트랩 값은 아직 계산하지 않았다.

## 대상과 입력

- MultiHiertt train, 질문이 속한 문서 안에서 검색, s3c 셀 문장, α=.7, 예산 20. 정답 셀 1개 문항만.
- **주 분석 = 최종 버전(v3.3u):** `results/rerun_20260926/mh/mh_train_s3c_records.jsonl`. 조회 212, 산술 71.
  1위 = `doc.context[0]` 이 정답 칸 문장과 같음(`rank1.py` 판정과 같음). 1위 칸 = 같은 문서에서 그 문장을 가진 색인 칸.
- **처음 버전(v1), T7:** `results/mh_arms/mh_train_cell_hv1_none_doc_kladder_records.jsonl` 의 `correct_at.doc['1']`.
  처음 버전이 채점하지 못한 조회 1문항(`9631214aea0f414ebe6e3763979862f5`, gold_in_header)은 경로가 없어 T7 에서 뺀다 → 조회 211, 산술 71.
  원고 5.7절의 분모 212 와 다르다는 점을 결과에 적는다.
- 코드: `results/lookup_vs_arith_20260928/posthoc/common.py` (커밋 a9c971c). 문항별 행을 만든다.

## 정의

**(a) 머리글 답 여부.** `rank1.py`(aac902b)의 `is_header` 그대로: 정답 문자열을 정규화(소문자, `[a-z0-9.]` 밖 문자 삭제, 양끝 `.` 삭제,
숫자면 Decimal 정규형)한 값이 정답 칸 행 경로 + 열 경로 원소 하나의 정규화 값과 같으면 "예". 경로는 그 버전 표 객체의 경로(식별자 원소 포함, rank1.py 와 같음).

**(b) 비교 단어 분류.** `common.py` 의 `CATS` 로 고정. 위에서부터 처음 맞는 분류, 대소문자 무시, 단어 경계 `\b`.

| 순서 | 분류 | 정규식 |
|---|---|---|
| 1 | 두 번째 | `second` |
| 2 | 기준값 비교 | `exceed\w*`, `greater than`, `less than`, `more than`, `lower than`, `higher than`, `larger than`, `smaller than`, `in the range`, `between` |
| 3 | 가장 큰 | `greatest`, `largest`, `highest`, `most`, `biggest`, `maximum`, `max` |
| 4 | 가장 작은 | `lowest`, `least`, `smallest`, `minimum`, `min` |
| — | 비교 없음 | 위 어느 것에도 안 맞음 |

"비교 있음" = 1~4 중 하나. 이 목록은 1위 적중 결과를 본 뒤 만들었다. 알려진 오류:
- 산술 1건 오분류: "what is the percent change in other purchase commitments between 2013-14 and 2015-16?" 는 값 비교가 아닌데 `between` 으로 '기준값 비교'에 들어간다.
- 조회 누락: "… ranks first?" 형태(예: "The Total fixed maturities, held-to-maturity-5 amount of which section ranks first?")는 '비교 없음'으로 들어간다.
- 조회 누락(표본 30개 확인 중 발견): "How many elements show negative value in2014 forVIEs ?" 는 '비교 없음'으로 들어간다.
목록은 고치지 않는다.

**(c) 구별 후보 수 m.**
- 단어 = `rag_agent.retrieve.encoders._tokenize`(소문자) − `sklearn.feature_extraction.text.ENGLISH_STOP_WORDS`, 서로 다른 단어.
  기존 path overlap 코드 `results/components_20260925/hitab_path_overlap.py`, `results/lookup_vs_arith_20260928/analyze.py` 와 같다.
- 칸의 경로 단어 = 행 경로 + 열 경로 원소 중 식별자(`^Table \d+$`, `^row \d+$`, `^column \d+$`)를 뺀 원소의 단어.
  식별자 제외는 지시서에 없던 세부이며, `analyze.py` 의 MultiHiertt 경로 정의와 같게 이 문서에서 정한다.
- Q = 질문 단어 ∩ 정답 칸 경로 단어.
- m = 정답 칸과 같은 표의 색인 칸 중 경로 단어가 Q 를 모두 포함하는 칸 수(정답 칸 포함). Q 가 비면 m = 그 표의 색인 칸 수.
- 색인 칸 = 그 버전 머리글 규칙으로 `build_corpus(..., "cell", ...)` 가 만든 칸 = 값이 비지 않은 데이터 칸. 주 분석은 최종 규칙(v3.3u), T7 은 처음 규칙(v1).

**(d) 무작위 순서 기준값** = 문항별 1/m 의 평균.

**(e) 통계.**
- 서로 다른 문항끼리의 비율 비교: Fisher 정확 검정 양측(`scipy.stats.fisher_exact`) + 차이의 Newcombe 95% CI(hybrid score, Wilson 구간 두 개로 만드는 방법, z=1.959964).
  차이 방향은 항목마다 표에 적는다.
- 표준화: 산술 구성 가중치 w_비교 = 4/71, w_없음 = 67/71 을 조회의 층별 1위 적중률에 적용한다.
  조회 표준화 = w_비교·p(조회 비교) + w_없음·p(조회 비교 없음). 표준화 잔차 = p(산술 전체) − 조회 표준화.
  부트스트랩: B=10,000, `numpy.random.default_rng(42)`, 반복마다 네 층(조회 비교, 조회 비교 없음, 산술 비교, 산술 비교 없음)의
  적중 수를 이 순서로 `rng.binomial(n, 관측 비율, size=B)` 로 뽑는다. 산술 전체 = 두 산술 층 합 / 71.
  95% CI = 백분위 2.5·97.5. p = 2 × min(잔차 ≤ 0 비율, 잔차 ≥ 0 비율), 1 을 넘으면 1. 조회 표준화 값의 CI 도 같은 표본으로 낸다.
  T7 의 가중치는 처음 버전 산술 71문항의 분류 구성으로 다시 센다(문항·질문이 같으므로 4/71, 67/71 이 되어야 한다).
- 반대 방향 표준화(조회 구성을 산술에 적용)는 하지 않는다(산술 비교 층 4문항).
- T4 검정: 관측 1위 적중 수를 문항별 성공 확률 1/m 의 푸아송 이항 분포와 비교하는 정확 양측 검정
  (관측 수의 확률 이하인 값들의 확률 합, 상대 허용 1e-7), 관측 비율의 Wilson 95% CI.

## 계산 (GATE 2) — `results/lookup_vs_arith_20260928/posthoc/`

- **T1** 조회: 머리글 답(예/아니오) × 비교 단어(있음/없음). 칸마다 문항 수, 1위 적중 수.
- **T2** (i) 전체 1위: 산술 − 조회 (45/71 대 76/212). (ii) 조회 안: 비교 있음 − 비교 없음. (iii) 비교 없음끼리: 산술 − 조회.
  (iv) 조회 표준화 값, 표준화 잔차, 부트스트랩 CI·p. (i)~(iii)은 비율·Fisher p·Newcombe CI.
- **T3** 유형(조회/산술) × 비교 단어(있음/없음): 문항 수, m=1 문항 수·비율, m 중앙값. 유형별 m=1 층과 m>1 층의 1위 적중 수·비율.
- **T4** 조회 비교 있음 141문항: 관측 1위 적중 vs 1/m 평균, 푸아송 이항 p, Wilson CI. m>1 문항만 한 번 더.
- **T5** 1위 실패 문항의 1위 칸 위치: (가) m 후보 집합 안 / (나) 같은 표의 집합 밖 / (다) 다른 표. 유형 × 비교 단어별 수와 비율. 최종 버전만.
- **T6** 수작업 판정용 CSV 283행(최종 버전, query_id 정렬, UTF-8 BOM): qid, 유형, 질문, 정답 칸 행 경로, 정답 칸 열 경로
  (경로는 표 객체 원소를 " > " 로 이음, 식별자 포함), 비교 단어 분류, 빈 열 "질문만으로 정답 칸이 하나로 정해지는가: 예/아니오/판단불가".
  1위 적중·순위·점수는 넣지 않는다.
- **T7** 처음 버전으로 T1~T4.
- **T8** `results/lookup_vs_arith_20260928/analyze.json`(aac902b)의 multihiertt 조회1·산술1 path overlap
  (`groups.<그룹>.path_overlap`, `path_overlap_success`, `path_overlap_fail` 의 평균·중앙값, 성공 = 예산 20 전부 포함)과
  같은 문항 집합의 m 중앙값·평균·m=1 비율을 한 표에 둔다. 문항 수가 두 쪽에서 같은지 assert 한다.

## 멈춤 규칙

결과를 본 뒤 정의·구간·단어 목록을 바꾸지 않는다. 바꿔야 할 문제가 보이면 계산을 멈추고 보고한다.
보고는 표와 "파일 경로:키"만. 해석·결론 문장을 쓰지 않는다.
