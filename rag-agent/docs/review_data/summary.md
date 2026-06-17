# 분석용 데이터 패키지 — 표 RAG 구조보존 실증연구

라이너 분석용. 함께 올리는 `results_long.csv`(질문별 정오답)와 `hpir_gate.txt`(분해 정확도)의
설계·집계를 요약한다. 모든 수치는 실측.

## 데이터 파일
- **`results_long.csv`** — long format, 375행. 컬럼:
  `experiment, model, split, qid, condition, correct(0/1), got_gold, pred, gold`
  - `experiment` ∈ {interaction, token_control, retrieval_pipeline}
  - `model` ∈ {llama-8b, gpt-oss-120b}
  - `split` ∈ {flat(WikiSQL), hier(HiTab)}
  - `condition` ∈ {flat_values(S0), flat_leaf(S1), header_path(S2), header_shuffle(토큰통제)}
  - **paired 설계**: 같은 `qid`가 모든 condition에 등장(동일 질문·모델·채점, 직렬화만 변경)
- **`hpir_gate.txt`** — HPIR 헤더경로 분해 정확도(HiTab dev, 무LLM).

## 실험 설계 요약
- 독립변수: 표 직렬화의 구조보존도 3단계 S0(값만)⊂S1(잎헤더=평탄화)⊂S2(전체 헤더경로).
- 층화/통제: flat(계층無, 반증통제) vs hier(계층有); header_shuffle(S2와 토큰동일·경로오배치=토큰통제).
- 지표: 수치/정확일치(correct). paired bootstrap 95% CI, seed=42.
- retrieval_pipeline: 표→행청크→BM25 top-k→직렬화→LLM계산, 검색은 조건 간 동일.

## 집계 결과(CSV에서 재계산 가능)
- interaction S2−S1 (McNemar): 8b hier +0.30 (p=0.070, ns), 120b hier +0.20 (p=0.125, ns); flat은 p=1.0 (null). 교호작용은 경향이나 n=20 검정력 부족.
- token_control(8b hier): header_path−header_shuffle +0.40, McNemar p=0.008 (확실).
- retrieval_pipeline(8b hier): header_path−flat_leaf +0.40, McNemar p=0.002 (확실); chunk_recall .60.
- HPIR 분해: fuzzy BOTH 0.61~0.635(예산 늘려도 천장), embedding BOTH 0.67.

## 정직한 한계(이미 인지)
1. 직렬화 기법 자체 비신규(OHD/STR/TABGR/MixRAG/Song).
2. HPIR 분해 0.67 천장 → 피연산자-완전 검색의 단일실패점.
3. FinQA/T²-RAGBench 미실행(무료 LLM 토큰한도): 8b는 다단계 재무계산 바닥, 대형모델 일일한도 정체.
4. n=20~25, 모델 2종.
