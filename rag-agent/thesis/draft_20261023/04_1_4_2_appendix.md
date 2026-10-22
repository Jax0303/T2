# 4.1·4.2·부록 교체 초안 (2026-10-23; 10월 주 설정으로 바꾸는 부분만. 비교군 재실행 결과 【대기】)

## 4.1.1 하드웨어 및 소프트웨어 환경 — 표 4-1·4-2 교체
문장: "모든 실험은 GPU 한 장을 장착한 컴퓨터 한 대의 로컬 환경에서 수행하였으며, 외부 API 는 4.4.3 절의 참고 비교(GPT-4o)에서만 사용하였다."
| 항목 | 내용 |
|---|---|
| CPU | Intel Core i9-13900K |
| GPU | NVIDIA GeForce RTX 4070 Ti (12GB) |
| 메모리 | 15GB (WSL2 할당) |
| 운영체제 | Ubuntu 24.04.5 LTS (WSL2) |
| 언어 | Python 3.12.3 |
| 라이브러리 | PyTorch 2.14.1 (CUDA 13.0), transformers 5.19.0, sentence-transformers 6.1.0, faiss 1.15.1 (비교군·한 건씩 생성); vLLM 0.31.0 (PyTorch 2.13.0, transformers 5.17.0; 본 방법·전달 형식·표 전체 입력) |
표 4-2 실험에 사용한 모델:
| 용도 | 모델 | 설정 |
|---|---|---|
| 임베딩 | BAAI/bge-base-en-v1.5[4] | 768차원, 학습하지 않음 |
| LLM(본 방법·전달 형식·표 전체 입력) | Qwen3.5-4B | bf16(양자화 없음), 비생각 모드, 그리디 디코딩, vLLM 연속 묶음, 출력 상한 4,096 토큰 |
| LLM(비교군 파이프라인) | Qwen3.5-4B | bf16, 비생각 모드, 그리디 디코딩, transformers 한 건씩 생성(각 방법의 생성 상한) |
| 재정렬 | BAAI/bge-reranker-v2-m3 | TableRAG(Yu) 비교군에서만 사용 |
| 참고 비교 외부 모델 | GPT-4o (gpt-4o-2024-08-06) | OpenAI API, temperature 0 (4.4.3 절) |
비고 문장: "두 생성 엔진의 출력은 같은 가중치·같은 greedy 디코딩이지만 연속 묶음 실행은 부동소수 합산 순서 때문에 실행마다 달라질 수 있어, 본 방법은 3회 실행 다수결로 보고한다(4.2.3). 한 건씩 생성한 50문항과 묶음 생성의 정확도 차이는 없었다(39 대 39)."
(옛 표의 RTX 3060 Ti 8GB·Qwen3-8B NF4 4비트는 10-01 발표판 설정 — 삭제.)

## 4.1.2 데이터셋 — 표 4-3 답변 평가 열 교체
"답변 정확도는 검색 성능 비교와 같은 문항 집합으로 일치시키기 위해 … 200문항으로 평가한다" → "답변 정확도는 추출 표본 882문항 전체로 평가한다." 표 4-3 의 '답변 평가' 열: 211 / 300 / 71 / 300 / 882. (200문항 문장·열 삭제.)

## 4.2.1 비교군 — 표 4-4
문장 "모든 LLM 호출에는 Qwen3-8B 4비트 모델을" → "모든 LLM 호출에는 Qwen3.5-4B bf16 모델을". 표 4-4 '답변 평가' 열 200 → 882. TableRAG(Chen) 문단의 128/512 토큰 설명은 유지하되 숫자는 새 실행으로 교체【대기】.
본 방법 행 추가(표 4-4 또는 본문): "제안 방법: 상위 20개 셀 문장을 검색하고, 그 셀이 속한 표의 행·열을 되살린 부분 표(채운 부분 표)로 LLM 에 입력. 지시문 P*(부록 B)."

## 4.2.2 평가 지표 — 답변 채점 문단 교체
"답변 정확도는 LLM 의 최종 답이 정답을 포함하는 문항의 비율이다(형식 허용 채점). 본 방법·표 전체 입력은 지시문이 요구한 마지막 줄('Final answer:') 의 값을, 비교군은 각 파이프라인이 낸 최종 답을 채점 대상으로 한다. 조회 문항은 정규화한 문자열 포함 여부로, 산술 문항은 수치 비교로 판정하되 10의 거듭제곱 배 척도 차이·부호·반올림 차이를 허용한다(규칙은 공개 코드 score_v2.included_v2). 예/아니오 문항은 최종 답의 yes/no 로 판정한다. MultiHiertt 공식 채점(EM)도 보조 지표로 함께 보고한다." (옛 문단의 DocMath-Eval 0.15% 허용 설명은 10-14 채점 규칙 v2 와 맞는지 확인 뒤 정리【확인】.)

## 4.2.3 실험 구성 — 교체
"실험은 세 가지로 구성한다. 첫째, 검색 성능 비교(4.3.1, 2,885문항; RAPTOR·TableRAG(Chen)은 LLM 을 쓰므로 882문항). 둘째, 답변 정확도 비교(4.3.2, 882문항): 본 방법과 표 전체 입력은 vLLM 으로 3회 실행해 문항마다 2회 이상 맞힘(다수결)으로 판정하고, 비교군 파이프라인은 한 건씩 결정적 생성으로 1회 실행한다. 셋째, 전달 형식·답변 상한·리더 LLM·오답 원인 분석(4.4)." 【비교군 실행 횟수는 결과 뒤 확정】

## 4.3.2 답변 정확도 — 표 4-7 교체 【대기】 + 그림 4-1 다시 그리기(make_figs.py)
## 4.3.3 비교 분석 — 오답 분류 문단(200문항 85건) → 4.4.4 로 대체; "재현율 높은 방법이 답변도 높은 것은 아니다" 문단은 새 수치로【대기】.

## 5장 결론 — 수치 문장 교체【대기】: "200문항에서 57.5% …" → 882 결과. 한계 셋째("200문항에서만") 삭제. 향후 연구의 "검색된 셀만 마크다운 부분 표로 전달하는 조건" 은 이번에 수행(4.4.1)했으므로 삭제.

## 부록 B LLM 지시문 — P* 전문으로 교체
You are a financial analyst answering questions about tables from a company's annual report. Use the following pieces of retrieved context to answer the question. The context contains cells from the report's tables; the meaning of each value is given by its row and column headers. The context contains what is needed to answer; do not say that you don't know, always give your best answer. Before answering, write out the calculation as an explicit arithmetic expression … Rules: (1)…(6)… On the last line write 'Final answer: ' followed by only the final value, with no units or explanation.
【scripts/prompt_variants.py R1X1F1B2D1C1E0 의 전문을 그대로 옮긴다 — 규칙 (1)~(6) 포함】
사용자 입력 형식: "Question: {질문} \nContext: {입력} \nAnswer:". 지시문 선택 절차(검증 395문항, 147 변형) 한 문단.

## 부록 D 정답 셀만 입력한 결과 — 882 로 교체
표 D-1: 문항 그룹 | 문항 수 | 정답 셀만(O1) | 정답 표 전체(O2) | 정답 표 안 D(O3) | 본 방법 D
조회·셀1 211 | 168 | 161 | 163 | 153 ; 조회·셀2+ 300 | 254 | 249 | 248 | 245 ; 산술·셀1 71 | 33 | 50 | 49 | 49 ; 산술·셀2+ 300 | 209 | 185 | 191 | 182 ; 전체 882 | 664(75.3) | 645(73.1) | 651(73.8) | 629(71.3)
설명: 산술·셀1 에서 정답 셀만 주면 오히려 낮다(33 대 49) — 정답 셀 표시가 계산에 필요한 값을 다 담지 않는 문항(주가 수익률 기준 100, 평균의 개수 등). 오답 218 의 분류표는 4.4.4 로.

## 부록 F(신설) 리더 LLM 비교 상세 — 4.4.3 표의 층별 수치, GPT-4o 호출 기록(served model, system_fingerprint 종류 수, usage, 비용 추정 $7.5), 재현 점검(HF 50/50, vLLM 26/50, API 270/882).
