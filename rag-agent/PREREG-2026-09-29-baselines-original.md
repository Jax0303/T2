# 사전등록: 비교군 4개를 원논문 방법(공식 코드)으로 재구성, 표준 Recall@k·Precision@k·MRR, 공식 EM+F1 (2026-09-29)

비교군 결과를 보기 전에 커밋한다. 수치는 `results/mh_only_20260928/baselines/<name>/` 와 `final/` 의 결과 파일에서만 읽는다.
사용자 결정(2026-09-29 새벽): 비교군 목록·지표·k·채점은 아래로 확정, 다시 묻지 않는다.

## 왜 바꾸나 (공개)

- 원고의 `tablerag_leaf/path`·`trag_hetero` 는 원논문 방법이 아니라 **색인 글 형식만** 옮긴 것이었다(Chen: 질의 확장·스키마 검색·코드 실행 없음, Yu: 질의 분해·SQL·반복 없음 → Yu 논문 자신의 비교군 NaiveRAG 에 해당). 사용자가 원논문 PDF 3편(Chen 2024, Yu 2025, RAPTOR 2024)을 대조해 확인.
- `rowcol`·`randrow` 는 TAP4LLM(Sui 2023) 유래로 RAG 방법이 아님 → 제외. Mixture-of-RAG(KDD 2026)는 문서 더미에서 문서를 찾는 문제라 closed-domain 설정에 맞지 않음 → 관련 연구 인용만.
- 지표: 20셀 문맥의 "정답 셀 전부 포함" 비율은 주지표에서 뺀다(지도교수 지시). 표준 Recall@k 로 되돌린다.

## 비교군 (원방법, 공식 코드, 같은 조건)

| 이름 | 원논문 | 공식 코드 | 구현 범위 |
|---|---|---|---|
| `langchain_chunk` | LangChain 공식 RAG 튜토리얼 | docs.langchain.com/oss/python/langchain/rag | splitter(1000/200) → 임베딩 → 유사도 상위 4 청크 → 튜토리얼 프롬프트 |
| `raptor` | Sarthi et al., ICLR 2024 | parthsarthi03/raptor | 100토큰 청크 → UMAP+GMM 군집 → LLM 요약 트리 → collapsed tree 2,000토큰(원설정) 및 575토큰(본 방법 문맥 길이 맞춤) |
| `tablerag_chen` | Chen et al., NeurIPS 2024 | google-research table_rag @08a8d67 | 질의 확장 → 스키마·셀 검색(top-5, B=10,000) → PyReAct pandas 실행 |
| `tablerag_yu` | Yu et al., EMNLP 2025 | yxh-y/TableRAG | 질의 분해 → 청크 검색(회수 30 → 재정렬 3) → SQL(SQLite) 실행 → 반복 ≤5 |
| 참고 | 표 전체 입력(검색 없음) | — | 기존 결과 378/882 |

같은 조건: 검색 범위 = 질문의 문서, **표만**(본문 단락 제외, 기존 청크 arm 과 같은 렌더러·머리글 최종 규칙); 모든 LLM 호출 = Qwen3-8B 4bit 비생각·greedy(seed 42); 임베딩 = bge-base-en-v1.5; Yu 의 재정렬기 = bge-reranker-v2-m3(논문이 이름 붙인 BGE 계열, 로컬 캐시).
원설정과 다른 것(모델 교체, 계층 머리글 → 경로 결합 열 이름 `official_frame(mode="path")`, 문서당 표 여러 개 처리, MySQL→SQLite, 답 추출용 마커)은 비교군마다 `README.md` 에 목록으로 남긴다. 각 비교군은 먼저 882 중 3문항 점검 실행 뒤 전체 실행(GPU 1개, 순차).

## 지표 (`results/mh_only_20260928/final/rpm.py`)

- **Recall@k, Precision@k, MRR — k = 검색 단위 개수**(셀·청크·노드·문서 등 방법 고유 단위), k ∈ {1, 5, 10, 20, 30}. 문항 평균. 단위가 배달하는 셀에 정답 셀이 들어 있으면 그 정답 셀을 찾은 것으로 센다(기존 GATE 1 정의와 같음).
  - 단위 크기가 방법마다 다르다(셀 1개 대 청크 ≈ 37셀)는 사실을 표 옆에 단위당 평균 셀 수·토큰 수로 적는다. **해석은 하지 않고 사실만 적는다.** "전달 셀 수 k" 같은 다른 정의는 쓰지 않는다.
  - 본 방법 k≤20 은 기존 측정값(`metrics.json`)과 일치해야 한다(`rpm.py` 로 재계산해 대조). k=30 은 순위 100 재검색(`retrieval_r100/`, 이 등록 전 시작)에서 읽는다.
- **답변: MultiHiertt 공식 EM + 공식 F1**(`rag_agent/eval/multihiertt_f1.py`, 45bd9cc 의 `get_span_selection_metrics`·`evaluate_span_program_result` 이식, 검사 `tests/test_multihiertt_f1.py` 9개 통과). 882 문항. 논문마다 정규화가 달라(Chen EM, Yu LLM 심판, RAPTOR 데이터셋별) 데이터셋 공식 채점으로 통일.
  - 본 방법 기존 답변(`cell_uniq.jsonl`)의 EM 재계산 = 저장값 408, 불일치 0. F1 평균 .4853.

## 판정 (결과 전 고정)

- 답변: 본 방법 대 비교군 4개, 같은 882 문항 정확 McNemar, Holm(가족 = 4개), 차이의 95% 구간(층화 부트스트랩, `mh_cap300_report.py` 방식). 검색: 같은 2,885 문항, Recall@20 의 문항별 값 차이에 부호 검정.
- 사용자 목표는 "답변 ≥ 10%p 우세". **이 목표는 판정 기준이 아니다** — 리더 상한(정답 셀이 문맥에 다 있을 때 49.9% → 검색이 완벽해도 882 기준 최대 약 440, +3.6%p)으로 검색 개선만으로는 도달할 수 없음을 결과 전에 적어 둔다. 결과가 어떻게 나오든 그대로 싣는다.

## 예측 (결과 전)

1. 표준 Recall@k(단위 개수 기준): 청크·노드 단위 비교군(`langchain_chunk`, `tablerag_yu` 청크 단계, `raptor`)이 모든 k 에서 본 방법보다 **높다**(단위 하나가 셀 수십 개). `tablerag_chen` 은 모든 k 에서 본 방법보다 낮다(숫자 열이 min/max 요약 문서 하나로 접힘).
2. 답변 EM: 본 방법 > `langchain_chunk`(기존 고정 청크 368·Yu 청크 361 과 같은 수준 예상), 본 방법 > `raptor` 2,000토큰(문서 대부분을 요약과 함께 넣음 → 표 전체 입력 378 근처 예상). `raptor` 575토큰: 방향 예측 없음.
3. 본 방법 > `tablerag_chen`(Qwen3-8B 4bit 의 pandas 코드 오류율이 높을 것으로 예상, 근거 없음 — 확인 대상). `tablerag_yu`: **방향 예측 없음**(SQL 실행이 산술 371문항에서 유리할 수 있음).
4. 어느 비교군에서든 답변 차이 10%p 이상은 나오지 않는다(리더 상한).

## 실행 순서

본 방법 순위 100 재검색(진행 중) → 비교군 4개 3문항 점검(서브에이전트, 병렬 구현·순차 GPU) → 사용자에게 점검 결과 보고 → 전체 실행 순서: `langchain_chunk` → `raptor` → `tablerag_chen` → `tablerag_yu`(각 2,885 검색 + 882 답변, GPU 잠금 `.gpu.lock`) → `final/` 표.
