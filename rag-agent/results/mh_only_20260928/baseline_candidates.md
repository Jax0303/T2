# GATE 3. 비교군 후보표와 현재 비교군 판정 (2026-09-28)

기준(지도교수 지시): 2026년 기준 (i) 계층 표를 포함한 문서에서 셀·근거 검색 후 답변하는 같은 문제, (ii) 공개 코드, (iii) MultiHiertt 또는 계층 표 벤치마크 평가.
출처: 오늘 오후 별도 조사(`results/fair_mh_20260928/baselines.md`, 1차 출처 URL 열람 목록 포함). 확인 안 된 칸은 빈칸. 판정 기호: ○ 충족, × 불충족, — 미확인.

## 후보표

| 저자(1저자) | 연도 | 제목 | arXiv | 공개 코드 | 평가 벤치마크 | 검색 단위 | (i) | (ii) | (iii) | 8GB·무학습·무API 재현 |
|---|---|---|---|---|---|---|---|---|---|---|
| Chen, S.-A. | 2024 | TableRAG: Million-Token Table Understanding with Language Models (NeurIPS 2024) | 2410.04739 | github.com/google-research/google-research/tree/master/table_rag | ArcadeQA, BirdQA, TabFact (단일 대형 평면 표) | 셀(열이름·값 쌍) + 스키마 | × (단일 평면 표, 문서 안 여러 계층표 아님) | ○ | × | 셀 표현 이식만 가능(질의 확장·프로그램 실행 없음) |
| Yu, X. | 2025 | TableRAG: A Retrieval Augmented Generation Framework for Heterogeneous Document Reasoning (EMNLP 2025) | 2506.10380 | github.com/yxh-y/TableRAG | HybridQA, WikiTQ, HeteQA(304) | 1,000자 청크(텍스트) + SQL(표) | △ (이종 문서, 표는 SQL 단위) | ○ | × | 청킹 표현만 이식(SQL·BGE-M3 재정렬 없음) |
| Zhang, C. | 2026 | Mixture-of-RAG (KDD 2026) | 2504.09554 | github.com/ChiZhang-bit/Mixture-of-RAG | DocRAGLib(자체, MultiHiertt 표본 포함) HiT@1 54.10 / HiT@5 76.03 | 문서 (H-RCL 행·열 수준 요약) | △ (검색 단위가 문서) | ○ | △ (MultiHiertt 표본 포함 자체 벤치마크) | × (text-embedding-3-large, GPT-4o 재정렬) |
| Alonso, I. & Lapata, M. | 2026 | A Table Is Worth 64 Tokens | 2608.26949 | — | MultiHiertt dev 1,044 EM(HTML 입력 Qwen3.5-9B 39.6, Gemma 4 26B-A4B 44.2) | 표 (BGE-M3, k=5) | △ (표 단위 검색 + VLM) | — | ○ | 무학습이나 이미지 입력(표현 통제 실험과 무관) |
| — (MoCA-Agent) | 2026 | MoCA-Agent | 2606.11537 | — | MultiHiertt dev EM 71.17 (Qwen3.6-27B) | 검색 없음(에이전트) | × | — | ○ | × (27B) |
| — (Fortune / Formula-R1) | 2025–26 | Fortune | 2505.23667 | github.com/microsoft/Fortune | MultiHiertt 40.85 / Fortune++ 51.73 (분할 미확인) | 검색 없음(수식 생성 RL) | × | ○ | ○ | × (RL 학습) |
| — (GRAB) | 2026 | Latent Bridges / GRAB | 2606.28916 | — | MultiHiertt 자체 분할 test 1,044 EM 21.55 | 그래프 인코더(학습) | △ | — | ○ | × (학습) |
| — (FT-RAG) | 2026 | FT-RAG | 2605.01495 | — | 자체 Multi-Table-RAG-Lib 9,870 QA, 표·셀 Hit Rate | 항목 단위 의미 그래프 | ○ (문제 정의 가장 근접) | — | × | — (코드·모델 미확인) |
| Si, — (TabRAG) | 2025 | TabRAG (NeurIPS 2025 AI4Tab) | 2511.06582 | — | TAT-DQA, MP-DocVQA, WikiTQ | 페이지 → VLM JSON | × (PDF 페이지) | — | × | × (Qwen3-VL-8B + Qwen3-Embedding-8B) |
| Zhao, Y. (DocMath-Eval) | 2024 | DocMath-Eval (ACL 2024) | 2311.09805 | github.com/yale-nlp/DocMath-Eval | MultiHiertt 재주석(simplong dev 100/test 400) | 벤치마크(방법 아님) | — | ○ | ○ | 벤치마크 |
| Srivastava, — | 2024–25 | 프롬프트 기법 비교 | 2402.11194 | — | TATQA·FinQA·ConvFinQA·MultiHiertt | 검색 없음 | × | — | ○ | 검색 없음 |
| Zhang, T. (TAT-LLM) | 2024 | TAT-LLM (ICAIF 24) | 2401.13223 | — | FinQA·TAT-QA·TAT-DQA | 검색 없음(미세조정) | × | — | × | × (학습) |
| — (GraphOTTER) | 2024 | GraphOTTER (COLING 2025) | 2412.01230 | — | HiTab EM 73.74 (Qwen2-72B) | 단일 표 그래프 추론 | × (단일 표) | — | ○ (HiTab) | × (72B) |
| 공식 리더보드 | — | MultiHiertt CodaLab | — | codalab.lisn.upsaclay.fr/competitions/6738 | 비공개 test: 1위 51.788, 2위 46.169, 3위 45.594 (지표·날짜 미확인) | — | — | — | ○ | — |

기준 (i)(ii)(iii)를 모두 ○로 충족하는 **공개 코드가 있고 로컬에서 돌릴 수 있는** 후보는 조사 범위에서 없었다. 문제 정의가 가장 가까운 FT-RAG(2605.01495)는 코드·모델을 확인하지 못했다.

## 현재 비교군 판정

| 현재 비교군 | (i) 같은 문제 | (ii) 공개 코드 | (iii) MultiHiertt/계층표 평가 | 판정 |
|---|---|---|---|---|
| 1,000자 청크 | △ 일반 RAG 관행(특정 논문 없음; Yu 2025 NaiveRAG 설정과 같음) | 해당 없음 | × | 관행 하한선으로 유지 가능. "방법" 아님을 표기 |
| TableRAG(Yu) 청킹 이식 | △ (이종 문서; 표는 SQL) | ○ | × | 유지 시 "청킹 표현 이식 변형"으로만. SQL·재정렬 미재현 |
| TableRAG(Chen) 셀 검색 이식 | × (단일 평면 표) | ○ | × | 원 방법의 LM 질의 확장·프로그램 실행 **미재현** → "셀 검색 구성요소만 비교"로 명시하거나 제외 후보 |
| 행·열 단위(RowCol) | × (Chen 2024 기본선) | ○ | × | Chen 이식과 같은 처리 |
| 표 단위 | △ (Alonso & Lapata 2026 이 MultiHiertt dev 에서 표 단위 검색 사용) | — | ○ | 단위 크기 사다리로 유지 가능(문헌 선례 있음) |
| 표 전체 입력(검색 없음) | 참고 조건 | — | — | 참고 조건으로 유지 |
| 무작위 행 | — | — | — | **삭제**(지시) |
| MT2Net | — | — | ○ | **제외**(지시), 선행 연구 인용만 |

새 비교군 추가는 사용자가 고른 뒤 구현한다.
