# MultiHiertt 비교군 문헌·코드 조사 (2026-09-28, 1차 출처 대조)

별도 조사 에이전트가 2026-09-28 오후에 작성. 저장소 코드는 미열람이라 "적응 변형" 여부는 이 세션이 확인해 적었다: 세 이식 비교군(TableRAG Chen 셀·스키마, RowCol·RandRow, TableRAG Yu 청킹)은 모두 우리 검색기(bge-base + BM25 혼합, 예산 20셀)에 **텍스트 표현만 이식한 변형**이며 원 논문 시스템 재현이 아니다(`rag_agent/serialization/tablerag_unit.py`, `scripts/mh_arms.py`).

판정 기준: "한 재무 문서 안 표들에서 셀 단위 근거 검색, 고정 임베딩 검색기 아래 **셀 텍스트 표현만 바꿔** 비교, 로컬 모델·무학습".

## 1. 기존 비교군

| 비교군 | 논문/코드·공개일 | 원래 다룬 문제·검색 단위 | MultiHiertt 평가 | 필요 학습·정답 정보 | 모델·연산 | 우리 저장소 재현 성격 | 이번 주 적합성 |
|---|---|---|---|---|---|---|---|
| 고정 1,000자 청크 | 특정 논문 없음. Yu 2025의 NaiveRAG가 "1000토큰 청크, 200 겹침" | 문서 텍스트 조각 검색 | 해당 없음 | 없음 | 임베딩만 | 일반 RAG 기본선 | **적합**(표준 하한선) |
| TableRAG (Chen 2024) | arXiv 2410.04739 v1 2024-10-07, NeurIPS 2024. 코드 google-research/table_rag | **단일 대형 평면 표**(ArcadeQA·BirdQA·TabFact). 스키마 검색 + 셀 검색, 셀 단위 = 서로 다른 (열이름, 값) 쌍, 인코딩 예산 B=10,000, K=5. 계층 헤더 처리 언급 없음 | 없음 | 무학습 | GPT-3.5-turbo·Gemini-1.0-Pro·Mistral-Nemo, text-embedding-3-large; vLLM 로컬 지원 | 셀/스키마 표현(leaf/path)만 우리 검색기에 이식 → **파이프라인 적응 변형** | **적합**(셀 표현 비교의 직접 대조군). 원 논문은 문서 안 여러 계층표 문제가 아님을 명시 |
| RowCol / RandRow | Chen 2024 논문의 기본선 | RowCol: 행·열 임베딩 상위 K개씩 → K×K 부분표, K=30. RandRow: 행 K개 균등 표집 | 없음 | 무학습 | 위와 동일 | 행/열 단위 이식 → 적응 변형 | **적합**(규칙 기반 하한선) |
| TableRAG (Yu 2025) | arXiv 2506.10380 v1 2025-06-12, v2 2025-09-30, EMNLP 2025 main. 코드 yxh-y/TableRAG | 이종 문서 QA: 질문 분해 → 텍스트 검색(1000토큰 청크, 상위30→재정렬 상위3, BGE-M3) → **SQL 실행**(MySQL). 표는 DB 테이블 단위 | 없음(HybridQA·WikiTQ·자체 HeteQA 304문항) | 무학습, MySQL, LLM API 키 | Claude-3.5-Sonnet, DeepSeek-V3/R1, Qwen2.5-72B | 청킹 부분만 이식 → **적응 변형**(SQL 단계 없음) | 적합하되 "청킹 표현만 가져온 변형"으로 표기 필수 |
| 표 단위 / 행 단위 색인 | 특정 논문 없음. 표 단위는 Alonso & Lapata 2026(arXiv 2608.26949)이 MultiHiertt dev에서 "표를 원자 검색 단위, BGE-M3, k=5"로 사용 | 표/행 통째 검색 | 2608.26949: dev 1,044문항, EM | 없음 | 임베딩만 | 자체 구현 | **적합**(단위 크기 사다리) |

## 2. 2024-09 ~ 2026-09 후보

| 방법 | URL·날짜 | 하는 일 | MultiHiertt 결과(명시된 대로) | 8GB GPU·무학습·무API 재현 |
|---|---|---|---|---|
| Mixture-of-RAG (MixRAG, Zhang) | arXiv 2504.09554 v1 2025-04-13, v3 2025-12-16, KDD 2026. 코드 ChiZhang-bit/Mixture-of-RAG | 문서 단위 검색. H-RCL: 계층 행·열 수준 요약 표현, BM25+임베딩 앙상블 + GPT-4o 재정렬, RECAP 추론 | 자체 DocRAGLib(2,178문서·4,468 QA, "MultiHiertt 표본 포함 + HiTab 링크 크롤링"). HiT@1 54.10%, HiT@5 76.03% (문서 수준). MultiHiertt 원 분할 결과 없음 | **아니오**: text-embedding-3-large + GPT-4o 필수, 검색 단위가 문서(셀 아님) |
| Alonso & Lapata, "A Table Is Worth 64 Tokens" | arXiv 2608.26949, 2026-08-27 | 표를 이미지로 압축해 VLM 입력, BGE-M3 표 단위 검색 k=5 | dev 1,044, EM(테스트 라벨 비공개라 dev 사용 명시). HTML 직접 입력: Qwen3.5-9B 39.6%, Gemma 4 26B-A4B 44.2%. 나머지 칸은 추출값 불일치로 미확인 | 무학습이지만 VLM·이미지 입력 → 표현 통제 실험과 무관. 표 단위 검색 근거로만 인용 |
| MoCA-Agent | arXiv 2606.11537 v2 2026-06-17 | 4개 전문가 에이전트 + 코드 샌드박스 | dev EM 71.17% (Qwen3.6-27B, vLLM) | **아니오**: 27B, 검색 없음(문서 전체 입력 여부 미확인) |
| Fortune / Formula-R1 | arXiv 2505.23667 v1 2025-05-29, v3 2026-03-23. 코드 microsoft/Fortune | 수식 생성 RL | 검색 결과 요약: Fortune 40.85%, Fortune++ 51.73% (분할·초록 내 명시 미확인) | **아니오**: RL 학습 |
| GRAB, Latent Bridges | arXiv 2606.28916, 2026-06-27 | 그래프 인코더+잠재 브리지(91M 학습), Qwen3-4B-Base 고정 | 자체 분할(train 7,047/val 783/test 1,044) EM: GRAB 21.55, 제로샷 7.18 | **아니오**: 학습 필요 |
| DocMath-Eval simplong | arXiv 2311.09805, ACL 2024. yale-nlp/DocMath-Eval | MultiHiertt 재주석 벤치마크(dev 100/test 400), 정확도 | 최고 Claude-3.5-Sonnet PoT 65.0% | 방법이 아니라 벤치마크. 문헌만 |
| Srivastava 외 | arXiv 2402.11194 v1 2024-02-17, v3 2025-10-09 | 프롬프트 기법 비교(TATQA·FinQA·ConvFinQA·MultiHiertt) | 수치 미확인 | 검색 없음. 문헌만 |
| FT-RAG | arXiv 2605.01495, 2026-05-02 | 표를 항목 수준 의미 단위 그래프로 분해, 이웃 확장; 자체 Multi-Table-RAG-Lib 9,870 QA, 표·셀 수준 Hit Rate | 없음 | 코드·모델 미확인. **문제 정의가 가장 가까움** → 문헌 대조 필수 |
| TabRAG (Si) | arXiv 2511.06582 v1 2025-11-10, NeurIPS 2025 AI4Tab | PDF 페이지 레이아웃 분할 → VLM(Qwen3-VL-8B) JSON 표현, Qwen3-8B·Qwen3-Embedding-8B | 없음(TAT-DQA·MP-DocVQA·WikiTQ 등) | 로컬 가능하나 페이지 입력·8B 임베딩 → 8GB 무리, 문제 다름 |
| TAT-LLM | arXiv 2401.13223 v1 2024-01-24, ICAIF 24 | LLaMA-2 미세조정, FinQA·TAT-QA·TAT-DQA | 없음 | 아니오(학습) |
| Table-R1 (Inference-Time Scaling) | arXiv 2505.23621 v1 2025-05-29, EMNLP 2025 | SFT/RLVR 7B | 초록·README에 언급 없음 | 아니오(학습, 검색 없음) |
| GraphOTTER | arXiv 2412.01230, 2024-12-02, COLING 2025 | 단일 표 → 그래프 추론, HiTab EM 73.74%(Qwen2-72B) | 없음 | 아니오(72B, 검색 아님) |
| TabSQLify | arXiv 2404.10150, NAACL 2024 | text-to-SQL로 부분표, WikiTQ 64.7%·TabFact 79.5% | 없음 | 문제 다름 |
| ITR | ACL 2023 (amazon-science/robust-tableqa) | 행·열 대조학습 검색, WikiTQ·WikiSQL | 없음 | 아니오(학습) |
| HybridRAG | arXiv 2408.04948, 2024-08-09, ICAIF 24 | 실적발표 녹취 KG+벡터 | 없음, 표 없음 | 문제 다름 |
| HC-RAG | arXiv 2608.12335, 2026-06-03 | 증거 그래프, FinQA·TAT-QA·DocFinQA·FinanceBench·Multi-Doc-2025 | 없음 | 문헌만 |
| BM25→Corrective RAG 벤치 | arXiv 2604.01733, 2026-04-02 | 재무 텍스트+표 검색 전략 10종 비교, BM25가 dense 우세 | 없음 | 혼합 검색 정당화 근거로 인용 |
| 공식 리더보드 | codalab.lisn.upsaclay.fr/competitions/6738 | 비공개 test | 1위 long.yin 51.788, 2위 46.169, 3위 45.594. 지표(EM/F1)·날짜 미확인 | — |

## 3. 권고

- **확인한 후보 중 가장 강한 비교군(이번 주 실행 가능)**: 이미 있는 셋 — (1) TableRAG(Chen 2024) 셀·스키마 표현 이식판, (2) 같은 논문의 RowCol·RandRow, (3) TableRAG(Yu 2025) 청킹 이식판 + 고정 1,000자 청크 + 표/행 단위. 2024-09 이후 조사 범위에서 "셀 텍스트 표현만 바꿔 같은 검색기에 넣을 수 있는" 무학습·무API 방법은 이 외에 없었다.
- **문헌 대조만**: MixRAG(H-RCL은 표현 아이디어로 인용, 재현은 GPT-4o 의존·문서 단위), FT-RAG(문제 정의 가장 근접, 코드 미확인), MoCA-Agent·Fortune·GRAB(학습 또는 27B), Alonso & Lapata(dev·EM 표 단위 검색 선례), DocMath simplong·Srivastava(벤치마크/프롬프트).
- 서술 주의: Chen 2024·Yu 2025 모두 MultiHiertt에서 평가된 적 없음. 우리 표에는 "원 논문 재현"이 아니라 "표현 이식 변형"으로 명기.

## 4. 열어본 URL

arxiv.org/abs/2410.04739, arxiv.org/html/2410.04739, github.com/google-research/google-research/blob/master/table_rag/README.md, arxiv.org/abs/2506.10380, arxiv.org/html/2506.10380, github.com/yxh-y/TableRAG, arxiv.org/abs/2504.09554, arxiv.org/html/2504.09554, github.com/ChiZhang-bit/Mixture-of-RAG, github.com/psunlpgroup/MultiHiertt, codalab.lisn.upsaclay.fr/competitions/6738 (#results, #learn_the_details-evaluation), arxiv.org/abs/2401.13223, github.com/yale-nlp/DocMath-Eval, docmath-eval.github.io, arxiv.org/html/2311.09805, arxiv.org/abs/2505.23621, github.com/Table-R1/Table-R1, arxiv.org/abs/2412.01230, arxiv.org/html/2412.01230, arxiv.org/abs/2511.06582, arxiv.org/html/2511.06582v2, arxiv.org/abs/2408.04948, arxiv.org/abs/2605.01495, arxiv.org/abs/2604.01733, arxiv.org/abs/2607.17742, arxiv.org/abs/2608.26949, arxiv.org/html/2608.26949, arxiv.org/abs/2609.11390, arxiv.org/html/2505.20368v1, arxiv.org/abs/2608.12335, arxiv.org/html/2608.12335, arxiv.org/html/2606.28916, arxiv.org/pdf/2605.29606, arxiv.org/html/2606.11537, arxiv.org/abs/2505.23667, arxiv.org/abs/2402.11194, aclanthology.org/2025.findings-emnlp.225 (+pdf). 검색만(미열람): TabSQLify·ITR·Fortune GitHub·TAT-LLM GitHub 요약.
