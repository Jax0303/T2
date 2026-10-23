# notes — 사전등록·문헌 대조·작업 기록 (달별)

`rag-agent/` 루트에 쌓이던 날짜 문서를 달별 디렉터리로 옮겼다(9월분은 2026-10-20 T2-1 복사본에서, 9·10월 전부는 2026-10-23 이 저장소에서).
새 문서는 `notes/YYYY-MM/` 에 둔다(예: `notes/2026-10/PREREG-2026-10-23-….md`). 파일 이름 규칙은 그대로다: `종류-YYYY-MM-DD-주제.md`.

- 사전등록(`PREREG-*`)은 실행 뒤 본문을 고치지 않는다(각 문서 머리말). 상태가 바뀌면 이 색인만 갱신한다.
- 문서끼리의 참조는 파일 이름만 쓴다. 이름이 유일하므로 다른 달 문서도 이름만으로 찾는다. 코드·결과 디렉터리의 주석도 파일 이름으로 가리킨다.
- `CLAUDE.md`·`README.md` 는 루트에 남는다. 결과 표·보고(`results/*.md`)는 옮기지 않았다.
- HiTab 노선 문서는 2026-09-28 지도교수 지시로 주 평가·원고에서 빠졌다(`PREREG-2026-09-28-mh-only.md`). 10월의 HiTab 실행(10-19)은 참고·부록용이다.
- 10월의 현재 진행 기준: `PREREG-2026-10-23-baselines-newregime.md`(비교군 전부 새 설정 재실행 결과), `PREREG-2026-10-22-reader-sweep.md`(리더 교체·오답 원인).
  원고 정본은 `thesis/thesis_working_20261023_v23.hwpx`.

## 2026-09 (43)

| 날짜 | 파일 | 종류 | 데이터셋 | 내용 |
|---|---|---|---|---|
| 09-08 | `PREREG-2026-09-08-policy-answer-em.md` | 사전등록 | HiTab | 청킹 정책 4조건(P1~P4)의 답변 정확도 |
| 09-08 | `PREREG-2026-09-08-reader-prompt.md` | 사전등록 | HiTab | 리더 프롬프트 개입, 단위 불일치 결함 25건 제외 |
| 09-09~10 | `BUGFIX_LOG.md` | 버그 기록 | HiTab | 채점한 문맥 ≠ 리더가 받은 문맥 (P1 2건+2건), gold 조건 재실행 |
| 09-10 | `LITERATURE-2026-09-10-retrieval-vs-answer.md` | 문헌 대조 | — | 검색 정확도가 답변 정확도로 이어지지 않는 현상(원문 PDF 대조) |
| 09-10 | `PREREG-2026-09-10-tablerag-fair.md` | 사전등록 | HiTab | TableRAG(Chen) 를 표 하나 범위(원 설정)에서 다시 잼 |
| 09-11 | `EVALUATION-INTEGRITY-2026-09-11.md` | 평가 무결성 | HiTab | 근거 계약 v2, "gold 는 상한 아님", 청킹 행은 원 시스템 재현 아님 |
| 09-12 | `HANDOFF-2026-09-12-audit.md` | 인수인계 | HiTab | 슬라이드 RowCol 수치 출처 감사, 전체 경로 복원 진단 |
| 09-12 | `PREREG-2026-09-12-five-representations.md` | 사전등록 | HiTab | 다섯 표현 비교(행 보존 Markdown 청킹 추가) |
| 09-13 | `PREREG-2026-09-13-header-units-note.md` | 사전등록 | MultiHiertt | 헤더 행 추정 규칙 v2 |
| 09-13 | `PREREG-2026-09-13-multihiertt-arms.md` | 사전등록 | MultiHiertt | 다중 조회·산술 검색 정확도·답변 EM, 모집단 정의 |
| 09-13 | `PREREG-2026-09-13-reader-qwen3.md` | 사전등록 | MultiHiertt | 리더 설정 고정(Qwen3-8B, cot, 384토큰) |
| 09-13 | `PREREG-2026-09-13-table-label.md` | 사전등록 | MultiHiertt | 표 고유 라벨(표 직전 문단) — 검색을 낮춰 채택 안 함 |
| 09-13 | `REPORT-2026-09-13.md` | 작업 기록 | 둘 다 | 비교군 실사, 검색-EM 간극, MultiHiertt 첫 결과. `analysis/mh_final_report.py` 가 생성 |
| 09-13 | `VERIFICATION-2026-09-13.md` | 코드 대조 | 둘 다 | HiTab 채점기·TableRAG(Chen)·TableRAG(Yu) 원본 코드와 함수 실행 대조 |
| 09-14 | `DATA-USE-2026-09-14.md` | 데이터 사용 이력 | MultiHiertt | train/validation 사용 이력, 규칙을 train 표본 보고 고친 기록 |
| 09-14 | `PREREG-2026-09-14-header-v3.md` | 사전등록 | MultiHiertt | 원표→문장 변환 규칙 v3(정정 1~4 = v3.1~v3.3) |
| 09-18 | `LITERATURE-2026-09-18-context-pruning-top1.md` | 문헌 대조 | — | 검색 후 top-1 만 남기기(스니펫만 확인, 원문 재확인 필요) |
| 09-18 | `RESULTS_KSWEEP.md` | 결과 | HiTab | 문맥 크기 k 대 리더 EM(bge-large 조건, 300건) |
| 09-19 | `PREREG-2026-09-19-selector-gold-fairness.md` | 사전등록 | HiTab | LLM 셀렉터를 다른 셀 단위 arm 에도 적용해 공정성 검증 |
| 09-21 | `LITERATURE-2026-09-21-k-ladder-convention.md` | 문헌 대조 | — | 검색 지표는 k별, 답변 정확도는 운영점 하나(네 편 PDF 대조) |
| 09-21 | `PREREG-2026-09-21-fair-llm-filtering-hitab.md` | 사전등록 | HiTab | LLM 필터링을 7개 arm 전부에 적용한 공정 비교 |
| 09-23 | `PREREG-2026-09-23-reader-thinking-pilot.md` | 사전등록 | MultiHiertt | 리더 재선택: 비생각 cot 대 생각 모드(시간 가드로 제외) |
| 09-23 | `THESIS-INTERIM-2026-09-23.md` | 중간 정리 | 둘 다 | 논문 설정·결정·수치 요약(9월 23일 시점). 그 뒤 결정은 `CLAUDE.md`·`PREREG-2026-09-28-mh-only.md`·10월 문서에 있다 |
| 09-24 | `PREREG-2026-09-24-mh-answer-cap300.md` | 사전등록 | MultiHiertt(+HiTab 표 전체) | 답변 재실행(그룹당 상한 300, 채택 리더), "표 전체" 비교군, 고유화 v3.3u 이탈 기록 |
| 09-26 | `PREREG-2026-09-26-judging-criteria.md` | 사전등록 | 둘 다 | 문서·표 특정 사람 판정 기준 |
| 09-26 | `PREREG-2026-09-26-rerun-dev-alpha.md` | 사전등록 | 둘 다 | 주 결과표 재실행, dev 에서 라벨 섞기 비율 선택 |
| 09-26 | `PREREG-2026-09-26-s3c-answer-hitab300.md` | 사후 조건 | HiTab | s3c 답변 정확도 300건 |
| 09-27 | `PREREG-2026-09-27-reader-format.md` | 사전등록 | MultiHiertt | 리더 입력 형식: 셀 문장 20셀 대 행 확장 |
| 09-27 | `PREREG-2026-09-27-rerank.md` | 사전등록 | 둘 다 | 교차 인코더 재정렬 진단(본 방법에 포함 안 함) |
| 09-28 | `PREREG-2026-09-28-fair-mh.md` | 사전등록 | MultiHiertt | 공정 비교: 분리 지표, LLM 필터 단계. 같은 날 `mh-only` 로 대체 |
| 09-28 | `PREREG-2026-09-28-hitab538-answer.md` | 사전등록 | HiTab | 538표 한 색인 조건 답변 300건 |
| 09-28 | `PREREG-2026-09-28-lookup-arith-posthoc.md` | 사후 분석 | MultiHiertt | 조회 대 산술 1위 적중률 차이 정의 고정 |
| 09-28 | `PREREG-2026-09-28-mh-name-append.md` | 사전등록(진단) | MultiHiertt | 정답 열 이름을 질문에 붙인 검색 진단 |
| 09-28 | `PREREG-2026-09-28-mh-only.md` | 사전등록 | MultiHiertt | MultiHiertt 단독 재평가(지도교수 지시): GATE, 분리 지표, dev 로 k 선택, test 실행 |
| 09-28 | `PREREG-2026-09-28-table-description.md` | 사전등록 | MultiHiertt | 데이터셋 `table_description` 문장 대 본 방법 셀 문장 |
| 09-28 | `PREREG-2026-09-28-tablerag-official-dtype.md` | 사전등록 | 둘 다 | TableRAG(Chen) 재구현을 원본 숫자 열 규칙으로 재검색, 사후 대체 |
| 09-29 | `PREREG-2026-09-29-baselines-original.md` | 사전등록 | MultiHiertt | 비교군 4개를 원논문 방법(공식 코드)으로 재구성, 표준 Recall@k·Precision@k·MRR, 공식 EM+F1 |
| 09-29 | `PREREG-2026-09-29-bigdoc.md` | 사전등록 | MultiHiertt | 큰 문서 조건(표 전체 입력이 길어질 때) |
| 09-29 | `PREREG-2026-09-29-chen-max-tokens.md` | 사전등록 | MultiHiertt | TableRAG(Chen) 호출당 생성 상한 128→512 재실행 |
| 09-29 | `PREREG-2026-09-29-evidence-groups.md` | 사전등록 | MultiHiertt | 공식 근거 셀 성격별 보조 분할 표 |
| 09-29 | `PREREG-2026-09-29-gold-only-answer.md` | 사전등록 | MultiHiertt | 정답 셀만 입력한 답변 상한(882) |
| 09-29 | `PREREG-2026-09-29-header-ablation.md` | 사전등록 | MultiHiertt | 헤더 경로 제거 ablation |
| 09-30 | `PREREG-2026-09-30-cell-tutorial-prompt.md` | 사전등록 | MultiHiertt | 본 방법 셀 문장 + LangChain 튜토리얼 지시문 |

## 2026-10 (42)

10월은 전부 MultiHiertt 882문항(새 문항 600·dev 395 포함)이 기준이고, HiTab 이 적힌 것만 HiTab 이다.
입력 이름: 잘라낸 표(D) · 셀 문장 목록(C) · LangChain 청크(A). 10-15 이후 지시문은 P*, 리더는 Qwen3.5-4B 무양자화.

| 날짜 | 파일 | 종류 | 내용 |
|---|---|---|---|
| 10-01 | `PREREG-2026-10-01-alpha-dev.md` | 사전등록 | 하이브리드 가중치 α 검증 집합 비교(0.7 유지) |
| 10-06 | `PREREG-2026-10-06-claim-chain.md` | 사전등록 | 주장의 인과 사슬을 잇는 실험 A·B·C |
| 10-06 | `PREREG-2026-10-06-post-defense.md` | 사전등록 | 심사 후 보완 실험(dev k=1·3, Yu 882, 1.7B·bge-m3) |
| 10-07 | `PREREG-2026-10-07-claims.md` | 사전등록 | 주장 확정 실험 1~5 |
| 10-08 | `PREREG-2026-10-08-bigdoc-scale.md` | 사전등록 | 큰 문서 조건 확장: 문서가 LLM 에 안 들어갈 때 셀 단위 대 청크 검색 |
| 10-08 | `LATER-2026-10-08-levers.md` | 기록 | 나중에 고려할 것(기록만, 지금 안 함) |
| 10-08 | `MEETING-2026-10-08-kim.md` | 회의 | 세션 결론 + 지도교수 질문 대비 답변 |
| 10-12 | `PREREG-2026-10-12-cellform-dev.md` | 사전등록 | 셀 문장 입력 개선 후보 dev 선택(K1~K4) |
| 10-12 | `PREREG-2026-10-12-cellform.md` | 사전등록 | 셀 문장의 답변 이득은 어디서 오는가(형식 대 셀 고르기) |
| 10-12 | `PREREG-2026-10-12-main-baselines-lcprompt.md` | 사전등록 | 주 비교군 = LangChain 청크·RAPTOR, 지시문 = LangChain 튜토리얼 |
| 10-13 | `PREREG-2026-10-13-gpt4o.md` | 사전등록 | GPT-4o 로 RAG 없이·RAG 출력으로 답변 정확도 확인 |
| 10-13 | `PREREG-2026-10-13-yesno-judge.md` | 사전등록(사후 수정) | 예/아니오 문항 채점에 사람 판정 추가 |
| 10-14 | `PREREG-2026-10-14-score-fix.md` | 사전등록(사후 수정) | 정답 포함 채점 구멍 3개 수정(형식 허용 v2) |
| 10-15 | `PREREG-2026-10-15-prf.md` | 사전등록 | 검색 결과를 질문 앞에 붙여 다시 검색(의사 적합성 피드백) |
| 10-15 | `PREREG-2026-10-15-prompt-order.md` | 사전등록 | 지시문에서 검색 결과를 질문 앞에 두기(순서 효과) |
| 10-15 | `PREREG-2026-10-15-prompt-sweep.md` | 사전등록 | 양자화 없는 4B 로컬 LLM + 지시문 전수 탐색(P* 선택) |
| 10-15 | `PLAN-2026-10-15-t2ragbench.md` | 계획(미등록) | T²-RAGBench 에서 본 방법 재기 |
| 10-17 | `PREREG-2026-10-17-delivery-format.md` | 사전등록 | 전달 형식 비교: 검색한 셀을 어떤 모양으로 LLM 에 넘길 것인가 |
| 10-17 | `PREREG-2026-10-17-delivery-format-882.md` | 사전등록 | 전달 형식 882 확인(D 대 A·C) |
| 10-18 | `PREREG-2026-10-18-D-vs-A-replication.md` | 사전등록 | D 대 A 882 반복 실행 판정 |
| 10-18 | `PREREG-2026-10-18-table-cap.md` | 사전등록 | 잘라낸 표의 표 수 제한(D1·D2·DF2), dev 395 |
| 10-19 | `PREREG-2026-10-19-882-cap4096.md` | 사전등록 | 882 확인 실행을 4,096 토큰 표준으로 3회(D·C·A) |
| 10-19 | `PREREG-2026-10-19-delivery-ceiling.md` | 사전등록 | 전달 단계 상한 진단(정답 셀만 O1·정답 표 O2·O3), 882 |
| 10-19 | `PREREG-2026-10-19-hitab-confirm.md` | 사전등록 | HiTab 확인 표본 691, D 대 A, 3회(참고) |
| 10-19 | `PREREG-2026-10-19-hitab-newregime.md` | 사전등록 | HiTab 300 새 설정 재측정(A·C·D), 3회(참고) |
| 10-19 | `PREREG-2026-10-19-hitab-pathform.md` | 사전등록 | HiTab 691 경로형 목록 대 문장형 목록(참고) |
| 10-19 | `PREREG-2026-10-19-mh-confirm-arith.md` | 사전등록 | 새 문항 600(산술 셀 2개 이상)에서 D 대 A, 3회 |
| 10-19 | `PREREG-2026-10-19-prompt-table-rule.md` | 사전등록 | 지시문에 표 설명·표끼리 합산 금지 규칙 추가, dev 395 |
| 10-19 | `PREREG-2026-10-19-sentence-form.md` | 사전등록 | 셀 문장 형태(경로형 대 문장형) × 꺽쇠 설명, dev 395 |
| 10-19 | `PREREG-2026-10-19-thinking-dev.md` | 사전등록 | 생각 모드 dev 395 측정 |
| 10-20 | `PREREG-2026-10-20-bracket-explain.md` | 사전등록 | 꺽쇠 경로 설명 지시문 추가 확인(새 문항 600) |
| 10-21 | `PREREG-2026-10-21-ceiling-600.md` | 사전등록 | 새 문항 600 전달 단계 상한 진단(O1·O3) |
| 10-21 | `PREREG-2026-10-21-decompose-dev.md` | 사전등록 | 질문 쪼개기 검색, dev 395, 검색 단계만 |
| 10-21 | `PREREG-2026-10-21-select-dev.md` | 사전등록 | 셀 고르기 단계(줄 단위 선택), dev 395 |
| 10-22 | `PREREG-2026-10-22-budget-forcing.md` | 사전등록 | 잘린 답변 마무리 단계 |
| 10-22 | `PREREG-2026-10-22-multiprompt-600b.md` | 사전등록 | 지시문 4개 × 세 입력 × 3회, 새 문항 600b |
| 10-22 | `PREREG-2026-10-22-multitable-confirm.md` | 사전등록 | 정답이 표 2개 이상에 걸친 새 문항 190, 독립 확인 |
| 10-22 | `PREREG-2026-10-22-question-type-882.md` | 사전등록 | 질문 종류별(조건형 대 값 지정형) D 대 A, 882 |
| 10-22 | `PREREG-2026-10-22-reader-sweep.md` | 사전등록 | 리더 LLM 교체(로컬 무양자화 2종 + GPT-4o), 오류 원인 분류 |
| 10-22 | `PREREG-2026-10-22-self-consistency-dev.md` | 사전등록 | self-consistency dev 395(사용자 지시로 중단) |
| 10-22 | `PREREG-2026-10-22-width-multitable.md` | 사전등록 | 표 너비·정답 표 수에 따른 격차 확인 |
| 10-23 | `PREREG-2026-10-23-baselines-newregime.md` | 사전등록 | 비교군 전부 10월 주 설정(882, Qwen3.5-4B 무양자화) 재실행, 원고 4장 표 교체. **현재 진행 기준** |
