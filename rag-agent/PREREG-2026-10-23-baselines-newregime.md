# 사전등록 2026-10-23 — 비교군 전부를 10월 주 설정으로 재실행(882문항, Qwen3.5-4B bf16 무양자화), 원고 4장 답변 표 교체 (실행 전)

## 왜
- 사용자(10-23): "비교군도 다 새로 돌려야 함. LangChain 청크만 돌렸던 건 샘플로 결과만 빨리 보려던 것." 원고(10-01 발표판, `thesis/찐최종.hwpx`)의 답변 표 4-7 은 200문항·Qwen3-8B 4비트·각 방법 원래 지시문이다. 10월 주 설정(882, Qwen3.5-4B bf16, 본 방법 = 셀 문장 검색 + 채운 부분 표 전달 + 지시문 P*)으로 모든 비교군을 다시 돌려 표를 교체한다.
- 심사 지적(10-01)에도 맞다: 양자화 없는 모델, 리더 교체 강건성(`PREREG-2026-10-22-reader-sweep.md`).

## 고정 조건
- 문항: 882(`results/mh_arms/cap300_20260924/cell_uniq.jsonl` 의 id·순서). 정답 공식 train 파일.
- 리더: **모든 LLM 호출 = `Qwen/Qwen3.5-4B` bf16(양자화 없음), 비생각, greedy(temperature 0)**. 비교군은 `rag_agent.llm.factory` 로컬 HF 백엔드(`local:Qwen/Qwen3.5-4B?quantization=none`; 10-23 00:10 적재 확인: Qwen3_5ForCausalLM, 7.83GB, enable_thinking=False). 본 방법 D·C·A(P*)는 기존 vLLM 3회 결과(`format_dev_20261017/final882_cap4096_run1~3`) 재사용 — 같은 가중치·bf16·greedy, HF 한 건씩 대 vLLM 묶음의 정확도 차이 0/50(reader-sweep 재현 점검). 엔진 차이는 원고에 적는다.
- 각 비교군의 지시문·호출 구조·생성 상한은 10-01 설정 그대로(LLM 만 교체):
  | 팔 | 코드 | 입력·호출 | 생성 상한 | 검색 레코드 |
  |---|---|---|---|---|
  | LangChain 청크 | `scripts/baselines/langchain_chunk.py --stage answer`(tutorial 지시문 + 마지막 줄 문장) | 1,000자 청크 상위 4개, 1회 호출 | 384 | 10-01 `langchain_chunk_records.jsonl` 재사용(LLM 무관) |
  | RAPTOR | `scripts/baselines/raptor.py` | 붕괴 트리 2,000 토큰, 요약 100 토큰 | QA 384 | 트리는 요약 LLM 이 바뀌므로 **새로 생성** |
  | TableRAG(Chen) | `scripts/baselines/tablerag_chen.py --max-tokens 512`(10-01 본문 표 설정), `PYTHONHASHSEED=0` | 질의 확장 + 스키마·셀 검색 + PyReAct 최대 5회 | 512/호출 | 파이프라인 전체 새로(`--limit 882`: 882 부분집합만) |
  | TableRAG(Yu) | `scripts/baselines/tablerag_yu.py --stage answer` | 질의 분해·청크 검색·재정렬·SQL 최대 5회차 | 512/호출 | 1회차 검색 캐시 재사용(LLM 무관) |
  | 표 전체 입력(참고) | `scripts/prompt_sweep.py --engine vllm`, 입력 = 문서 표 전부 markdown(`results/gpt4o_20261013/fulldoc_records.jsonl`, 10-13 과 같음), **지시문 P\*** | 1회 호출 | 4,096 | 검색 없음 |
  | 본 방법 D(채운 부분 표, P*) | 기존 3회 | 상위 20셀 → 채운 부분 표 | 4,096 | — |
  | 본 방법 C(셀 문장 목록, P*)·A(LangChain 청크 + P*) | 기존 3회 | 지시문 통일 비교(10-01 부록 A 의 역할) | 4,096 | — |
- 채점(모든 팔 같음): 주 = **형식 허용 v2**(`results/score_fix_20261014/score_v2.included_v2`)를 각 방법의 최종 답에 적용(비교군은 파이프라인이 낸 `pred`, 본 방법·표 전체는 마지막 줄 값) + 예/아니오 규칙(`score_formats2.yn`); 보조 = 공식 EM(`mh_exact_match`). 10-14 규칙 그대로, 결과를 보고 바꾸지 않는다. 채점 스크립트 `results/baselines_newregime_20261023/score_newregime.py`(실행 전 작성).
- 통계: 본 방법 D(3회 다수결) 대 각 비교군, 같은 문항 정확 McNemar, Holm 묶음 = 비교군 5개(LangChain·RAPTOR·Chen·Yu·표 전체). 층별(조회 셀1·2+, 산술 셀1·2+) 표.
- 비교군은 1회 실행(에이전트 파이프라인 시간 때문), 본 방법은 3회 다수결. 1회 실행의 vLLM 잡음은 비교군에 없고(HF 한 건씩 = 결정적), 본 방법 쪽 ±10~18 은 해석에 반영.

## 예측 (실행 전 고정; 맞힘/882, 형식 허용 v2)
- 본 방법 D 620(70.3%, 기존). 표 전체 P* 600~640. TableRAG(Yu) 480~560. LangChain(tutorial, 384) 480~540. RAPTOR 380~470. TableRAG(Chen) 150~300.
- D 대 Yu·LangChain·RAPTOR·Chen: 모두 D 우세, Holm p<.05 (각 80%). 표 전체 대 D: 차이 20 이내 50%, D 우세 유의 30%.
- 10-01 순서(D > Yu > 표 전체 ≈ LangChain > RAPTOR > Chen) 유지 60%.

## 소요 (사전 추정: 10-01 실측 초/문항 × 882 ÷ 1.5~2(bf16 4B 가 4비트 8B 보다 빠름))
- 표 전체 P*(vLLM) 0.7h → LangChain 1h → RAPTOR 5~7h(트리 포함) → Yu 9~13h(10-01 89초/문항) → Chen 6~8h(10-01 50초/문항). 합계 22~30 GPU시간, 순차. 3문항 시험 실행으로 갱신한다.

## 선행 조건 (10-23 00:17 확인)
- **`vendor/` 가 이 PC 에 없다**(.gitignore; 집 PC 에만 있음): `vendor/raptor`(parthsarthi03/raptor `7da1d48`, 패치 0), `vendor/tablerag_yu`(yxh-y/TableRAG `f8d798f`, 패치 1: NL2SQL 지시문 "MySQL"→"SQLite"), `vendor/table_rag`(google-research `08a8d67` sparse + **우리가 쓴 `agent/local_backend.py`**(HuggingFaceEmbeddings·LocalQwen 백엔드) + `VENDOR_COMMIT`). 패키지 `tiktoken 0.14.0`·`umap-learn 0.5.12` 도 없다. 자동 모드에서 외부 코드 clone·pip 가 차단되어 **사용자가 집 PC 의 `rag-agent/vendor/` 를 복사하거나 직접 clone·설치해야 한다.** RAPTOR·Yu·Chen 은 그 뒤 시작.
- 그 전에 돌릴 수 있는 것: 표 전체 P*(vLLM), LangChain 답변(HF; 답변 단계는 langchain 패키지를 안 씀).

## 이탈 기록
(실행 전)

## 결과 (실행 뒤 채움)
