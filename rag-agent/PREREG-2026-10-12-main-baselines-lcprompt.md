# 사전등록: 주 비교군 = LangChain 청크·RAPTOR, 답변 지시문 = LangChain 튜토리얼 — 2026-10-12, 실행 전

## 결정 (사용자, 2026-10-11~12)
- 답변 비교 지시문은 LangChain 튜토리얼 rlm/rag-prompt(+ 'Final answer:' 한 문장) 하나로 통일한다. 사용자 메시지 하나, system 없음.
- 주 비교군은 LangChain 청크와 RAPTOR. 근거: 두 논문·튜토리얼 모두 답변 지시문을 방법의 일부로 주장하지 않는다
  (RAPTOR 논문 arXiv 2401.18059 은 QA 지시문을 싣지 않고 부록 D 요약 지시문만 싣는다).
- Chen(arXiv 2410.04739, 3.3 "Core Components"에 코드 실행 풀이, 부록 D·E·F 지시문)과 Yu(arXiv 2506.10380, 4단계·부록 G 지시문)는
  지시문이 방법 단계라 바꾸지 않는다. **보류** — 나중에 각자 원래 지시문으로 같은 문항을 돌린다.
- 문제 정의·검색 범위(질문의 문서)는 바꾸지 않는다.

## 이미 있는 결과 (같은 지시문, 882문항, 다시 돌리지 않음)
- 본 방법 셀 문장 20개: `results/post_defense_20261006/prompt2x2/cells_lcprompt/answer.jsonl`
- LangChain 청크 4개: `results/mh_only_20260928/baselines/langchain_chunk/langchain_chunk_answer.jsonl`

## 새 실행 (results/main_lcprompt_20261012/)
1. **RAPTOR 200문항, LangChain 지시문.** 기존 RAPTOR 검색 결과(`baselines/raptor/raptor_records.jsonl`, 2,000토큰, 원래 코드의
   노드 결합 그대로)의 `context_2000` 을 `langchain_chunk.py --stage answer` 에 넣는다. 코드 변경 없음, 레코드 형식만 바꾼다
   (`context ← context_2000`, `correct ← correct_2000`, `cells_in_context ← cells_2000(이미 개수)`, `m ← len(gold_ids)`).
2. **RAPTOR 882문항으로 확장.** `raptor.py --limit 882 --tree-dir baselines/raptor/trees`(트리 캐시 재사용, 없는 문서만 새로 만든다),
   `--batch-size 128`. 그 `context_2000` 으로 1과 같은 방법의 LangChain 지시문 답변. 원래 지시문 답변은 같이 생기지만 주 결과가 아니다.
- LLM Qwen3-8B 4bit 비생각 greedy, 384토큰, seed 42. 채점 공식 EM + 형식 허용(주), 같은 문항 정확 McNemar, 비교 둘(본 방법 대 각 비교군) Holm.

## 예측 (실행 전 고정)
- RAPTOR 2,000토큰 입력은 문서 평균 약 1,100토큰(cl100k)보다 커서 대부분 문서 전체 + 요약을 넘긴다 → LangChain 청크와 비슷한 수준.
  200문항 형식 허용에서 RAPTOR 는 LangChain(79)과 ±10 안, 본 방법(91)보다 낮다고 예측한다.
- 판정: 본 방법 대 RAPTOR 가 Holm 조정 p<.05 이면 "같은 지시문에서 본 방법이 높다". 아니면 수치만 적는다.

## 보고
- 결과는 이 파일 끝에 붙인다. 원고 반영은 나중(사용자).

## 결과 1 — 200문항 (2026-10-12 03:14, 형식 허용 / 공식 EM)
| 방법 (LangChain 지시문) | 허용 | 공식 | 조회 | 계산 | 정답 셀 포함 | 평균 입력 토큰 |
|---|---|---|---|---|---|---|
| 본 방법 셀 20개 | 91 | 80 | 68 | 23 | 178 | 699 |
| LangChain 청크 4개 | 79 | 60 | 51 | 28 | 197 | 1,398 |
| RAPTOR 2,000토큰 | 49 | 41 | 37 | 12 | 161 | 1,556 |
- 본 방법 대 LangChain 31:19, p=.119 / 본 방법 대 RAPTOR 52:10, p<.0001 (Holm 두 비교: RAPTOR 만 유의).
- **예측 틀림:** RAPTOR 를 LangChain ±10(69~89)으로 예측했으나 49. 원래 지시문 RAPTOR 는 77(같은 입력). 실행 오류 아님(답 표시 199/200,
  "모름" 1건, 출력 정상). 예: yes/no 질문에 원래 지시문은 "No", LangChain 지시문은 숫자를 답함. 원인은 단정하지 않는다.
- RAPTOR 입력은 원래 코드 get_text 대로 노드마다 줄바꿈을 공백으로 바꿔 한 줄로 이어 붙인 것(표 행 구분이 사라짐) — 기록만.
- 882 확장(raptor882_build)은 03:14 시작.
- 이탈(실행 중, 2026-10-12 03:25): 882 확장에서 원래 지시문 답변(raptor.py 2단계, 주 결과 아님, 약 2시간)은 건너뛴다.
  기록(raptor_records.json)이 끝나면 raptor.py 를 PID 로 멈추고 같은 변환 + LangChain 지시문 답변을 돌린다(`skip_orig.sh`).
  원래 지시문 882 가 필요하면 같은 --out-dir 에 --resume 으로 이어 돌릴 수 있다.
