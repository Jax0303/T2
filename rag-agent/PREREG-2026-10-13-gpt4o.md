# 사전등록: GPT-4o 로 RAG 없이·RAG 출력으로 답변 정확도 확인 — 2026-10-13, 실행 전

## 왜
지도교수 지시(2026-10-13, 2026-10-12 보고 PPT 의 882 답변 정확도 표를 보고): 더 좋은 모델인 GPT-4o 로 **RAG 를 거치지 않고 바로** 돌리면
어떻게 나오는지 확인하라. 말미에 **RAG 에서 나온 것을 GPT 에 넣어 보라**고도 함. 둘 다 본다.
이 실험은 논문의 폐쇄 환경 제약(외부 API 없음, CLAUDE.md §1) 밖의 **참고 비교**다. 본 방법의 수치가 아니다.

사용자 예상(실행 전): 낮게 나올 것 — MultiHiertt 에 주석을 더해 쉽게 만든 파생 데이터셋으로 LLM 들을 평가한 논문(DocMath-Eval 로 기억, 미대조)에서
GPT-4o 가 62 정도였다고 기억.

"RAG 없이"의 해석 둘(2026-10-13 사용자 결정: ② 주 조건, ① 보조):
- ① 질문만: 표를 전혀 보여 주지 않는다. GPT-4o 가 데이터를 이미 알고 있는지 보는 조건.
- ② 표 전부: 검색 없이 질문의 문서에 있는 표 전부를 그대로 넣는다.

## 고정 조건
- 문항: MultiHiertt train 882(`results/mh_arms/cap300_20260924/cell_uniq.jsonl`, 셀 문장 분해 실험과 같은 문항·순서). 정답은 공식 파일.
- LLM: `openai:gpt-4o`(별칭. 실제 서빙된 스냅샷 이름은 응답의 `model` 을 결과 요약 `reader_details.served_model` 에 기록), temperature 0,
  최대 출력 384 토큰, 한 문항에 한 호출, seed 없음(API 는 완전 재현을 보장하지 않는다 — 같은 입력을 다시 돌리면 다를 수 있음을 결과에 적는다).
  기본 URL 은 OpenAI 공식(`.env` 의 OPENAI_COMPAT_BASE_URL 은 다른 제공자라 무시), 키는 `.env` 의 OPENAI_API_KEY_OLD(`--reader openai:gpt-4o?key_env=OPENAI_API_KEY_OLD`).
- 지시문: LangChain rlm/rag-prompt **원문 그대로**, 사용자 메시지 하나, system 메시지 없음(`langchain_chunk.py --prompt original`) — PPT 의 882 결과와 같음.
  ①만 문맥 문장("Use the following pieces of retrieved context to answer the question.")을 뺀 지시문(`--prompt closed`), Context 줄 없음.
- 채점(주): **정답 포함 여부**(`results/cellform_20261012/score_inclusion.py`, PPT 표와 같은 채점). 보조: 공식 EM/F1(코드가 같이 계산), 마지막 숫자 채점.
- 입력 토큰 수는 OpenAI 응답의 `usage.prompt_tokens` 를 그대로 기록(`n_tok`), 출력은 `n_tok_out`, 잘림은 `finish_reason`.

## 조건 (`results/gpt4o_20261013/build_inputs.py` 로 ①②를 만든다. ③④는 저장된 882 입력 그대로)
| 조건 | 넣는 것 | 정답 셀 포함 | 셀 평균 | 토큰 평균(Qwen3 토크나이저, 사전 계산) |
|---|---|---|---|---|
| ① closed | 질문만 | 0 | 0 | 0 |
| ② fulldoc | 문서의 표 전부, LangChain 청킹 원문과 같은 markdown(`langchain_chunk.doc_text`, 머리글 v3.3u) | 882 | 146.9 | 1,404 |
| ③ ours | 본 방법 검색 순위대로 셀 문장, LangChain 청크 4개와 같은 길이(`results/cellform_20261012/C_ours_budget_records.jsonl`) | 862 | 43.7 | 1,257 |
| ④ langchain | LangChain 상위 청크 4개 원문(`results/cellform_20261012/A_chunk_md_records.jsonl`) | 867 | 127.3 | 1,273 |
- 이 882 에서는 **검색이 입력량을 줄이지 않는다**(② 1,404 ≈ ③ 1,257 ≈ ④ 1,273). ②는 정답 셀을 항상 담는다.
- 문단은 넣지 않는다(이 문항 집합은 표 근거만 필요한 질문이다). 넣으면 평균 2,490 토큰이 더 든다.
- 실행 순서: ②·③ 동시에(프로세스 둘), 끝나면 ④·①. 실행 전 ②에서 2문항 연기 시험(서빙 모델·usage 필드 확인, 결과는 판정에 안 씀).
- 비용 추정(실행 전): 입력 합계 약 370만 토큰, 출력 50만 토큰 이하.

## 비교와 판정 (같은 문항 정확 McNemar)
| 비교 | 보는 것 | Holm |
|---|---|---|
| ② 대 ③ | 검색이 GPT-4o 에게 도움이 되는가(표 전부 대 본 방법 RAG 출력) | 둘 묶음 |
| ③ 대 ④ | 같은 LLM 에서 본 방법 대 LangChain(PPT 표의 GPT-4o 판) | 둘 묶음 |
| ① | 서술만(예/아니오 54문항은 찍어도 50% 맞는다 — 유형별로 적는다) | — |
| GPT-4o ③④ 대 Qwen3-8B 4비트 ③④(같은 입력, `results/cellform_20261012/ans_*`) | LLM 을 바꾸면 얼마나 오르나 | 보조, 서술 |
- 조회/계산, 4그룹(층)별로도 적는다. 오답은 ②③④ 공통 오답 수를 센다(LLM 이 아니라 문항·채점 쪽 원인 후보).

## 예측 (실행 전 고정)
- ① 10% 이하. 맞힌 것은 대부분 예/아니오 문항.
- ② ≥ ③ ≥ ④. ②-③ 차이는 3%p 이내(유의하지 않을 것), ③-④ 는 유의(조회 문항에서).
- GPT-4o ③·④ 는 같은 입력의 Qwen3-8B 4비트(48.2%·43.7%)보다 10%p 이상 높다.
- 사용자 예측: 전체 60% 전후 이하.

## 이탈 기록 (실행 중 바뀐 것)
- (없음)
