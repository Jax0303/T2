# 사전등록: TableRAG(Chen) 호출당 생성 상한 128 → 512 재실행 — 2026-09-29, 실행 전

## 배경 (사용자 질문 "Chen 은 코드가 틀렸다는 거야? 고쳐야 하는 거 아님?")
기존 Chen 200 문항 실행(`baselines/tablerag_chen/`, 원본 설정 max_tokens=128)의 풀이 기록을 분류한 결과(사후):
- 코드 실행 관찰 1,014건 중 SyntaxError 360건, 그중 343건이 잘림형("was never closed", "unterminated")이고 127/200 문항에서 발생.
- 원인: 원본 `TableAgent.max_tokens=128`(vendor/table_rag/agent/agent.py:46)이 Thought 문장과 Action 코드를 합쳐 128토큰에서 자른다.
  원논문은 GPT-3.5/4 를 썼고, Qwen3-8B 는 Thought 를 길게 써 코드가 줄 중간에서 잘린다 — LLM 치환(같은 조건 규칙)에서 생긴 부작용.
- 그 밖의 오류(ValueError 67, KeyError 31 등)는 원본 dtype 추론·계층 열 이름에서 오는 것으로 이번 변경 대상이 아니다.
- 우리 구현 코드의 결함은 확인되지 않았다(오류는 LLM 이 작성한 코드의 실행 오류).

## 조건
- 변경: 모든 LLM 호출(질의 확장·풀이)의 생성 상한 128 → 512 (`tablerag_chen.py --max-tokens 512`). 512 = TableRAG(Yu) 실행의 호출당 상한과 같은 값.
- 나머지 전부 기존 실행과 동일: 200 문항(ids 앞 200), Qwen3-8B 4bit 비생각 greedy, bge-base, top_k 5, max_depth 5, PYTHONHASHSEED=0, 같은 DB 캐시.
- 출력: `results/mh_only_20260928/baselines/tablerag_chen_mt512/` (기존 파일 덮어쓰지 않음).

## 보고 규칙 (결과 전 고정)
- 논문 본문 표(검색 표 4-5, 답변 표 4-8, 부록 A-2)의 TableRAG(Chen) 행은 **512 결과**로 한다. 128(원본 설정) 결과는 부록에 병기한다. 결과 방향과 무관하게 이 규칙을 따른다.
- 본문에 이탈(상한 변경)과 이유(잘림 343/360)를 적는다.

## 예측
1. 잘림형 SyntaxError 가 기존 343건에서 80% 이상 줄어든다.
2. 답변 정확도가 기존 12/200 보다 오르지만 본 방법 93/200 보다 낮고, McNemar p < .05.

## 실행
`baselines/tablerag_chen_mt512/run.sh` — GPU 잠금 대기, Yu 답변 200 이 끝난 뒤 시작.
