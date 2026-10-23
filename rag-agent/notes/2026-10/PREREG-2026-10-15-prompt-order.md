# 사전등록: 답변 지시문에서 검색 결과를 질문 앞에 두기 — 순서 효과 200문항 시험 (2026-10-15)

## 왜
모든 882 답변 실행(Qwen3-8B 4비트 두 조건·GPT-4o 네 조건)은 LangChain rlm/rag-prompt 원문 순서 그대로 **Question → Context → Answer** 다
(`scripts/baselines/langchain_chunk.py --prompt original`). 사용자 지적(2026-10-15): 검색 결과를 질문 앞에 두었어야 한다.
긴 입력 뒤에 질문을 두는 것이 일반 권고(Liu et al. 2023 query-aware contextualization, Anthropic/OpenAI 프롬프트 지침)이고,
이전 본 방법 지시문(cot)은 Context → Question 이었다. 10-07 2×2 의 "지시문 효과"에는 순서 차이가 섞여 있었다.
원문에서 벗어나는 변경이므로, 먼저 200문항으로 순서 효과가 있는지 본 뒤 882 재실행 여부를 정한다.

## 조건
- 모델·입력·채점 고정: Qwen3-8B 4비트, 비생각 모드, greedy, 최대 384토큰(cellform 882 와 같음). 입력은 `results/cellform_20261012/C_ours_budget_records.jsonl`(본 방법)·`A_chunk_md_records.jsonl`(LangChain) 그대로.
- 바뀌는 것 하나: `--prompt original_cf` = 지시문 문장은 원문 그대로, `Context: … \nQuestion: … \nAnswer:` 순서. (`PROMPT_CF`, 이 커밋에서 추가)
- 문항: 882 중 200, `results/order_pilot_20261015/ids200.jsonl`(시드 20261015 셔플 뒤 앞 200, query_id 정렬).
- 비교 기준: 같은 200문항의 기존 Question-first 출력(`C_ours_budget`·`A_chunk_md`). 채점 `results/score_fix_20261014/score_v2.included_v2`;
  예/아니오 정답에 yes/no 낱말 없는 출력은 10-13 규칙대로 판정표(새 출력분은 Claude 가 같은 기준으로 판정해 `judgments_cf.csv` 에 적고 사용자 확인).
- 검정: 조건마다 순서 전·후 같은 문항 정확 McNemar. 본 방법 대 LangChain 차이도 전·후로 적는다.

## 예측 (실행 전)
- 순서를 바꾸면 두 조건 모두 0~+3%p(200 기준 0~+6문항). 유의(p<.05)까지는 안 갈 것.
- 본 방법 대 LangChain 차이는 ±2%p 안에서 유지(200 기준 전 차이 = 기존 출력에서 계산해 결과 절에 적음).
- 결정 규칙: 어느 한 조건이라도 +3%p 이상이고 p<.05 면 882 전 조건을 Context-first 로 재실행할지 사용자에게 묻는다. 아니면 원문 순서 유지, 결과는 부록 한 줄.

## 이탈 기록
- 첫 실행이 모델 적재에서 멈춤(연구실 PC venv 에 `accelerate` 없음) → 설치(1.15.0) 뒤 같은 명령으로 재실행. 출력 없음 상태였으므로 결과 영향 없음.
- 새 출력 중 예/아니오 정답에 yes/no 낱말 없는 19건(본 방법 12·LangChain 7)을 Claude 가 결론 방향으로 판정(`judgments_cf.csv`), 19건 모두 정답 방향. 사용자 확인 전.

## 결과 (2026-10-15 01:30, `results/order_pilot_20261015/score_cf.txt`)
| 조건 | 200 Question-first | Context-first | 0→1 : 1→0 | p |
|---|---|---|---|---|
| Qwen 본 방법 | 116 (58.0) | 113 (56.5) | 11 : 14 | .69 |
| Qwen LangChain | 97 (48.5) | 91 (45.5) | 11 : 17 | .35 |
- 본 방법 대 LangChain: Question-first 116 대 97(32:13 p=.007) → Context-first 113 대 91(34:12 p=.002). 차이 유지.
- 예측 대조: "0~+3%p" 예측과 달리 두 조건 모두 −1.5·−3.0%p(유의 아님). 결정 규칙대로 **원문 순서(Question → Context) 유지**, 882 재실행 없음.
- 유형별 바뀐 곳 없음(예/아니오 14→14·12→12, 산술 2+ LangChain 15→11 이 가장 큰 변화).
