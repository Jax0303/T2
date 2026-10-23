# 사전등록: 셀 문장 입력 개선 후보 dev 선택 — 2026-10-12, dev 답변 실행 전

## 왜
`PREREG-2026-10-12-cellform.md` 882 결과: 같은 길이에서 본 방법 425 대 LangChain 385(+4.5%p), 이득은 셀 문장 형식·조회 문항에서.
계산 문항은 111 대 110 으로 같고, 본 방법 계산 오답의 다수가 필요한 값을 다 찾지 못함(139)·"못 구한다" 답(34, LangChain 18; 주가 수익률 12문항).
입력 쪽(같은 길이 규칙 유지)에서 고칠 후보를 dev 에서 고르고, 고른 하나만 882 로 확인한다. 평가 문항(882)으로 고르지 않는다.

## 고정 조건
- dev: MultiHiertt validation 332(`results/reader_format_20260927/mh_dev_pop332.jsonl`, 이전 k·α 선택에 쓴 문항). 정답 `official_answers("validation")`.
- LLM Qwen3-8B 4비트 비생각 greedy 384, 묶음 128. 지시문 LangChain 원문(`--prompt original`). 채점 정답 포함(`results/cellform_20261012/score_inclusion.py`).
- 예산 T = 문항별 LangChain 청크 4개 원문 토큰 수(Qwen3 토크나이저). dev LangChain 검색은 `langchain_chunk.py --stage retrieve --split validation`(이번에 `--split` 추가, train 기본 동작 그대로).

## 후보 (`results/cellform_dev_20261012/build_candidates.py`, 실행 전 생성)
| 이름 | 규칙 | 정답 셀 포함 | 셀 평균 | 토큰 |
|---|---|---|---|---|
| A | LangChain 청크 4개 markdown (기준) | 318 | 132.8 | 1,309 |
| C | 현재 본 방법: 점수 순 셀 문장, T 직전까지 | 325 | 46.7 | 1,293 |
| K1 | C 와 같은 셀, (표, 행, 열) 순으로만 다시 정렬 | 325 | 46.7 | 1,293 |
| K2 | 점수 순 셀마다 그 행의 나머지 셀을 먼저 채움, 행 단위 묶어 출력, T 직전까지 | 289 | 48.0 | 1,295 |
| K3 | 점수 순 셀을 같은 행끼리 한 줄로 "행 경로 > 열 경로: 값 \| …", T 직전까지 | 328 | 59.9 | 1,278 |
| K4 | C 규칙, 경로에서 인위 식별자 마디(Table k, row N, column N) 제거, T 직전까지 | 325 | 48.0 | 1,293 |
- 경로로 다시 만든 문장이 dev 모든 셀에서 색인 문장과 글자 단위로 같음(코드 assert).

## 선택 규칙 (결과 전 고정)
- K1~K4 중 dev 정답 포함 수가 가장 큰 하나. 그 수가 C 보다 **클 때만** 채택, 같거나 작으면 C 유지. 동률 후보는 이름 순 앞.
- 채택 후보는 train 882 에 같은 규칙으로 입력을 만들어 돌리고, 882 에서 채택 후보 대 C, 채택 후보 대 A 를 같은 문항 정확 McNemar(Holm 둘)로 본다.
- 예측: K3 가 가장 높다(셀 수 증가 + 행 흐름이 보여 계산·주가 수익률 거절이 준다). K2 는 정답 셀 포함이 줄어 C 보다 낮다.
