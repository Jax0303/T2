# FAILURE_ANALYSIS.md — 나이브 codegen 실패 원인 분해 (방법 설계의 근거)

생성: 2026-06-07 · 근거: `results/phase3_FULL.json` (adaptive, 로컬 Qwen2.5-7B-4bit,
hard-class dev 476쿼리, 540-pool) · 분해 스크립트는 본 문서 §4 재현.

## 1. 한눈 요약
나이브 codegen(질의→분류→VDB검색→LLM이 Python 생성→실행)의 정답률 NM=0.202(96/476).
실패를 원인별로 기계 분해한 결과:

| 원인 | 건수 | 전체대비 | 답변실패중 |
|---|---|---|---|
| ① 검색 실패(틀린 표) | 139 | 29.2% | — |
| ⑥ 코드 실행됨, 값 틀림 (silent wrong) | **198** | 41.6% | **82.2%** |
| ④ 숫자 근소차(<10%) | 18 | 3.8% | 7.5% |
| ③ 코드 없음(direct 경로 fallback) | 17 | 3.6% | 7.1% |
| ⑤ 타입 불일치(gold 숫자/pred 문자) | 8 | 1.7% | 3.3% |
| 정답 | 96 | 20.2% | — |

두 축의 병목: **검색(29%)** 과 **답변(71%)**. 답변 실패의 **82%가 bucket⑥**.

## 2. 핵심 진단 — "조용한 grounding 오류"
bucket⑥ 198건 중 **179건이 숫자를 출력**(실행 예외/에러 문자열 0건). 즉 코드는 정상 실행되는데
**엉뚱한 셀을 짚거나 엉뚱한 연산**을 해서 틀린다. 실측 사례:

- gold 51.5 ← 코드 `cell("south asia","family class")>=52 ? ...` 헛논리 → `'3'`
- gold 59.9 (두 셀 합) ← `100 - 6.0` → `'94.0'`
- gold 3.4 ← 셀 둘 빼기, 자릿수/연산 어긋남 → `'34'`
- gold -0.0726 ← 잘못된 열 바인딩으로 `colnum(...).iloc[0]` → `'1059'`

원인 메커니즘: 나이브 helper `cell("문자열","문자열")`/`find_col(...)`이 **헤더를 자유추측**으로
매칭하고, **매칭이 빗나가도 예외 없이 조용히 잘못된 값/NaN을 반환**한다. 모델은 자기 grounding이
틀린 걸 알 신호를 못 받는다. (HiTab 질문 다수가 평서문 형태라 의도 파악 실패도 일부 겹침.)

## 3. 방법 설계로의 함의 (노블티 포인트)
- 기존 **Self-Debugging(Chen et al., ICLR 2024)**, **LEVER(Ni et al., ICML 2023)**, **CodeT(Chen et al., 2022)**
  는 *실행 예외 / 단위테스트 / 실행 일치* 신호로 코드를 고친다. 그러나 **여기 실패의 거의 전부가 예외 0건**
  — 코드가 멀쩡히 돌며 틀린다. 기존 신호로는 안 잡힌다.
- 따라서 제안 방법의 차별점 = **grounding-trace 피드백 기반 자가수정**:
  1. **스키마 바인딩**: 실제 top/left 헤더경로 인벤토리를 모델에 제공 → 자유추측 금지, 존재 경로에 바인딩.
  2. **grounding 추적**: 셀 접근 API가 (요청 헤더 → 매칭된 경로/열·행 인덱스 → 반환값, 모호/빈값 플래그)를
     로그 → 코드가 에러 없이 끝나도 이 **트레이스를 모델에 되먹여** 조용한 오류를 교정(≤k회).
  3. **숫자 verifier(±2%)** 로 최종 점검.
- 토대 재사용: `rag_agent/stores/original_store.py`의 `value_at`/`find_rows_by_header`/`resolve`가 이미
  헤더경로 매칭(정확+퍼지)을 제공 → 여기에 "트레이스 로깅"을 입혀 API로 노출.

## 4. 재현
```python
# results/phase3_FULL.json 의 rows를 (correct, table_correct, code, answer, gold_answer)로
# 버킷팅: 검색실패 / 코드없음 / 숫자근소차(<10%, ×100·÷100 허용) / 타입불일치 / 값틀림
```
결과 카운트: `results/phase_naive_failure_buckets.json`.

## 5. 다음
- 제안 방법(Grounding-trace codegen) 구현 → 동일 질의셋에서 나이브 codegen 대비 NM·버킷 변화 측정.
- 평가: 로컬 Qwen(무제한). 반복은 층화 소표본 → 확정 비교는 hard-class 전체.
- (Groq는 일일 토큰 한도 소진으로 보류; `docs/leakage_note.md`/한계에 기록.)
