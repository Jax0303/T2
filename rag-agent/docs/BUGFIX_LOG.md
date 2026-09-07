# BUGFIX_LOG

## 2026-08-31 — `gold_parts()` 콤마 오파싱 (Phase 4f/4g)

**대상**: `analysis/phase4_summary.py: gold_parts()`

**증상**: AIT-QA의 천단위 콤마 정답(`'1,179'`, `'13,200'` 등)에서 모델이
콤마 없는 동일 수치(`'1179'`, `'13200'`)를 출력했는데 오답으로 채점됐다.

**원인**: `ast.literal_eval('1,179')`는 괄호 없는 콤마 표현식을 튜플
리터럴로 파싱해 `(1, 179)`를 돌려준다. `gold_parts`가 이를 다중 gold
`['1', '179']`로 취급 → `em()`의 다중 gold 분기 진입 → 예측을 콤마로 split한
`['1179']`와 길이가 달라 즉시 0 반환. `norm_em`(= 통화기호 제거, 천단위 콤마
제거)은 호출조차 되지 않았다.

**성격**: 사전등록된 정규화 규칙(천단위 콤마 제거)이 실행되지 않은 구현
버그다. 채점 규칙 변경이 아니며 PREREGISTER 개정이 아니다.

**수정**: 문자열이 `[` 또는 `(` 로 시작할 때만 `ast.literal_eval` 컨테이너
파싱을 시도한다. 그 외는 단일값으로 취급한다. `em()`, `norm_em()`, `_same()`,
R1 규칙은 변경하지 않았다.

**영향**: 전 1156행 재채점 결과 0→1 7행, 1→0 0행. 전부 `aitqa`
(P1 1, P4 3, gold_cell 3). 다른 4개 pool 변동 없음.
query_id: q-401(P1), q-146/q-196/q-118(P4), q-396/q-118/q-267(gold_cell).

**재집계 산출물**: `results/phase4/FINAL.md` (`analysis/phase4g.py`).
수정 전 수치는 `results/phase4/final_summary.md`, `phase4b.md`, `phase4e.md`에
그대로 남겨 두었다.
