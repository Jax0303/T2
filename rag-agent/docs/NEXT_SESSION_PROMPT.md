# 다음 세션 시작 프롬프트 (붙여넣기용)

아래 블록을 그대로 복사해서 새 세션 첫 메시지로 붙여넣으면 됩니다.

---

OSC 완전성 연구 이어서 한다.

## ✅ 직전 세션(2026-07-03) 달성 — 견고화 3종 클리어 (커밋 92b39af, e13eddf, dc1fbe9)
1. **held-out 확인**: 설정 전부 동결(bge-reranker·top-2·k=10)하고 HiTab **test** split(arith m≥2, n=151)
   1회 실행 → 통과. same-depth 셋 다 p≤0.001·손해 0; **strict 셀-매칭도 셋 다 유의**
   (BM25 +.066 p=.002, dense +.060 p=.022, hybrid +.060 p=.012). Δ가 dev보다 작은 건
   test의 합계행 의존 쿼리가 적어서(29% vs 45%) — 메커니즘 예측대로. 사후선택 비판 방어 완료.
2. **쿼리 재작성 반례 제거**(`scripts/osc_query_rewrite_baseline.py`): " total overall" 덧붙이기는
   OSC 최대 +0.037, dense는 오히려 유의하게 손해(−.043). 필요 합계행 도달률 재작성 0.51~0.72 vs
   주입 0.88~0.92. 셀-매칭에서 주입이 재작성 검색기도 셋 다 이김(p≤.003). → §5.7 negative result.
3. **MultiHiertt 일반화 = 예측된 null**(`rag_agent/bench/multihiertt.py` + 
   `scripts/osc_total_augment_multihiertt.py`, n=226, gold 셀 99.1% 대조검증 내장):
   재무테이블은 합계에 "Total" 이름이 붙어 plain이 이미 0.94~1.00 도달 → 주입 Δ0.000, 손해 0.
   **병리는 코퍼스 속성**, §5.1b 진단이 배포 전 테스트 역할. OSC가 HiTab 밖에서 처음 측정됨.
   (데이터: `data/multihiertt/dev.json`, HF `yilunzhao/MultiHiertt`에서 다운로드, git 제외)

## 다음 목표 (택1)
1. **HiTab gold 수동 감사**: 쿼리 50개 샘플로 value-matching gold와 is_total_row 휴리스틱의
   정밀도/재현율 측정 → 부록 한 문단 (반나절, 저비용 보험).
2. **end-to-end 정확도**: 주입 전/후 답 정확도. 솔버가 관건 — groq 70b는 rate-limit 심함,
   로컬 또는 배치 API 필요. ⚠️ `results/e8_ohd_baseline.json`은 150/161 rate-limit 에러라
   증거로 못 씀(재실행 대상, 커밋 안 함).
3. **figure 갱신**: `plot_osc_frontier.py`에 strict cell-matched 패널 + test-split 패널 추가.
4. 논문 §5 번호 정리(5.8이 두 주제를 담게 됨) + related work에 MultiHiertt 인용 추가.

## 환경 (중요 — 매 세션 *직접* 확인할 것)
- **작동 인터프리터 = `/usr/bin/python3`** (venv 아님. `.venv`는 deps 없음 — 쓰지 말 것).
- 실행: `PYTHONPATH=. /usr/bin/python3 scripts/...` · 작업 디렉토리 `/mnt/c/Users/ugh/T2/rag-agent`.
- 세션 시작 확인: `/usr/bin/python3 -c "import numpy,scipy,sentence_transformers,rank_bm25;print('OK')"`
- 모집단: HiTab dev n=161 / test n=151 / MultiHiertt dev n=226. 전부 LLM-free, seed 42.
- ⚠️ 이 저장소 파일을 Windows 편집기가 CRLF로 바꿔놓는 경우 있음 → 커밋 전 `file`로 확인,
  `sed -i 's/\r$//'`로 복원 (HEAD는 LF).
- ⚠️ MultiHiertt 셀값("$1,350")은 공유 `_to_float`이 못 읽음 → 어댑터가 float로 정규화함.
  공유 파싱 코드는 건드리지 말 것(HiTab 동결 결과가 흔들림).

## 결과·스크립트 요약
- 헤드라인: `results/osc_total_augment.json`(dev) + `results/osc_total_augment_TESTSPLIT.json`(held-out).
- 진단: `results/dense_ceiling_diag.json`(dev) + `_TESTSPLIT.json`(test, 65%/5.5배 재현).
- 반례: `results/osc_query_rewrite_{always,ratio}.json`.
- 일반화: `results/osc_total_augment_multihiertt.json`(진단+주입+검정 통합).
- 논문: `docs/PAPER_DRAFT.md` — §5.10(승리+held-out), §5.7(재작성 negative), §5.8(MultiHiertt 경계조건).
- 발표: `docs/LAB_MEETING_BRIEF.md` ★★2026-07-03 섹션.

## 정직성 가드 (잊지 말 것)
- WebFetch 자동요약은 환각함 → 논문 인용 전 원문 확인.
- "열거 단독은 dense보다 낮다, 승리는 증강에서 온다"를 숨기지 말 것.
- 셀 수 항상 같이 보고. strict-superset 승리는 "같은 깊이+오버헤드"임을 명시.
- Bonferroni ×3 시 strict dense는 marginal(.067) — 논문에 명시돼 있음, 지우지 말 것.
- MultiHiertt null은 실패가 아니라 경계조건 — "이겼다"로 왜곡하지 말 것.

각 단계 결과 보여주고 유의성까지 보고해줘.
