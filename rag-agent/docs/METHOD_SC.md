# METHOD_SC.md — Self-Consistency with table-verified weighted voting

갱신: 2026-06-08 · 코드: `scripts/method_grounded.py` (`--mode sc`, `sc_vote`,
`--mode sc_plain`/`cot` baselines) · 결과: `results/method_{naive,grounded,hpir,sc}_pc20.json`(seed42),
`results/method_{grounded,sc}_s123.json`(seed123) · 채점: `scripts/rescore.py`

> 본 문서의 모든 수치는 **저장된 예측을 현재(수정된) 채점으로 재계산한 값**이다(아래 §5 버그 참조).
> 미완료 실험(plain-SC·CoT)은 **수치를 싣지 않는다.**

---

## 1. 무엇을 하는가 (제안 방법)
들어온 쿼리에 대해 답을 **한 번만 생성하지 않고**, grounded codegen으로 **여러 번** 생성한 뒤
**원본 표(헤더트리)에 비춰 grounding이 맞는 후보에 가중치를 줘 다수결**한다.
- floor = grounded greedy 답을 항상 후보에 포함(회귀 방지)
- 각 샘플의 셀 접근 trace가 깨끗(NO_MATCH/EMPTY 없음)하면 가중치↑
- floor보다 *엄격히* 더 받은 답만 채택 → 단일생성 grounded 밑으로 잘 안 내려감

## 2. 왜 그렇게 했는가 (근거)
사전 실패분석(`FAILURE_ANALYSIS.md`): 답변 실패의 **82%가 "코드는 실행되는데 엉뚱한 셀/연산"**
(예외 0건) = *조용한 grounding 오류*. 모델이 틀린 줄 모른 채 자신 있게 틀린다 → 한 번 생성으론
못 잡는다. 그래서 **여러 샘플 + 표-검증 투표**로 일관되게 맞는 grounding에 수렴시키는 처방.

## 3. 실험 셋업 (통제)
- 데이터: **HiTab dev의 hard-class 부분집합** (multi_op_formula / arithmetic_agg / pair_or_topk_arg /
  single_arg / comparison_or_count) — **쉬운 simple_lookup은 제외**. 클래스당 20개 × 5 = **N=100**.
- 검색=**gold 표 고정**(=답변단계만 격리; table-reasoning 논문 관행과 동일).
- 모델: **Qwen2.5-7B-Instruct 4-bit (로컬)**. 지표: **NM(±2%)** + **EM(strict denotation)**.
- seed=42(in-sample), seed=123(out-of-sample). paired 통계(bootstrap CI, exact McNemar).

> ⚠️ **절대 수치 주의**: 본 평가는 *가장 어려운 부분집합 + 소형 로컬모델*이라 절대값이 낮다.
> HiTab 논문들의 전체셋·대형/파인튜닝 모델 수치와 **직접 비교 금지**. 본 기여의 주장은
> *동일 통제 하의 상대 향상*이다.

## 4. 결과 (확정)

| 방법 | seed42 NM | seed42 EM | seed123 NM | seed123 EM |
|---|---|---|---|---|
| naive (direct codegen) | 0.0500 | 0.0300 | — | — |
| HPIR (header-path hint, global) | 0.2800 | 0.1600 | — | — |
| grounded (schema + self-repair) | 0.3200 | 0.1800 | 0.2500 | 0.1100 |
| **SC (제안)** | **0.4200** | **0.2700** | **0.3400** | **0.1700** |

**SC vs grounded (paired):**
| 비교 | ΔNM | 통계 |
|---|---|---|
| seed42 (in-sample) | +0.1000 | — |
| seed123 (out-of-sample) | +0.0900 | McNemar p=0.0039 (b01=9, b10=0) |
| **pooled N=200** | **+0.0950** | 95%CI [+0.0500, +0.1450], McNemar p=0.00016 |

→ **SC가 단일생성 grounded를 NM·EM 양쪽, 두 seed 모두에서 이기고, 합산 N=200에서 유의(p≈2e-4).**

## 5. 방법론적 발견 — 채점 버그 정정
초기엔 naive가 NM 0.41로 가장 높아 보였으나, 채점을 점검하니 **빈 예측('')을 정답으로 세는 버그**였다.
`scripts/rescore.py`로 저장된 예측을 수정 채점하니 **naive는 실제 NM 0.05 / EM 0.03**. 이 정정이
"naive가 엔티티 질문에 강하다"는 잘못된 결론(및 그에 기반한 라우팅 시도)을 뒤집었다.

## 6. 시도했으나 이기지 못한 것 (정직)
- **HPIR(전역 헤더경로 힌트)**: NM 0.28 < grounded 0.32. 엔티티 답 질문을 망침.
- **검색측 HPIR 확장**(`results/hpir_retrieval.json`): dense R@1 0.6164 → 0.5937 로 **하락**.
- **답변전략 라우팅**: 버그 정정 전 수치로 정책을 골라 일반화 실패(out-of-sample에서 역전).
→ 결론: 본 데이터/모델에서 **효과가 확인된 것은 SC**.

## 7. 한계 / 다음 단계
- N=100/run은 소표본(같은 grounded가 seed42 0.32 vs seed123 0.25). 대표본 필요.
- 절대값은 hard-subset+7B 탓 → 전체셋/대형모델로 보강 가능.
- **아직 안 돌린 베이스라인(분야 비교용)**: plain Self-Consistency(단순 다수결, 코드 `--mode sc_plain` 준비됨),
  Chain-of-Thought(`--mode cot` 준비됨), Chain-of-Table/DATER, HiTab 원논문 수치 인용.
  (코드는 들어있으나 **실행·수치는 미완료** → 본 문서에 싣지 않음.)
