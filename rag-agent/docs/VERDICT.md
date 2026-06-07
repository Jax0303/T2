# VERDICT.md — 판정 (수치 + 규칙만, 형용사 없음)

갱신: 2026-06-07 · 근거: `results/phase2_retrieval.json`, `results/phase2_answers.json`,
`results/phase2_by_class_r1.json`. (Phase 3 완료 시 Gate 1·3 추가)

## 설정
전체 코퍼스 3597표 · dev 1671질의 · 검색 임베더 bge-large-en-v1.5 · 답변 LLM
local Qwen2.5-7B-Instruct(4bit) · 답변 컨텍스트 top-1 표(plain_markdown, 4000자 절단) · seed=42.

## Phase 2 검색 (라우팅 없음)
| method | R@1 | R@5 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| dense_header_path | 0.490 | 0.775 | 0.832 | 0.616 | 0.664 |
| dense_plain_markdown | 0.390 | 0.669 | 0.736 | 0.515 | 0.563 |
| hybrid_rrf | 0.376 | 0.672 | 0.775 | 0.514 | 0.571 |
| dense_json_kv | 0.321 | 0.611 | 0.677 | 0.452 | 0.501 |
| bm25(튜닝 k1=0.9,b=0.4) | 0.241 | 0.472 | 0.550 | 0.347 | 0.389 |

R@1 paired bootstrap(1000,95%CI): 전 method가 best-dense 대비 유의 하락(sig=True).

## Phase 2 답변 end-to-end
| baseline | R@1 | EM | NM | F1 |
|---|---|---|---|---|
| oracle(상한) | 1.000 | 0.268 | 0.447 | 0.330 |
| dense_header_path | 0.490 | 0.177 | 0.290 | 0.219 |
| bm25 | 0.241 | 0.101 | 0.157 | 0.124 |
| nocontext(하한) | 0.000 | 0.031 | 0.087 | 0.064 |

Gate 2: 상한(oracle≥검색) True · 하한(nocontext≤검색) True · BM25 grid 로그 존재 · leakage_note.md 존재 → **통과**.

## retrieval–answer gap
- across-baseline Spearman ρ(R@1, NM) = **+1.000** (p<0.001).
- per-query 조건부:
  - dense_header_path: P(정답|top1 적중)=0.494 (n=818) vs P(정답|불일치)=0.094 (n=853).
  - bm25: P(정답|top1 적중)=0.483 (n=402) vs P(정답|불일치)=0.054 (n=1269).

## Decision Gate 판정 (§5)

**Gate 2 (dense가 BM25를 R@k 격차 >0.4로 압도 → "VDB 불필요" 폐기):**
격차 = R@1 0.249, R@5 0.303, nDCG 0.275 — **전부 <0.4 → 트리거 미충족**.
단, 전체 3597풀에서 dense는 BM25를 모든 지표에서 유의하게 이김(이전 540풀의 "어휘 우위 → VDB 불필요"
결론은 풀 확대 시 **뒤집힘**). 판정: "VDB 불필요" 프레이밍은 540풀에 국한된 artifact. 풀 크기가
결론을 결정하는 통제변수임을 보고.

**Gate 4 (R@k는 올랐으나 EM 정체·ρ 약함 → gap 자체를 기여로):**
ρ=+1.000, EM도 R@1 따라 단조 증가(dense 0.177 > bm25 0.101 > nocontext 0.031) → **트리거 미충족**.
HiTab 전체코퍼스에서는 검색 개선이 답변 개선으로 직결됨(TARGET ρ=−0.85과 반대 양상). 즉 본 데이터셋의
gap은 "검색이 무의미"가 아니라 "검색이 병목이자 지렛대"임을 보고.

**추가 관찰(병목 분해):** 정답표를 줘도(oracle) NM=0.447 → 실패의 약 55%는 **답변(LLM/표읽기) 측**,
나머지는 검색 측(dense R@1=0.49). 두 병목이 곱으로 작용. 큰 표 4000자 절단이 oracle 상한을 일부 눌렀을
가능성은 `leakage_note.md`/한계로 기록.

**Gate 1 (oracle-router vs always-dense), Gate 3 (학습형 라우터<oracle 90%):** Phase 3에서 판정 → TODO(measure).
