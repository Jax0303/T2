# 선행연구 대조 — "검색 성공 시 top-1만 남기고 나머지 버리면 답변 정확도가 오르는가" (2026-09-18)

교수님 코멘트(관련 없는 후보를 버리고 관련 있는 것만 남기면 답변 정확도가 오른다)를 계기로,
20개 검색 결과 중 정답이 든 1개만 남기고 19개를 버리는 것이 RAG 연구에서 타당/가능한지,
2026년 기준 선행연구가 어디까지 왔는지 찾아봤다.

**읽은 방식 주의**: 이번 세션은 WebSearch 스니펫/요약만 확인했다. `LITERATURE-2026-09-10`처럼
원문 PDF를 열어 수치·정의를 대조하지 않았다. 아래 내용을 실험 설계나 논문 인용에 쓰기 전에
반드시 원문을 열어 재확인할 것.

**결론**: 두 갈래로 나뉜다. (1) oracle/gold-context 상한선 진단 실험으로는 이미 표준 기법이라
타당함. (2) 실제 배포 시스템에서 "성공 후보만 남기고 나머지를 버린다"는 메커니즘 자체는
2023~2025년 사이 여러 논문이 이미 다룬 문제라 새 기여가 아니다. 게다가 "무조건 1개로
고정 절단"하는 형태는 최근 연구에서 반박되는 지점이 있다.

## 1. 이미 존재하는 연구 계보 (동일 방향, novelty 없음)

| 방법 | 저자/연도 | arXiv | 방식 |
|---|---|---|---|
| Self-RAG | Asai et al., 2023 | 2310.11511 | reflection token으로 각 검색 결과의 관련성 판정 후 사용 여부 결정 |
| Corrective RAG (CRAG) | Yan et al., 2024 | 2401.15884 | 경량 평가기가 신뢰도 판정 → decompose-then-recompose로 핵심만 남김, 3단계(Correct/Ambiguous/Incorrect) 대응 |
| FILCO | Wang et al., 2023 | 2311.08377 | string inclusion / lexical overlap / conditional cross-mutual information으로 컨텍스트 필터링, 프롬프트 길이 최대 64% 감소 |
| RECOMP | Xu, Shi & Choi, 2023 | 2310.04408 | 관련 없으면 빈 요약(empty summary) 반환하는 선택적 압축 |
| Sparse RAG | ICLR 2025 | 2405.16178 | 검색 문서마다 LLM이 관련성 점수 산정, 낮은 것 동적 드롭 |

다섯 편 모두 "관련 없는 후보를 버리고 관련 있는 것만 남긴다"는 발상을 이미 구현·검증했다.
이 아이디어 자체를 새 기여로 제시할 수는 없다.

## 2. "top-1 고정 절단"에 대한 반박 근거

위 방법 중 어느 것도 "무조건 1개만 남긴다"는 고정 규칙을 쓰지 않는다. 이유:

- **Cuconasu et al., *The Power of Noise*, SIGIR 2024, arXiv:2401.14887**: 무작위 비관련
  문서를 추가했을 때 정확도가 최대 35%까지 오히려 오르는 현상 보고. 노이즈 제거가 항상
  정확도를 높인다는 가정이 단조적이지 않다.
  - 반박: **The Powerless Noise, SIGIR 2026, arXiv:2607.03615** — 원 세팅에서는 재현되지만
    프롬프트 문구·디코딩 설정을 바꾸면 효과가 약해지거나 사라진다고 보고. 아직 논쟁 중인 현상.
- **LeVine & Varjavand, *Relevance Isn't All You Need*, ICLR 2025, arXiv:2504.07104**: 관련성만
  극대화해 후보를 좁히는 방식이 오히려 최종 응답 품질을 떨어뜨릴 수 있음을 보이고, depth·
  diversity·clarity 등 다기준 재순위화를 대안으로 제시. "가장 관련 있는 1개만 남긴다"는
  설계와 정면으로 배치.
- **The Distracting Effect, ACL 2025, arXiv:2505.06914**: 비관련 문서의 방해 효과는 이진이
  아니라 연속적이며, 순위가 높을수록 방해 효과도 커진다고 보고. 연속 스코어링이 단순 top-1
  절단보다 근거 있는 접근이라는 뜻.

## 3. MultiHiertt/HiTab류 멀티홉 설정에 대한 구조적 문제

멀티홉 QA는 정의상 정답 도출에 **2개 이상의 증거(셀/텍스트) 결합**이 필요한 질의를 포함한다
("no single fragment suffices"). Top-1 고정 절단은 단일 증거로 충분한 질의에는 유효하지만,
다중 증거가 필요한 질의에서는 정답 도달이 원천적으로 불가능해진다 — 튜닝으로 해결되는 문제가
아니라 설계 자체가 상한을 낮추는 문제.

## 4. 방법론 주의 — oracle leakage

"검색이 성공했으면"에서 "성공 여부"를 정답 라벨로 판정해 거른다면, 이는 테스트 시점에 정답을
미리 아는 **oracle/gold-context 상한선 실험**이다 (예: arXiv:2601.19827의 "GOLD" 세팅 — 검색기가
이보다 잘할 수 없는 상한선으로 명시하는 방식). 이 수치는 **실제 시스템 성능으로 보고하면 안
되고 상한선(ceiling)으로만 보고해야 한다** — 실제 추론 시점에는 어떤 후보가 "성공"인지 미리
알 수 없어 순환논리가 된다. `real-ceiling-not-data-2026-09-17` 메모의 상한선 분석 방법론과
같은 프레임으로 처리하면 된다.

## 5. 우리 파이프라인에 시스템적으로 적용 가능한 것

`bottleneck-investigation-2026-09-16`에서 이미 관측: `structural_leaf + top20 LLM selector`
체인에서 top20 전체를 넘기지 않고 selector가 1개를 골라 넘겼을 때 답변 정확도가
.6307 → .7316로 상승. "관련 있는 것만 남기면 정확도가 오른다"는 이미 실증됨. 남은 병목은
selector가 가끔 잘못 고르는 것(selector mis-pick).

**CRAG식 confidence-gated 가변 k** — 다음 수순으로 테스트 가능:

- 지금처럼 무조건 1개로 강제하지 말고, selector가 확신도(confidence)를 같이 출력.
- 확신 높음 → top-1만 전달 (지금과 동일).
- 확신 낮음/애매 → top-3 정도로 fallback (버리지 않고 몇 개 더 남김).
- 근거: CRAG(2401.15884)의 3단계 구조, Relevance Isn't All You Need(2504.07104)의 "확신
  없는 상태에서 무조건 1개로 줄이면 품질 하락" 결과.
- 멀티홉(≥2 증거 필요) 질의에 대한 안전판도 됨 — 확신 낮은 케이스가 다중 증거 필요 케이스와
  겹칠 가능성.
- 이건 검색(retrieval) 쪽이 아니라 selector/reader 쪽 개입이라, 이미 소진된 retrieval
  escalation 라인(`colbert-rerank-closed-2026-09-17`)과 겹치지 않는 새 축. 기존 top20 selector
  체인 데이터로 "확신도-정답 여부" 상관관계부터 확인하면 바로 테스트 가능.

## 인용

- Asai et al. *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.* 2023. arXiv:2310.11511
- Yan et al. *Corrective Retrieval Augmented Generation.* 2024. arXiv:2401.15884
- Wang et al. *Learning to Filter Context for Retrieval-Augmented Generation.* 2023. arXiv:2311.08377
- Xu, Shi & Choi. *RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation.* ICLR 2024. arXiv:2310.04408
- *Accelerating Inference of Retrieval-Augmented Generation via Sparse Context Selection (Sparse RAG).* ICLR 2025. arXiv:2405.16178
- Cuconasu et al. *The Power of Noise: Redefining Retrieval for RAG Systems.* SIGIR 2024. arXiv:2401.14887
- *The Powerless Noise: How Experimental Settings Shape the Reported Power of Noise.* SIGIR 2026. arXiv:2607.03615
- LeVine & Varjavand. *Relevance Isn't All You Need: Scaling RAG Systems With Inference-Time Compute Via Multi-Criteria Reranking.* ICLR 2025. arXiv:2504.07104
- *The Distracting Effect: Understanding Irrelevant Passages in RAG.* ACL 2025. arXiv:2505.06914
