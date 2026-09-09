# 선행연구 대조 — "검색 정확도가 답변 정확도로 이어지지 않는다" (2026-09-10)

우리 관찰(검색 .9142 → 답변 .7164)이 새 발견인지 확인하려고 원문을 읽고 대조했다.
**결론: 현상은 확립돼 있다. 우리 설계는 표준이다. 새로 주장할 수 있는 자리는 셋뿐이다.**

읽은 방식: PDF/HTML 전문. 검색 스니펫만으로 인용한 항목은 없다. 다만
Power of Noise 의 Table 1 격자는 텍스트 추출이 깨져 본문 서술문과 ADORE 수치만 썼다.

## 1. 우리 수치와 선행 수치

`Cuconasu et al., The Power of Noise, SIGIR 2024, arXiv:2401.14887` 은 우리와 **같은
실험**을 한다 — gold 문서를 아는 오라클 세팅에서 distracting 문서를 하나씩 더한다.
정의도 우리 상황 그대로다(§4.2, 원문):

> **Distracting documents**, while not directly answering the query, are semantically
> or contextually linked to the topic. For instance, if one asks for the color of
> Napoléon's horse, a passage describing the color of Joséphine de Beauharnais'
> horse … would [be distracting].

우리 20셀이 정확히 이것이다 — 같은 표, 같은 행/열의 이웃 칸.

| 조건 | Power of Noise (Llama2, NQ) | **본 연구 (Qwen2.5-7B-4bit, HiTab, n=862)** |
|---|---:|---:|
| gold 만 | 0.5642 | **0.9629** |
| + hard distractor 1 | 0.4068 | — |
| + 2 | 0.3815 | — |
| + 4 | 0.3626 (−0.2016, −36%) | — |
| + 9 (문맥 10셀) | — | **0.8353** (−0.128) |
| + 19 (문맥 20셀) | — | **0.7958** (−0.167, −17%) |

원문 서술: *"a clear pattern of progressive accuracy degradation as the number of
distracting documents … increases … deteriorating by more than 0.38 (−67%) in some
cases. Even more importantly, adding just one distracting document causes a sharp
reduction in accuracy, with peaks of 0.24 (−25%)."*

`The Distracting Effect, arXiv:2505.06914` Table 1 (gold only → gold + hard distractor):

| 모델 | 하락 |
|---|---:|
| Llama-3.2-3B | 82.6 → 71.5 (−11.1pt) |
| Llama-3.1-8B | 80.6 → 73.9 (−6.7pt) |
| Llama-3.3-70B | 81.1 → 75.2 (−5.9pt) |
| **본 연구 (7B, distractor 19개)** | **96.3 → 79.6 (−16.7pt)** |

**정직하게 읽을 것**: 저쪽은 distractor 1개, 우리는 19개다. **개당 피해는 우리가 훨씬
작다.** 셀은 문서보다 작아 하나가 덜 해롭지만 개수가 많아 총 피해가 크다고 써야 한다.

## 2. 검색기가 좋을수록 더 해로운 오답을 가져온다 — 이미 측정돼 있다

Power of Noise, hard-negative 로 학습된 최신 검색기 ADORE 로 distractor 를 고른 경우:

    distracting 1개 0.4068 / 2개 0.3815 / 4개 0.3626
    (distractor 없음 = 0.5642, Contriever 로 고른 경우보다 유의하게 낮음)

이것이 우리 표 1·2 의 순위 뒤집힘(MT2Net 검색 .8153 > 고정청킹 .7669 인데 답변
.5631 < .6024)을 설명하는 기전이고, **문헌이 이미 인정한 기전**이므로 인용할 수 있다.

## 3. k 곡선 — RAGGED (arXiv:2403.09040)

reader 4개(FlanT5-XXL, Flan-UL2, LLaMa2 7B/70B) × NQ·HotpotQA·BioASQ:

> decoder-only 모델은 문맥창이 길어도 **문서 5개 이하만 효과적으로 쓴다.** LLaMa 두
> 모델 모두 NQ 에서 k≤3, HotpotQA 에서 k≤2 에 정점을 찍고 이후 꾸준히 하락.
> Flan(encoder-decoder)은 k=20 부근 정점.

우리 리더는 decoder-only 7B 이고 **1셀이 정점**이다. 방향 일치. 최적 k 가 더 작은 것은
단위가 문서가 아니라 셀이기 때문으로 설명된다.

## 4. 반박 논문 — The Powerless Noise (arXiv:2607.03615, SIGIR 2026)

Power of Noise 재현 검증. 원 세팅에서는 재현되지만 **프롬프트 문구와 디코딩 설정을
조금만 바꿔도 효과가 나타나거나 약해지거나 사라진다**고 보고한다.

⚠️ 이 반박의 표적은 **"무작위 문서가 도움이 된다"** 쪽이다. **"distracting 문서가
해롭다"는 반박 대상이 아니다.** 우리 주장은 후자라 직접 타격은 아니다. 다만 리뷰어가
"당신 결과도 프롬프트·디코딩에 민감한 것 아니냐"고 물을 것이므로 **리더를 바꿔
곡선을 다시 그려두면** 그 질문이 막힌다.

## 5. 표 QA — CABINET (ICLR 2024 spotlight, arXiv:2402.01155)

주장 동일: *"only a small part of the whole table is relevant … irrelevant parts act as
noise and are distracting information, resulting in sub-optimal performance."* 표 크기별
저하도 보인다(500셀 이상 WikiTQ 에서 CABINET 38% vs OmniTab 28%).

**겹치지 않는 이유**: CABINET 은 **표 하나가 주어진** 상태에서 표 안을 걸러낸다.
다중 표 검색을 하지 않고, **계층표를 다루지 않는다.**

## 6. 대조 결과

| 주장 | 선행연구 | 우리 위치 |
|---|---|---|
| 검색 지표가 답변 정확도를 예측 못 한다 | 있음 (UDCG, arXiv:2510.21440) | 재확인 |
| 의미적으로 가까운 오답이 해롭다 | 있음 (Power of Noise, Distracting Effect) | 재확인 |
| 후보를 더 넣으면 떨어진다 | 있음 (RAGGED) | 재확인 |
| 좋은 검색기가 더 해로운 오답을 가져온다 | 있음 (ADORE 실험) | 재확인 |
| 표에서 무관한 셀이 노이즈다 | 있음 (CABINET) | 재확인, 단일 표 한정 |
| **EM = P(hit)×P(정답\|hit) 분해** | **없음** | **빈자리** |
| **색인 단위가 P(정답\|hit) 를 바꾼다** | **없음** — 전부 passage 고정 | **빈자리** |
| **계층표 + 다중 표 검색** | **없음** | **빈자리** |

원문에서 명시적으로 확인한 부재:
- UDCG: end-to-end 정확도를 retrieval × conditional 로 **분해하지 않는다**.
  granularity 를 바꾸지 않는다 — **전부 full passage**, §7.4 는 k(1~10)만 바꾼다.
- Distracting Effect: **분해하지 않는다.** grounded/ungrounded 를 따로 보고할 뿐.
  **텍스트 구절만 다루고 표는 없다.**
- RAGGED: recall@k 와 reader EM 을 같이 싣지만 **조건부 정확도를 분리하지 않는다.**

## 7. 그러므로 논문에 쓸 기여는 셋

1. **분해 프레임.** `EM = P(검색 성공)·P(정답|성공) + P(실패)·P(정답|실패)`.
   실측 `0.7164 = 0.9142×0.7792 + 0.0858×0.0471`. 문헌은 두 항을 따로 보고할 뿐
   곱으로 묶지 않는다. 이 프레임이 "검색 .9면 답변 .9" 기대가 왜 틀리는지에
   산수로 답한다.
2. **색인 단위가 두 번째 항을 바꾼다.** 발표된 색인 단위 8개에서 검색 순위와 답변
   순위가 뒤집힌다. 문헌은 전부 passage 로 고정하고 k 만 바꾼다.
3. **계층표 + 다중 표 검색 세팅.** CABINET(단일 표)도 TableRAG(표별 인덱스)도
   다루지 않는다.

**"distractor 가 해롭다"를 기여로 쓰면 안 된다.** 그건 SIGIR 2024 다.

## 인용

- Cuconasu et al. *The Power of Noise: Redefining Retrieval for RAG Systems.* SIGIR 2024. arXiv:2401.14887
- Trappolini, Cuconasu, Filice, Maarek, Silvestri. *Redefining Retrieval Evaluation in the Era of LLMs.* arXiv:2510.21440
- *The Distracting Effect: Understanding Irrelevant Passages in RAG.* arXiv:2505.06914
- Hsia et al. *RAGGED: Towards Informed Design of Scalable and Stable RAG Systems.* arXiv:2403.09040
- Patnaik et al. *CABINET: Content Relevance based Noise Reduction for Table Question Answering.* ICLR 2024 spotlight. arXiv:2402.01155
- *The Powerless Noise: How Experimental Settings Shape the Reported Power of Noise.* SIGIR 2026. arXiv:2607.03615
