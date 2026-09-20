# 선행연구 대조 — 검색 지표는 k별로, 답변 정확도는 운영점 하나 (2026-09-21)

§3 의 답변정확도@k 열 4개를 지우기로 하면서, "다른 논문은 k별 답변 정확도를 재는가"를
원문으로 확인했다. 읽은 방식: 네 편 모두 PDF 전문을 내려받아 해당 절·그림 캡션을 직접 읽었다.
검색 스니펫만으로 인용한 항목은 없다.

**결론: 주 결과표는 넷 다 단일 k다. k별 답변 정확도는 넷 중 셋이 재지만 전부 ablation·부록이다.**
따라서 §0.1 의 "주 표는 단일 운영점" 규칙은 관행과 어긋나지 않는다. 다만 "다른 논문은 k별로
안 잰다"는 서술은 **틀렸다** — 재되 주 표에 올리지 않는 것이다.

## 1. 네 편이 무엇을 싣는가

| 논문 | 검색 지표 | 답변 정확도를 k별로? | 주 표의 k |
|---|---|---|---|
| MultiHiertt (Zhao et al., ACL 2022) | recall 두 점만, 본문 서술로 | **없음** | top-10 고정 |
| TableRAG (Chen et al., NeurIPS 2024) | Table 3 에 recall/precision/F1 | 있음 — Figure 5 | K=5 |
| TableRAG-Huawei (Yu et al., arXiv 2506.10380) | top-30 회수 후 리랭크 | 있음 — 부록 Figure 11 | k=3 |
| Lost in the Middle (Liu et al., 2023, arXiv 2307.03172) | contriever recall@k | 있음 — Figure 11 | 주 실험은 문서 10/20/30 고정 |

### MultiHiertt (우리 §2 의 baseline 원 논문)
검색기는 **본문 한 문장**으로만 보고한다: *"we have 76.4% recall for the top-10 retrieved
facts and 80.8% recall for the top-15 retrieved facts."* 표가 아니라 서술이고, k 도 두 점뿐이다.
주 결과표(Table 4, Dev/Test × EM/F1)는 **top-10 고정**이다(TAPAS 레그만 top-15 를 쓴다고
따로 밝힌다). k 를 바꿔 가며 EM/F1 을 낸 표는 없다.

### TableRAG (우리 §1·§3 의 baseline 원 논문)
- Table 3 이 검색 품질 전용 표다. 캡션: *"Evaluation of retrieval performance. TableRAG shows
  best retrieval quality on all tasks. R: recall, P: precision."* **방법마다 운영점 하나씩**이고
  k 사다리가 아니다.
- 주 표(Table 2)의 조건: *"budget B = 10,000 and the retrieval limit K = 5. For
  RandRowSampling and RowColRetrieval, we increase the retrieval limit to K = 30."*
- k 사다리는 **Figure 5** 에 따로 있다: *"Impact of varying top retrieval results (K).
  Different K values influence both prompt length and accuracy. Each point is labeled with its
  corresponding K value."* 정확도를 세로축, 프롬프트 길이를 가로축에 놓고 점마다 K 를 붙인
  산점도다 — 표가 아니라 **비용-정확도 트레이드오프 그림**이다.

### TableRAG-Huawei (우리 `trag_hetero` 의 원 논문)
주 실험은 k=3 고정. 부록 E.2 에서만 k∈{1,3,5} 로 흔든다: *"While our main experimental setup
fixed k=3, we expanded our investigation on the HeteQA dataset … and varied k across the set
1,3,5."* 결론도 운영점 선택의 근거로만 쓴다 — *"setting k=3 strikes an effective balance."*

### Lost in the Middle — 우리가 §3 에서 본 현상과 같은 것
§5 *"Is More Context Always Better? A Case Study With Open-Domain QA"* 가 **검색 재현율과
리더 정확도를 같은 그림에 겹쳐** 그린다. Figure 11 캡션: *"Retriever recall and model
performance as a function of the number of retrieved documents. Model performance saturates
long before retriever recall, indicating that the models have difficulty making use of the extra
retrieved documents."* k = 5, 10, 20, 30, 40, 50.

본문: *"reader model performance saturates long before retriever performance saturates,
indicating that readers are not effectively using the extra context. Using more than 20 retrieved
documents only marginally improves reader performance (∼1.5% for GPT-3.5-Turbo and ∼1% for
Claude-1.3), while significantly increasing the input context length (and thus latency and cost)."*

**이것이 우리 §3 에서 재현율 .68→.96 인데 답변 정확도가 평평했던 것과 같은 현상이다.**
우리 것이 결함이거나 새 발견이 아니라, 2023년에 개방형 QA 에서 확립된 성질을 계층 헤더 표에서
다시 본 것이다. 이 문단은 **현상 자체를 주장으로 쓰지 말라는 근거**이기도 하다.

같은 절의 마지막 문장이 우리 escalation 노선과 직접 닿는다: *"effective reranking of retrieved
documents … or ranked list truncation (retrieving fewer documents when appropriate;
Arampatzis et al., 2009) may be promising directions."* 우리가 잰 k=1→필요시 k=20 은 뒤쪽,
ranked list truncation 계열이다. 재정렬 계열은 이 리포에서 이미 전부 기각됐다(`CLAUDE.md` §5).

## 2. 이 리포가 할 일

1. **주 표는 단일 운영점(예산 20) 하나 — §4.** 관행과 맞고 §0.1 과도 맞는다.
2. **k 민감도는 검색 지표로만 — §3.** 재현율·정밀도는 이미 query count=991 전수이고 §4 와
   검색 범위가 같다(질의가 속한 표 안). 표본을 §4 의 300 에 맞춰 줄일 이유가 없다.
3. **답변 정확도 k 사다리를 굳이 다시 만든다면** TableRAG Figure 5 처럼 **정확도 × 문맥 길이**
   그림으로 부록에 넣고, 조건은 §4 와 같게(query count=300, 질의가 속한 표 안, Qwen2.5-7B
   4bit) 맞춘다. 지금 지운 열은 query count=100 에 조건 기록이 없어 이 요건을 못 채운다.
4. **"검색 정확도가 .9면 답변도 .9여야 한다"(`CLAUDE.md` §0.2 의 교수 기대)에 대한 답**은
   이 문헌으로 이미 나와 있다 — 리더 정확도는 검색 재현율보다 먼저 포화한다. 우리 쪽 실측
   근거는 §4 의 "검색성공시 정답" 열과 `LITERATURE-2026-09-10-retrieval-vs-answer.md` 다.

## 3. 출처

- Zhao et al. 2022, *MultiHiertt*, ACL 2022, <https://aclanthology.org/2022.acl-long.454/>
- Chen et al. 2024, *TableRAG: Million-Token Table Understanding with Language Models*,
  NeurIPS 2024, <https://arxiv.org/abs/2410.04739>
- Yu et al. 2025, *TableRAG: A Retrieval Augmented Generation Framework for Heterogeneous
  Document Reasoning*, <https://arxiv.org/abs/2506.10380>
- Liu et al. 2023, *Lost in the Middle: How Language Models Use Long Contexts*,
  <https://arxiv.org/abs/2307.03172>
