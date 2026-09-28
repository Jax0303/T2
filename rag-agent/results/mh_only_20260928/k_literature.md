# 선행 RAG 논문의 상위 k 선택 근거 (2026-09-28 원문 대조, arXiv/ar5iv HTML 본문. ToR-RAG 는 KCI 초록만)

| 논문 | 주 결과 k | 정한 방법 | 원문 (위치) | k 변화 결과 |
|---|---|---|---|---|
| Lewis+ 2020 RAG (2005.11401) | 학습 5·10, 시험 15/50(NQ류), 10 | dev 튜닝 | "We consider k∈{5,10} for training and set k for test time using dev data." (§3) | RAG-Sequence 계속 상승, RAG-Token 10 정점 (Fig. 3) |
| Izacard & Grave 2021 FiD (2007.01282) | 100 | 근거 없음(곡선은 사후) | "For both training and testing, we retrieve 100 passages (unless said otherwise)" (§4) | 10→100 계속 상승 (Fig. 3) |
| Liu+ 2023 Lost in the Middle (2307.03172) | 변수 10/20/30, §5 최대 50 | 연구 변수 | "Using more than 20 retrieved documents only marginally improves reader performance" (§5, Fig. 11) | 포화 |
| Xu+ 2023 Retrieval meets Long Context (2310.03025) | top-5 | 입력 길이 + 절제 | "top-5 chunks can all fit into 4k sequence length" (§3.4); "best averaged results are obtained either from top 5 or top 10" (§4.4, Table 5) | 올랐다 내려감 |
| Jin+ 2024 Long-Context LLMs Meet RAG (2410.05983) | 모델 한도까지, 미세조정 40 | 입력 한도 | "Increasing the number of retrieved passages initially improves performance but then leads to a decline." (§3.1, Fig. 1) | 올랐다 내려감(강한 검색기) |
| Chen+ 2024 TableRAG (2410.04739) | K=5, B=10,000 | 근거 없음 | "we set the cell encoding budget B=10,000 and the retrieval limit K=5" (§4.3); "increasing the number K ... does not consistently improve performance" (§4.8, Fig. 5) | 일정하지 않음 |
| Yu+ 2025 TableRAG 이종 문서 (2506.10380) | 30 회수 → 재정렬 3 | 고정 후 민감도 확인(별도 dev 없음) | "setting k=3 strikes an effective balance ... while maintaining computational efficiency" (App. E.2, Fig. 11) | k∈{1,3,5} |
| Ram+ 2023 In-Context RALM (2302.00083) | 2 | dev 분석 | "Motivated by our previous findings, we used two retrieved documents." (§7) | 2개에서 대부분 이득 (dev, Fig. 8) |
| Cuconasu+ 2024 Power of Noise (2401.14887) | 입력 한도 내 변수 | 입력 길이 | "retrieving between 3 and 5 documents is the most effective choice" (§5.5) | 3~5 최적 |
| Herzig+ 2021 DTR (2103.12011) | K=10 표 | 근거 없음 | "In this work we set K=10" (§3) | 없음 |
| 유희경·문남미 2025 ToR-RAG (KCI 30(10)) | top-k=5 | 미확인(본문 접근 불가) | 초록 "MMR 기반 전략(λ=0.75, top-k=5)" | 미확인 |
