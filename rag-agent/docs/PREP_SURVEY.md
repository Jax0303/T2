# 검색 전 표 전처리(Table Preprocessing for Retrieval) 연구 조사 보고서

## TL;DR
- **선행연구는 "검색 전 표 전처리"를 (1) 직렬화 포맷 선택, (2) 구조 인코딩(행/열 임베딩·schema/cell 분리), (3) 메타데이터·요약·합성질문 증강의 세 갈래로 다뤄왔으나, "전처리 효과가 표 복잡도(flat vs hierarchical)에 따라 어떻게 달라지는가"를 검색(recall@k) 단위로 통제 비교한 연구는 확인되지 않았다 — 이것이 당신의 명확한 연구 공백이다.**
- **가장 단순하고 강력한 baseline은 BM25 + 표 제목/설명(metadata)이며, TARGET 벤치마크는 "table title이 있을 때 OTT-QA에서 sparse lexical retriever(BM25/TF-IDF)의 recall@10이 0.967·0.963인데 제목을 빼면 0.592·0.583으로 급락한다"고 보고한다 — 즉 전처리의 1차 변수는 정교한 직렬화가 아니라 어떤 메타데이터를 인덱싱에 넣느냐다. 당신의 파일럿(직렬화 포맷이 BGE-small에서 병목이 아니다)은 이 선행연구 흐름과 일치한다.**
- **월요일까지: OpenWikiTable(또는 WTQ 재구성) flat 표에서 raw 직렬화 vs 전처리(제목+스키마설명+합성질문 증강) 검색 성능을 BGE-small로 비교하고 R@1/5/10 + paired bootstrap(seed=42)로 검정한 뒤, 동일 파이프라인을 HiTab(계층형)에 적용해 "전처리 이득 × 복잡도" 곡선을 보이는 단계적 설계를 권장한다.**

## Key Findings

1. **DTR (Herzig et al., NAACL 2021)** 은 표를 평탄화한 뒤 행/열 좌표 임베딩을 추가하는 구조 인코딩 방식. NQ-TABLES test에서 BM25 R@10 40.06 → DTR 76.02, hard negative 추가 시 81.13.
2. **OpenWikiTable (Kweon et al., 2023, ACL Findings)** 은 표를 100단어 청크로 row-wise 분할 후 텍스트 직렬화하여 검색 코퍼스를 구성. 전처리(decontextualization·paraphrase)가 검색 성능을 좌우함을 보임.
3. **TableRAG (Chen et al., NeurIPS 2024)** 은 표 전체를 던지지 않고 schema retrieval과 cell retrieval을 분리, 각 셀을 독립 인코딩하여 백만 토큰 규모 표에 대응.
4. **표 직렬화 포맷 연구(Sui et al., WSDM 2024; Fang et al. survey 2024)** 는 주로 LLM의 표 *이해*(reasoning) 관점이며, *검색* 관점의 포맷 효과는 최근에야(arXiv:2604.24040) 다뤄지기 시작했다.
5. **연구 공백**: 전처리·직렬화 효과를 표 복잡도(flat↔hierarchical)에 따라 검색 recall로 통제 비교한 연구는 없다.

## Details

### 1. 선행연구별 표 전처리/검색 기법

**(A) DTR / Dense Table Retrieval — Herzig, Müller, Krichene, Eisenschlos. "Open Domain Question Answering over Tables via Dense Retrieval." NAACL 2021. arXiv:2103.12011.**
- 전처리/인코딩: dual-encoder(TAPAS 기반 표 인코더). 표를 토큰 시퀀스로 평탄화하되 행/열 좌표를 나타내는 임베딩을 추가하여 셀 위치를 인코딩. DTR-Schema(헤더+제목만), DTR-Text(BERT 초기화) 변형 비교.
- 데이터셋: NQ-TABLES(Natural Questions에서 추출, 코퍼스 169,898 표).
- 핵심 수치(test): BM25 R@1 16.77 / R@10 40.06 / R@50 58.39. DTR R@1 36.24 / R@10 76.02 / R@50 90.25. DTR+hard negatives R@1 42.42 / R@10 81.13 / R@50 92.56. 초록의 요약 문장(verbatim): "improves retrieval results from 72.0 to 81.1 recall@10 and end-to-end QA results from 33.8 to 37.7 exact match, over a BERT based retriever."
- 구조 인코딩의 한계: pre-training 제거(DTR -pt) 시 R@10이 47.80으로 급락 → 표 전용 사전학습이 핵심. 그러나 텍스트 인코더 변형 DTR-Text(R@10 72.00)와 표 전용 TAPAS 기반 DTR(R@10 76.02), DTR-Schema(74.24)의 차이는 수 점 수준으로, "표 전용 구조 인코딩"의 한계를 시사한다(후속 연구 arXiv:2309.10506이 동일 수치 재인용).

**(B) OpenWikiTable — Kweon, Kwon, Cho, Jo, Choi. "Open-WikiTable." Findings of ACL 2023. arXiv:2305.07288.**
- 코퍼스 구성: WikiSQL+WikiTableQuestions를 open-domain용으로 재구성. 표를 row-wise 100단어 청크로 분할 후 재인덱싱(splitted_tables.json). 표를 텍스트로 직렬화하고 표 설명(page title, section title, caption)을 질문에 통합(decontextualization). back-translation으로 paraphrase하여 질문-표 어휘 중복 감소(BLEU 7.28×10⁻²→6.56×10⁻²).
- 통계: 질문 67,023개, 표 코퍼스 24,680개, test 6,602개.
- 검색 수치(validation, BERT-BERT dual encoder): Original 데이터 k=5에서 25.0인데, decontextualized 91.6, paraphrased 89.5. BM25는 original 6.6 → decontextualized 45.5. → **전처리(decontextualization)가 검색 성능을 근본적으로 좌우.**
- end-to-end EM(test): Reader k=5 57.5 / k=10 64.5 / k=20 65.2; Parser k=5 65.2 / k=10 67.1 / k=20 67.9 (Parser가 작은 k에서 특히 우위). 복잡도별(test, k=20): Single-Easy Reader 74.0/Parser 82.8, Single-Hard 62.9/51.2, Multi-Easy 70.5/82.5, Multi-Hard 56.0/58.8.
- 시사점: BERT가 TAPAS보다 모든 단계에서 검색 우위 → "표 검색에 표 전용 모델이 필수는 아니다"(NQ-table에서의 Wang et al. 2022 결론 재확인).

**(C) TableRAG — Chen et al. "TableRAG: Million-Token Table Understanding with Language Models." NeurIPS 2024. arXiv:2410.04739.**
- 기법: query expansion + schema retrieval(컬럼명·타입) + cell retrieval(각 셀 독립 인코딩). 표 전체를 프롬프트에 넣지 않고 질문 관련 schema/cell만 검색.
- 데이터셋: ArcadeQA, BirdQA(저자들이 Arcade·BIRD-SQL에서 구축한 백만토큰 벤치마크), 합성 TabFact, WikiTableQA.
- 핵심 수치: ArcadeQA cell retrieval recall 98.3%(ReadSchema 12.4%, RowColRetrieval 66.5% 대비). WikiTableQA accuracy: TaBERT 52.30, Text-to-SQL 52.90, Binder 56.74, Dater 52.81, TableRAG 57.03. 검색 방식별 정확도(ArcadeQA/BirdQA): BM25 37.7/35.7, Hybrid 46.2/44.5, Embed 49.2/45.5. schema retrieval은 정확도 최대 +9.4%p, cell retrieval은 최대 +11.5%p.
- 시사점: "표를 분해(schema/cell 분리)하여 인덱싱"하는 것이 대규모 표에서 핵심. 단, 이 설정은 *단일 거대 표 내부 검색*이지 *표 코퍼스에서 표 선택*이 아님 — 당신 태스크와 구분 필요.

**(D) TabSD — Wang, Gan, Qi. "TabSD: Large Free-Form Table Question Answering with SQL-Based Table Decomposition." 2025. arXiv:2502.13422.**
- 기법: LLM이 SQL 생성 → 규칙 기반 Table Decomposer가 SQL의 컬럼명·조건·값을 파싱하여 sub-table 추출 → SQL Verifier(LLM)가 자기검증/정제 → sub-table을 Answer Generator에 전달.
- 데이터셋: WikiTableQuestions, HybridQA + 신규 SLQA, SEQA.
- 시사점: "검색"이라기보다 단일 표 내 SQL 기반 분해. 당신의 "검색 전 전처리" 프레임에서는 *전처리 기법*(질문→구조화 쿼리→관련 영역 선택)의 한 사례로 인용 가능. 정확한 베이스라인 대비 향상폭 수치는 본 조사에서 전체 표 미확보 — **확인 불가**.

**(E) DTR 이후 table-specific retriever들**
- **Syntax/Structure-aware Dense Retrieval (Jin et al., IJCNLP-AACL 2023, arXiv:2309.10506)** 및 **UTP**: 질문의 구문(syntax) 표현과 표의 구조적 헤더/값 표현을 분리 후 aggregator로 매칭. NQ-TABLES에서 DTR R@1 36.24 / R@10 76.02 대비 Ours(im) 47.03 / 84.76. hard negative 학습 시 DTR 42.42/81.13, UTP 50.39/85.40, Tri-encoder R@10 86.4, Ours(im) 54.12/90.41.
- **Tri-encoder (Kostić et al., 2021, arXiv:2108.04049)**: 표·텍스트 멀티모달 검색. 질문-표 어휘 중복(lexical overlap)이 높으면 BM25가, 낮으면 dense가 우위라는 분석 제시.
- **QGpT (Liang, Chang, Fan, TRL@ACL 2025, arXiv:2508.06168)**: 표를 그대로 임베딩하는 대신 표 일부(partial table)에서 LLM으로 합성 질문을 생성해 partial table과 함께 임베딩 → 질문-표 의미 정렬 개선. BGE-m3 dense 평균(MimoTable/OTTQA/FetaQA/E2E-WTQ): R@1 42.01→44.39(+2.38), R@5 63.34→66.84(+3.50), R@10 71.87→74.49(+2.62). Jina-ColBERT-v2: R@1 +2.30, R@5 +3.08, R@10 +2.64. 복잡한 MimoTable에서 이득 최대(BGE-m3 R@1 +6.39, R@5 +5.10, R@10 +5.53). 논문은 단일 글로벌 향상치를 명시하지 않으므로 위 수치는 Table 5의 "Avg" 열 기준. → **"표 표현(table representation) 자체를 질문 친화적으로 전처리"하는 최신 방향이며, 당신 아이디어와 가장 가까운 선행연구.** 단 여기서 "table complexity"는 표 *길이/크기*를 의미하며 계층 헤더가 아님.
- **Pneuma (2025, arXiv:2504.09207)**: LLM으로 schema summary + row summary를 생성해 인덱싱. BM25·LlamaIndex RAG·Solo 대비 relevant table hit rate 최대 +22.95%.

**(F) 직렬화 포맷 비교 연구**
- **Sui et al. "Table Meets LLM: Can Large Language Models Understand Structured Table Data?" WSDM 2024. arXiv:2305.13062**: CSV/JSON/XML/Markdown/HTML/XLSX 비교. GPT-3.5/4에서 마크업 언어, 특히 HTML이 구분자 기반("NL+Sep") 대비 +6.76%p 우위(verbatim: "using markup languages, specifically HTML, outperforms 'NL+Sep' with a 6.76% improvement"). 7개 task 최고 정확도는 HTML+format explanation+role prompt 조합에서 65.43%(verbatim). → 이는 *이해(reasoning)* 태스크이며 *검색*이 아님.
- **Fang et al. "Large Language Models on Tabular Data: A Survey." TMLR 2024. arXiv:2402.17944**: 직렬화를 text-based template / embedding-based 등으로 분류. Singha et al.(2023)은 DFLoader/JSON이 fact-finding에 유리, Sui et al.은 HTML/XML이 GPT에 유리(토큰 비용 증가 trade-off)하다고 보고.
- **"Improving Robustness of Tabular Retrieval via Representational Stability." 2026. arXiv:2604.24040**: *검색* 관점에서 csv/tsv/html/markdown/ddl이 같은 표라도 다른 임베딩·검색 결과를 낳음을 WTQ/WikiSQL/NQ-Tables 3개 벤치마크 × 4개 임베딩 모델(BGE-M3, MPNet, ReasonIR, SPLADE)로 보임. NQ-Tables에서 단일 retriever의 포맷 간 Recall@1 차이가 최대 0.26. 해결책으로 여러 직렬화 임베딩의 centroid 평균을 제안. → **단, 세 벤치마크 모두 flat 표이며, 직렬화 *포맷*을 변수로 둘 뿐 표 *복잡도*를 변수로 두지 않음.** 저자들은 adapter 이득을 "보편적 향상이 아닌 robustness 해석"으로 한정.

**(G) 메타데이터/요약/스키마 증강을 인덱싱에 활용한 연구**
- **TARGET 벤치마크 (Ji et al., 2025, arXiv:2505.11545)**: 표 제목(metadata)이 검색에 결정적. verbatim: "The strong performance of the sparse lexical retrievers with table title on the OTTQA dataset (recall@10 of 0.967 and 0.963) can be attributed to the high correspondence between Wikipedia table titles and the questions" — 제목 제외 시 BM25 0.592, TF-IDF 0.583으로 급락. dense table embedding(stella_en_400M_v5) full table 0.963 vs 컬럼명만 0.658. 결론: "descriptive metadata (e.g. table summaries or titles) can be key for lexical retrievers."
- **Pneuma, RASL(arXiv:2507.23104), MAG-SQL(arXiv:2408.07930)**: LLM으로 컬럼 설명·표 요약을 생성해 인덱싱/스키마 링킹에 활용.

### 2. WikiTableQuestions 및 flat→계층 데이터셋 현황

| 데이터셋 | 표 수 | 질문 수 | 복잡도 | 라이선스/다운로드 |
|---|---|---|---|---|
| WikiTableQuestions (Pasupat & Liang, ACL 2015, P15-1142) | 2,108 | 22,033 | flat (각 표 최소 8행·5열, 첫 행만 헤더; test 표는 train과 비공유) | CC BY 4.0, GitHub ppasupat/WikiTableQuestions, HF stanfordnlp/wikitablequestions |
| WikiSQL (Zhong et al. 2017) | 24,241 | 80,654 | flat, SQL 라벨 | 공개 |
| OpenWikiTable (Kweon et al. 2023) | 24,680 (코퍼스) | 67,023 | flat, 검색용 재구성, Text+SQL | CC BY-SA, GitHub sean0042/Open_WikiTable |
| TabFact | — | — | flat, fact verification | 공개 |
| HiTab (Cheng et al., ACL 2022, arXiv:2108.06712) | 3,597 | 10,672 | **계층형(공식 README verbatim: "98.1% of the tables in HiTab are with hierarchies"), StatCan/NSF/ToTTo 출처** | free & open, GitHub microsoft/HiTab |
| AIT-QA (Katsis et al., NAACL 2022, arXiv:2106.12944) | 116 | 515 | **계층형 헤더, 항공업 SEC 10-K(2017–2019)** | 공개 |

- 단계적 난이도 경로: **WTQ/OpenWikiTable(flat) → TabFact(flat, 검증) → HiTab(계층, 대규모) → AIT-QA(계층, 도메인 특화·소규모)**. AIT-QA는 표 116개로 작아 빠른 파일럿에 적합하나 통계검정력 한계.

### 3. 연구 공백 (Research Gap)

- **복잡도×전처리 통제 비교 부재**: 전처리/직렬화 효과를 표 복잡도(flat vs hierarchical)에 따라 *검색 recall*로 통제 비교한 연구는 확인되지 않음. arXiv:2604.24040은 포맷을, TARGET은 표 크기·메타데이터를 변수로 두지만 둘 다 주로 flat 표이며 계층 구조를 독립변수로 stratify하지 않음. Granite Embedding R2 보고서(arXiv:2508.21085)는 flat(OpenWikiTables·NQTables·OTT-QA)·hierarchical(MultiHierTT·AIT-QA) 데이터셋을 나란히 평가하나 이는 *모델 리더보드*이지 전처리 통제 실험이 아님.
- **당신 파일럿과의 정합성**: "BGE-small에서 직렬화 포맷이 검색 병목이 아니다"는 결과는 (a) OpenWikiTable이 BERT(텍스트)≈TAPAS(표 전용)임을 보인 점, (b) TARGET이 메타데이터(제목)가 포맷보다 훨씬 큰 변수임을 보인 점과 일치한다. 단 arXiv:2604.24040은 *다른 임베딩 모델*에서는 포맷 민감도가 존재(Recall@1 최대 0.26 차이)한다고 보고 → 당신 결과는 "BGE-small + flat 표"라는 특정 조건의 결론으로 한정해 방어해야 함.
- **이미 해결된 것**: "표 요약/설명 붙여 인덱싱"은 Pneuma·RASL·TARGET·QGpT에서 이미 다뤄짐. 단순히 "요약 붙이기"를 새 기여로 제안하면 안 됨. 차별점은 **(1) 복잡도 축을 따라 어떤 전처리가 언제 효과적인지의 체계적 매핑, (2) 합성질문/스키마경로 평탄화 같은 전처리가 계층 표에서 flat 표보다 더/덜 효과적인지**에 두어야 한다.

### 4. 월요일까지 실행 가능한 독립 실험 프로토콜

**환경**: WSL2, RTX 3060 Ti 8GB, BGE-small(로컬), FAISS, 무료 API만.

**데이터**: OpenWikiTable(코퍼스 24,680, test 6,602) 또는 WTQ를 검색 태스크로 재구성. 빠른 반복을 위해 코퍼스 일부(예: test 질문 1,000개 + 전체 코퍼스) 샘플(선행 OpenTab도 budget 한계로 2,000 샘플 사용).

**조건(독립변수=전처리)**:
- C0 (raw): 표를 그대로 직렬화(예: markdown/CSV)하여 인덱싱.
- C1 (+metadata): 표 제목·섹션·캡션을 직렬화에 prepend.
- C2 (+schema description): 컬럼명·타입·간단 설명을 추가.
- C3 (+synthetic question, QGpT류): partial table에서 합성 질문 생성 후 함께 임베딩(무료/로컬 LLM 사용).

**측정**: R@1/R@5/R@10(주지표), 가능하면 retrieved top-k를 무료 LLM reader/parser에 넣어 end-to-end answer accuracy(EM). OpenWikiTable의 Reader/Parser EM(test k=10: 64.5/67.1)을 참고 상한선으로 사용.

**통계**: paired bootstrap(질문 단위 재표집, 10,000회), 95% CI, seed=42 고정. 모든 임베딩/인덱싱 seed 고정으로 재현성 확보.

**확장 경로**: 동일 C0–C3 파이프라인을 HiTab(계층)에 적용. 계층 헤더 평탄화 방식(root-to-leaf 경로 concatenation, HiTab·API-Assisted Code Generation arXiv:2310.14687에서 사용)을 C2의 계층형 변형(C2-hier)으로 추가. 결과를 "전처리 이득 × 복잡도(flat/hierarchical)" 2×N 표로 제시 → 이것이 공백을 정면으로 채움.

**교수 질문 방어 논리**:
- "다른 논문은?" → DTR은 행/열 임베딩(구조 인코딩), OpenWikiTable은 decontextualization, TableRAG은 schema/cell 분해, QGpT는 합성질문 증강, TARGET은 메타데이터 효과. 대부분 flat 표 또는 단일 거대 표에 집중.
- "너는 어떻게 다른가?" → 나는 동일 전처리 집합을 **flat→hierarchical 복잡도 축에서 통제 비교**하여, "전처리 이득이 표 복잡도에 따라 어떻게 달라지는가"를 검색 recall로 정량화한다. 또한 파일럿에서 BGE-small+flat에서는 포맷이 병목이 아님을 확인했으므로, 변수를 포맷이 아닌 *정보 증강(metadata/schema/synthetic Q)*에 둔다.

## Recommendations

1. **즉시(1일차)**: OpenWikiTable 다운로드, BGE-small+FAISS로 C0(raw) vs C1(+제목) 검색 R@1/5/10 측정. C1이 크게 이기면(TARGET·OpenWikiTable 선행과 일치) baseline 확립. 임계: C1−C0가 paired bootstrap 95% CI에서 0을 포함하지 않으면 유의.
2. **2일차**: C2(+schema desc), C3(+synthetic Q) 추가. 무료/로컬 LLM으로 합성 질문 생성(QGpT 방식). flat 표에서 전처리 이득 곡선 확정.
3. **3일차**: 동일 파이프라인을 HiTab 100~500표 subset에 적용, 계층 헤더 평탄화(C2-hier) 포함. "복잡도×전처리" 표 작성. 시간 부족 시 AIT-QA(116표)로 축소.
4. **벤치마크/임계값**: 전처리 이득이 flat에서는 작지만 hierarchical에서 유의하게 크면(또는 반대면) 그 자체가 발표 가능한 발견. C3가 C1 대비 추가 이득이 없으면 "메타데이터가 지배적"이라는 TARGET 결론을 재확인하는 negative result로 보고.
5. **방어 자료화**: 각 조건의 정확한 선행연구 출처(위 표)와 수치를 슬라이드 1장에 정리.

## Caveats
- TableRAG의 schema/cell 분해는 *단일 거대 표 내부* 검색이며, 표 코퍼스에서 표를 고르는 당신 태스크와 다름 — 직접 비교 시 주의.
- TabSD의 정확한 성능 수치(베이스라인 대비 향상폭)는 본 조사에서 전체 표를 확보하지 못함 — 확인 불가.
- arXiv:2604.24040, 2603.07950, 2601.13111 등은 2026년 ID의 최신 프리프린트로 peer-review 전일 수 있음 — 인용 시 조건부로 다룰 것.
- OpenWikiTable·WTQ·NQ-Tables는 모두 flat 표 → flat 영역 선행연구는 풍부하나, hierarchical 검색 전처리 선행연구는 희소(공백의 근거이자, 동시에 baseline 부재라는 리스크).
- QGpT의 "table complexity"는 표 길이/크기(MimoTable)를 의미하며 계층 헤더가 아니므로, 당신의 "계층 복잡도" 공백 프레임과 혼동하지 말 것.
- end-to-end answer accuracy는 무료 LLM 품질에 좌우되므로, 검색 지표(R@k)를 1차 결론으로, answer accuracy를 보조 지표로 보고할 것.