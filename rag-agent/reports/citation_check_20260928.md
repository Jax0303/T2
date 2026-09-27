# 인용 원문 대조 (2단계, 2026-09-28)

- 대상 원고: 커밋 2dd7e9e 의 `thesis/src/*.md` — 초록 제외, 부록 포함. **이 단계에서 원고는 수정하지 않았다.**
- 방법: 문헌을 6개 묶음(A~F)으로 나눠 참고문헌 목록이 가리키는 판본(학회 논문집 PDF 우선, arXiv 는 버전 기록)을 직접 열고 본문·표·부록까지 대조했다. 코드·README·모델 카드에 관한 서술은 해당 공식 저장소·페이지를 열었다. 묶음 밖의 두 행(G)은 직접 판정했다.
- 원문을 열지 못한 경우는 "확인 불가"로 두었고, 기억이나 2차 자료로 판정하지 않았다. 내려받은 원문 PDF·추출 텍스트는 저작권 때문에 저장소에 넣지 않았다(4절의 URL·버전으로 다시 열 수 있다).
- 판정: 정확 / 부정확(내용이 다르거나 과장·축소·조건 누락) / 원문에 없음 / 확인 불가. 한 문장에 주장이 여럿이면 행을 나눴다.
- 연 판본 칸의 `[X:Y]` 기호는 4절 묶음 X 의 판본 목록 Y 를 가리킨다.

## 1. 판정별 개수

| 묶음 | 정확 | 부정확 | 원문에 없음 | 확인 불가 | 합계 |
|---|---:|---:|---:|---:|---:|
| A. HiTab(Cheng et al., 2022)·표 QA 서베이(Zhou et al., 2026) | 19 | 5 | 0 | 0 | 24 |
| B. MultiHiertt(Zhao et al., 2022)·공식 저장소 | 19 | 6 | 0 | 0 | 25 |
| C. TableRAG(Chen et al., 2024)·RowCol·RandRow 재구현 | 23 | 13 | 0 | 0 | 36 |
| D. TableRAG(Yu et al., 2025)·MixRAG·Cao·ITR·TAP4LLM | 35 | 10 | 1 | 0 | 46 |
| E. RAG·방해 정보·평가 관행(Lewis, Cuconasu, CABINET, Liu, Deng, Wang, TableLlama, RealHiTBench, TableBench) | 16 | 8 | 1 | 0 | 25 |
| F. 임베딩·BM25·리더 모델·통계 검정·도구(C-Pack, BM25, Qwen2.5, Qwen3, Holm, McNemar, bge-reranker, transformers) | 17 | 0 | 1 | 4 | 22 |
| G. 묶음 밖 직접 판정(서론 예시·인용 없는 일반 서술) | 1 | 0 | 1 | 0 | 2 |
| **합계** | **130** | **42** | **4** | **4** | **180** |

## 2. 정확이 아닌 항목

### 부정확 (42)

- **A6** `02_related.md:5` — "각 질문에는 답과 함께 답을 계산하는 데 참조한 셀(reference cells)과 계산식이 주석되어 있고"
- **A9** `02_related.md:5` — "본 연구는 이 계층 트리를 그대로 써서 셀의 머리글 경로를 만들고"
- **A15** `03_method.md:37` — "데이터셋이 준 표의 절 제목 앞에"
- **A22** `04_setup.md:55` — "…공식 채점 규칙을 이식한 채점기로"
- **A24** `08_refs.md:26` — Zhou, W., Ma, B., Friedrich, A., & Mesgar, M. (2026). … In *Proceedings of ACL 2026*. arXiv:2510.09671.
- **B2** `02_related.md:7` — "재무 보고서 문서에서 여러 개의 계층형 표와 본문 문단을 함께 읽고 수치 추론을 해야 하는 질문을 모았다"
- **B3** `02_related.md:7` — "각 질문에는 … 표 셀(table evidence)과 본문 문장(text evidence), 그리고 계산 과정을 나타내는 프로그램이 주석되어 있다"
- **B8** `02_related.md:7` — "주 결과표는 상위 10개 고정 조건이다"
- **B9** `03_method.md:21` — "표가 HTML로만 주어지고"
- **B10** `03_method.md:21` — "머리글 계층 정보가 없다"
- **B19** `04_setup.md:55` — "공식 규칙상 정답이 음수인 산술 문항은 텍스트 답으로 맞힐 수 없는데"
- **C3** `01_intro.md:28` — "RowCol(표의 행과 열을 각각 한 단위로 색인해 검색)"
- **C7** `02_related.md:13` — "스키마(열 이름과 자료형·값 범위 요약)와 셀(열 이름과 셀 값의 쌍)을 따로 검색해"
- **C8** `02_related.md:13` — "주 실험은 토큰 예산 B=10,000"
- **C9** `02_related.md:13` — "검색 개수 K=5로 설정했고"
- **C12** `02_related.md:13` — "TableRAG의 셀·스키마 텍스트 표현을 공식 코드와 대조해 이식하고(leaf, path 두 변형)"
- **C16** `04_setup.md:29` — "비교군의 텍스트 표현은 원 논문의 공개 코드와 대조해 이식했다."
- **C17** `04_setup.md:29` — "모든 비교군은 본 방법과 같은 … 하이브리드 검색기(α=0.7) … 를 쓴다"
- **C23** `04_setup.md:43` — "(2) 스키마 문서(열 이름과 자료형, 숫자 열은 최솟값·최댓값, 범주형 열은 자주 나오는 값 3개)는 만들지만, 원 논문처럼 별도 검색 경로로 찾지 않고 셀 문서와 한 색인에 섞어 같은 검색기로 찾는다."
- **C25** `04_setup.md:43` — "숫자 열은 개별 값을 색인하지 않고 요약 문서 하나로 접는다."
- **C28** `04_setup.md:47` — "두 데이터셋 모두 원 논문처럼 행(열)의 값을 \"\|\"로 이은 텍스트로 색인한다."
- **C30** `05_results.md:50` — "leaf와 path 표현은 (열 이름, 셀 값) 쌍만 담고 행 머리글을 담지 않는다. … 같은 열의 셀들이 열 이름과 값만으로는 구별되지 않는다." (낮은 이유로 제시)
- **C34** `06_discussion.md:7` — "(가장 큰 차이가 나는 비교군은 …) RandRow다. 이 표현들은 … 행 단위로만 색인해 상위 머리글이 빠진다(RandRow)."
- **C36** `08_refs.md:4` — "Chen, S.-A., Miculicich, L., Eisenschlos, J. M., et al. (2024). TableRAG: … In *Advances in Neural Information Processing Systems (NeurIPS 2024)*. arXiv:2410.04739."
- **D2** `01_intro.md:15` — "이 구성은 … Lin et al.(2023)의 정의와 같다" (상위 20셀이 표 대신 리더 입력)
- **D8** `02_related.md:11` — "본 연구의 과제 설정(표 안에서 셀을 골라 리더 입력을 대신함)은 이 정의와 같다"
- **D11** `02_related.md:11` — "표 안 셀을 고르는 단계는 논문마다 이름이 다르다" (ITR·TAP4LLM에 적용)
- **D14** `02_related.md:15` — "글자 수 기준으로 잘라(청크 1,000, 겹침 200)"
- **D16** `02_related.md:15` — "본 연구는 이 중 청크 표현만 재현했다"
- **D18** `02_related.md:17` — "HiTab과 MultiHiertt에서 가져온 표를 본문과 함께 문서로 재구성한 코퍼스(2,178문서)"
- **D26** `02_related.md:43` — ITR 검색 단위 "부분 표"
- **D27** `02_related.md:43` — ITR 검색 품질의 보고 "—"(확인하지 않은 칸)
- **D43** `08_refs.md:23` — Zhang, C., Chen, Q., & Zhang, M. (2026). Mixture-of-RAG … *Proceedings of the 32nd ACM SIGKDD … (KDD 2026)*, pp. 1880–1891.
- **D45** `08_refs.md:10` — Lin, W., Blloshmi, R., Byrne, B., de Gispert, A., & Iglesias, G. (2023). … *Proceedings of the 61st Annual Meeting of the ACL (ACL 2023)*, pp. 9909–9926.
- **E3** `02_related.md:23` — "질문과 관련은 있으나 답이 아닌 정보가 섞이면 리더의 정확도가 떨어진다는 것은 확립된 현상이다"
- **E4** `02_related.md:23` — "정답 문서가 있는 문맥에 주제상 가까운 방해 문서를 하나씩 더할수록 정확도가 단조롭게 떨어지는 것을 보였다"
- **E5** `02_related.md:23` — CABINET이 "표 전체 중 작은 부분만 질문과 관련이 있고, 나머지는 방해 정보로서 잡음으로 작용한다"고 보고함
- **E11** `02_related.md:31` — "(500건에) 부트스트랩 신뢰구간을 붙였으며"
- **E12** `02_related.md:31` — "TableLlama(Zhang et al., 2024) … 도 500건을 썼다"
- **E13** `02_related.md:31` — "RealHiTBench(Wu et al., 2025a)도 500건을 썼다"
- **E15** `02_related.md:33` — "유형별 값과 개수를 함께 싣는 것이 관행이다"
- **E16** `02_related.md:33` — "TableBench(Wu et al., 2025b)는 유형별 정확도를 유형 크기로 가중 평균한다"

### 원문에 없음 (4)

- **D34** `04_setup.md:35` — "고정 청크 \| 업계 관행" (1,000자, 제목·머리글 블록을 청크마다 반복)
- **E2** `01_intro.md:5` — "대규모 언어 모델(LLM)은 … RAG 방식으로 널리 쓰인다"
- **F15** `04_setup.md:75` — "이 방식은 한 번에 한 건씩 생성하는 방식과 부동소수 연산 순서가 달라 출력 문자열이 바이트 단위로 같지 않다"
- **G2** `01_intro.md:9` — "흔한 방식은 표를 마크다운이나 쉼표 구분 텍스트로 펴는 것이다"

### 확인 불가 (4)

- **F10** `01_intro.md:28` — "Holm(1979) 보정 p=.0040" (Holm 인용 부분만)
- **F11** `04_setup.md:69` — "정확 McNemar 검정(McNemar, 1947 …)" — 검정의 출처
- **F12** `04_setup.md:69` — "(McNemar, 1947; 불일치 쌍에 대한 양측 이항 검정)" — 정확 형태를 1947 논문에 귀속
- **F13** `04_setup.md:69` — "여러 비교를 함께 판정하는 곳은 비교 묶음 단위 Holm 보정값을 함께 적는다"

## 3. 전체 대조 표

| 번호 | 위치(파일:줄) | 원고 문장 | 인용 문헌 | 연 판본 | 원문 근거 위치(절·쪽·표) | 원문 근거 요지 | 판정 | 수정안 |
|---|---|---|---|---|---|---|---|---|
| A1 | 01_intro.md:15 | "closed-domain 설정 … 질문과 짝지어진 표(HiTab) … 가 직접 주어진다(Zhou et al., 2026 …)" | Zhou et al. (2026) | [A:Z] v2 | 그림 1 설명 p.1/12161; 그림 2 분류도 p.2/12162; §2 "Domains." p.3/12163; 부록 표 2 p.27/12187 | "inputs need to be retrieved from a data pool (open-domain) or directly given (closed-domain)"; 그림 2 "Closed … Cheng et al. (2022)"; 표 2 HiTab 행 Open domain "✗" | 정확 | — |
| A2 | 01_intro.md:15 | "… 또는 문서(MultiHiertt)가 직접 주어진다(Zhou et al., 2026 …)" | Zhou et al. (2026) | [A:Z] v2 | 부록 표 2 p.28/12189 | MultiHiertt (Zhao et al., 2022) 행: Open domain "✗", Single(표 1개) "✗", 추가 입력 text "✓" — 여러 표+문단 입력이 주어지는 closed-domain 으로 분류 | 정확 | — |
| A3 | 01_intro.md:15 | "질문과 짝지어진 표(HiTab) … 가 직접 주어진다(… Cheng et al., 2022)" | Cheng et al. (2022) | [A:H] | §3 Problem Statement, p.1097 | "Our dataset D = {(xi, ti, yi)} … is a set of N question-table-answer triples." | 정확 | — |
| A4 | 02_related.md:5 | "계층형 머리글을 가진 표에 대한 질의응답·문장 생성 데이터셋 HiTab을 공개했다" | Cheng et al. (2022) | [A:H] | 초록·§1, p.1094 | "We present a new dataset, HiTab, to study question answering (QA) and natural language generation (NLG) over hierarchical tables"; 계층형 = "header exhibits a multi-level structure" | 정확 | — |
| A5 | 02_related.md:5 | 과제 정의 "계층형 표 t와 자연어 질문 x가 주어졌을 때 답 y를 출력하는 것" | Cheng et al. (2022) | [A:H] | §3 Problem Statement, p.1097 | "given a hierarchical table t and a question x in natural language, output answer y." | 정확 | — |
| A6 | 02_related.md:5 | "각 질문에는 답과 함께 답을 계산하는 데 참조한 셀(reference cells)과 계산식이 주석되어 있고" | Cheng et al. (2022) | [A:H], [A:R] | [A:H] §2.3 각주 2, p.1096; [A:R] README 81–90행, data/test_samples.jsonl | [A:H] "For samples with XLOOKUP or IF formulas, we didn't explicitly provide the formulas … provide the answer cell reference(s)". [A:R] test 1,584건 중 argmax·pair-argmax 류는 answer_formulas 가 답 셀 참조("=A8")뿐, 11건은 "=" 없는 문자열(예 "Transportation"), 9건은 reference_cells_map 이 비어 있음 | 부정확 | "각 질문에는 답과 함께 답을 이루거나 계산하는 데 쓰인 셀의 참조(reference_cells_map)와 스프레드시트 계산식(answer_formulas)이 주석되어 있다. 단 최상급·비교(XLOOKUP·IF) 유형은 계산식 대신 답 셀 참조만 주어진다(Cheng et al., 2022, 각주 2)." |
| A7 | 02_related.md:5 | "계산 종류를 나타내는 집계(aggregation) 라벨이 붙어 있다" | Cheng et al. (2022) | [A:R] | README 51–53, 90행 | "`aggregation` is the aggregation(s) to derive the answer." test 라벨 예: none 1,133, pair-argmax 93, div 76 … | 정확 | — |
| A8 | 02_related.md:5 | "표마다 행·열 머리글의 계층 트리가 제공된다" | Cheng et al. (2022) | [A:H], [A:R] | [A:H] §2.6, p.1097; [A:R] README 150행, tables.zip | [A:H] "we adapt heuristics … to extract top and left hierarchical trees"; [A:R] "`top_root` and `left_root` are the parsed tree hierarchies"; raw 표 3,597개 전부 두 트리 있음 | 정확 | — |
| A9 | 02_related.md:5 | "본 연구는 이 계층 트리를 그대로 써서 셀의 머리글 경로를 만들고" | 인용 없음(데이터셋 [A:R]) | [A:R] | tables.zip(raw), README 192행 | test 표 538개 중 192개에서 데이터 열 226개가 top 트리 잎에 없고, 226개 모두 병합 머리글 영역 안. 원고 03_method.md:19도 "순서 정보로 보완"이라 적음 → "그대로"와 충돌 | 부정확 | "본 연구는 이 계층 트리를 따라 셀의 머리글 경로를 만들고(병합 머리글 때문에 트리에서 빠지는 열은 순서 정보로 보완, 3장)" |
| A10 | 02_related.md:29 | "입력 표가 직접 주어지는 설정을 closed-domain 설정으로 분류하고" | Zhou et al. (2026) | [A:Z] v2 | §2 "Domains.", p.3/12163; 그림 1 설명 p.1/12161 | "open-domain … and closed-domain … settings, depending on whether the relevant inputs are provided"; "closed-domain setting where target inputs are given directly" | 정확 | — |
| A11 | 02_related.md:29 | "HiTab을 그 예로 든다" | Zhou et al. (2026) | [A:Z] v2 | 그림 2 p.2/12162; §2 p.3/12163; 표 2 p.27/12187 | "closed-domain (Pasupat and Liang, 2015; Cheng et al., 2022; …)"; 표 2 HiTab Open domain "✗" | 정확 | — |
| A12 | 03_method.md:19 | "데이터셋이 표마다 행·열 머리글의 계층 트리를 준다" | 인용 없음(Cheng et al., 2022 데이터) | [A:H], [A:R] | [A:H] §2.6 p.1097, §8 p.1102; [A:R] README 150행 | [A:H] "recognized table hierarchies" 포함 배포; [A:R] top_root·left_root, raw 3,597/3,597 | 정확 | — |
| A13 | 03_method.md:19 | "병합 머리글 때문에 대응이 빠지는 열" | 인용 없음(데이터셋 [A:R]) | [A:R] | tables.zip(raw) 재계산; README 192행 | test 192/538 표, 226열이 top 트리 잎에 없음, 226열 모두 merged_regions 머리글 영역 안. README: 병합 셀은 "only its core cell … will have content" | 정확 | — |
| A14 | 03_method.md:37 | "HiTab의 표 제목" (데이터셋이 준 제목) | 인용 없음(데이터셋 [A:R]) | [A:R] | tables.zip raw/*.json `title` 필드 | raw 3,597개 전부 `title` 있음(README 표 형식 설명에는 없음) | 정확 | — |
| A15 | 03_method.md:37 | "데이터셋이 준 표의 절 제목 앞에" | 인용 없음(데이터셋 [A:R], [A:T]) | [A:R], [A:T] | [A:R] raw/*.json `title`; [A:T] totto_train_data.jsonl `table_section_title`; [A:H] §2.1 p.1095 | ToTTo 출신 1,851표는 `title` = ToTTo 절 제목(소문자 일치 1,835, 나머지 16은 괄호·발음부호 제거 또는 'none'). StatCan·NSF 1,746표는 보고서 표 제목(예 "rate of homicides, by province and territory, 1987 to 2017"). test 538표 중 260표가 StatCan·NSF | 부정확 | "HiTab에서는 데이터셋이 준 표 제목(`title`: StatCan·NSF 표는 보고서의 표 제목, ToTTo 출신 표는 위키백과 절 제목) 앞에, …" |
| A16 | 03_method.md:37 | "표의 원 출처인 위키백과 페이지의 제목(ToTTo 페이지 제목)이 있으면 그것을 붙인다" | 인용 없음(Cheng et al., 2022 데이터; ToTTo) | [A:H], [A:R], [A:T] | [A:H] §2.1 p.1095; [A:T] `table_page_title` | [A:H] "we include random 1,851 tables (50% of our dataset) from ToTTo" (위키백과); HiTab 표에는 페이지 제목 필드 없음, [A:T] 대응 줄의 table_page_title(예 "Charles Sims (American football)")이 위키백과 페이지 제목 | 정확 | — |
| A17 | 04_setup.md:12 | "질의 수 1,584" (HiTab) | 인용 없음(데이터셋 [A:R]) | [A:R], [A:H] | [A:R] data/test_samples.jsonl; [A:H] §3.2.5 p.1098, 표 3 p.1097 | [A:R] 1,584줄(train 7,417, dev 1,671, 합 10,672). [A:H]에는 분할별 질문 수가 없고 "We split 3,597 tables into train (70%), dev (15%) and test (15%)"와 총 10,672만 있음 | 정확 | — |
| A18 | 04_setup.md:19 | "test 분할 1,584 질의를 쓴다" | 인용 없음(데이터셋 [A:R]) | [A:R] | data/test_samples.jsonl | 1,584줄. 2025-10 품질 점검판 test_samples_qualitycheck.jsonl(231건 교체)도 1,584줄이며, 원고가 쓴 파일은 원래 test_samples.jsonl | 정확 | — |
| A19 | 04_setup.md:19 | "HiTab test 분할의 표는 538개" | 인용 없음(데이터셋 [A:R]) | [A:R], [A:H] | [A:R] test_samples.jsonl table_id; [A:H] §3.2.5 p.1098 | 서로 다른 table_id 538개(statcan 238, totto 278, nsf 22). 3,597×15% ≈ 540과 부합 | 정확 | — |
| A20 | 04_setup.md:19 | "HiTab이 붙인 집계 라벨과 정답 근거 셀 수로 세 유형으로 나눈다" | 인용 없음(데이터셋 [A:R]) | [A:R] | README 51–53, 71–76, 90–91행 | `aggregation` 라벨과 quantity_link 의 `[ANSWER]` 셀 목록이 데이터에 있음. [A:R]로 재계산: [ANSWER]가 데이터 셀·집계 none·셀 1개 = 991, 집계 있음 = 216(원고와 같음) | 정확 | — |
| A21 | 04_setup.md:55 | "HiTab은 데이터셋의 공식 채점 규칙 …" (공식 채점 코드의 존재·내용) | 인용 없음([A:R]) | [A:R], [A:H] | [A:R] qa/table/utils.py 10–52행(hmt_score·hmt_process_answer·hmt_equal), qa/datadump/utils.py 68–110행(normalize·naive_str_to_float), qa/table/experiments.py 156행; [A:H] §3.2.4 p.1098 | 수치는 `math.fabs(prediction - answer) < 1e-5`, 문자열은 WikiTableQuestions 정규화 후 일치, 목록은 길이·순서까지 일치. 원 논문의 지표 이름은 "Execution Accuracy (EA) … percentage of samples with correct answers" | 정확 | — |
| A22 | 04_setup.md:55 | "…공식 채점 규칙을 이식한 채점기로" | 인용 없음([A:R] 대 원고 코드) | [A:R] | [A:R] 위 A21 위치; 원고 코드 rag_agent/eval/metrics.py 187–212, 259–310행, scripts/answer_accuracy.py 318행 | 비교·정규화 규칙은 공식과 같으나, 실제 채점 함수 hitab_exact_match_text 는 (1) 여러 값 정답일 때 자유 텍스트 답을 `,` `;` `and` `to`로 나눠 값별 비교, (2) 여러 값 문자열 정답도 나눠 비교, (3) `$` 제거를 더함. 원고는 이 추가를 적지 않음 | 부정확 | "HiTab은 공식 저장소의 채점 규칙(qa/table/utils.py: 수치는 1e-5 이내, 문자열은 정규화 후 일치; 원 논문의 execution accuracy)을 이식하고, 리더가 자유 텍스트로 답하므로 정답이 여러 값이면 답을 쉼표·세미콜론·and·to로 나눠 값별로 비교하는 단계와 '$' 제거만 더한 채점기로 잰다." |
| A23 | 08_refs.md:5 | Cheng, Z., Dong, H., Wang, Z., et al. (2022). HiTab: … ACL 2022, pp. 1094–1110 | Cheng et al. (2022) | [A:H] 소개 페이지 https://aclanthology.org/2022.acl-long.78/ | 서지 정보, PDF 1쪽 머리글 | 저자 9명 첫 셋 Zhoujun Cheng, Haoyu Dong, Zhiruo Wang; 제목 동일; "Proceedings of the 60th Annual Meeting … (Volume 1: Long Papers), pages 1094–1110"; DOI 10.18653/v1/2022.acl-long.78 | 정확 | — |
| A24 | 08_refs.md:26 | Zhou, W., Ma, B., Friedrich, A., & Mesgar, M. (2026). … In *Proceedings of ACL 2026*. arXiv:2510.09671. | Zhou et al. (2026) | [A:Z] v2 초록 페이지; ACL 2026 소개 페이지 https://aclanthology.org/2026.acl-long.557/ | 서지 정보 | 저자 4명·순서·제목·연도 일치, arXiv 설명 "Accepted at ACL 2026 Main". 논문집 판은 "Proceedings of the 64th Annual Meeting … (Volume 1: Long Papers)", pp. 12161–12189, San Diego. 원고 항목에는 논문집 이름 전체와 쪽이 없음(같은 목록 Cheng 항목과 형식 불일치) | 부정확 | "Zhou, W., Ma, B., Friedrich, A., & Mesgar, M. (2026). Table Question Answering in the Era of Large Language Models: A Comprehensive Survey of Tasks, Methods, and Evaluation. In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 12161–12189. arXiv:2510.09671." |
| B1 | 01_intro.md:15 | "질문과 짝지어진 … 문서(MultiHiertt)가 직접 주어진다" | 인용 없음(문장의 인용 Zhou 2026·Cheng 2022는 다른 묶음) | [P][R] | [P] 3.2절 p.6591, 그림 3 p.6593; [R] README L34–47 | 문서마다 "annotators are required to compose one or two QA examples"(3.2절). 그림 3 입력은 "Whole Document containing Multiple hierarchical tables and paragraphs"와 Question. README: 항목마다 uid·paragraphs·tables·table_description·qa(질문 하나). | 정확 | — |
| B2 | 02_related.md:7 | "재무 보고서 문서에서 여러 개의 계층형 표와 본문 문단을 함께 읽고 수치 추론을 해야 하는 질문을 모았다" | Zhao et al.(2022) | [P] | 초록 p.6588; 3.3절 p.6592; 3.4절 p.6592; 6장 p.6596 | 문서 특성: "each document contain multiple tables and longer unstructured texts; most of tables contained are hierarchical"(초록). 질문 단위는 다름: "10.24% … only require the information in the paragraphs", "33.09% … only … one table", 표와 본문 모두 48.74%(3.4절). 범위 선택형 답 "set to ≤ 20%"(3.3절). | 부정확 | "Zhao et al.(2022)의 MultiHiertt는 여러 개의 표(대부분 계층형)와 긴 본문 문단이 든 재무 보고서 문서를 주고, 주로 수치 추론을 요구하는 질문을 모았다. 표와 본문을 함께 써야 하는 질문은 48.74%, 본문만 필요한 질문은 10.24%, 표 하나만 필요한 질문은 33.09%다(원 논문 3.4절)." |
| B3 | 02_related.md:7 | "각 질문에는 … 표 셀(table evidence)과 본문 문장(text evidence), 그리고 계산 과정을 나타내는 프로그램이 주석되어 있다" | Zhao et al.(2022) | [P][R][H] | [P] 3.2절 p.6591; [R] README L41–47; [H] train.json | "For those questions requiring numerical expression, the annotators are then asked to write down the reasoning programs"(3.2절). train.json 7,830건 중 program 빈 문항 1,524건(question_type=span_selection), text_evidence 빈 문항 2,908건, table_evidence 빈 문항 713건. | 부정확 | "각 질문에는 답에 필요한 표 셀(table evidence)과 본문 문장(text evidence)이 근거로 표시되어 있고(질문에 따라 한쪽만 있다), 수치 계산이 필요한 질문에는 계산 과정을 나타내는 프로그램도 주석되어 있다." |
| B4 | 02_related.md:7 | "먼저 사실(fact)을 검색하고 그 결과로 추론하는 2단계 모델을 제안" | Zhao et al.(2022) | [P] | 4장 p.6592, 그림 3 p.6593 | "MT2Net first applies fact retrieving module to extract relevant supporting facts … Then, a reasoning module is adapted to perform reasoning over retrieved facts"(4장). | 정확 | — |
| B5 | 02_related.md:7 | "상위 10개 사실의 재현율 76.4%" | Zhao et al.(2022) | [P] | 5.4절 p.6594 | "we have 76.4% recall for the top-10 retrieved facts". 검색기는 BERT-base 분류기(5.2절 p.6594). 분할·재현율 정의는 원문에 없고 원고도 덧붙이지 않음. | 정확 | — |
| B6 | 02_related.md:7 | "상위 15개 사실의 재현율 80.8%" | Zhao et al.(2022) | [P] | 5.4절 p.6594 | "and 80.8% recall for the top-15 retrieved facts". 조건은 B5와 같음. | 정확 | — |
| B7 | 02_related.md:7 | "검색기 성능을 본문에서 … (상위 10개·상위 15개) 보고" | Zhao et al.(2022) | [P] | 5.4절 p.6594 | 두 재현율은 표가 아니라 5.4절 본문 한 문장에만 나오고, 검색 개수는 10과 15 두 가지다. | 정확 | — |
| B8 | 02_related.md:7 | "주 결과표는 상위 10개 고정 조건이다" | Zhao et al.(2022) | [P] | 5.2절 p.6594; 5.4절 p.6594; 5.1절 p.6593–6594; 표 4 p.6594; p.6595 | "we take the top-10 retrieving facts as the retriever results"(5.2절), "same fact retrieving results for all 'Retrieving + Reasoning' models"(5.4절). 다만 표 4의 TAPAS는 "table with most portion of top-15 retrieved facts", Longformer는 "encode the whole document", TAGOP·FinQANet은 "flatten each table by rows"(p.6595)로 자체 검색. | 부정확 | "주 결과표(Table 4)에서 MT2Net과 '사실 검색+추론' 비교군은 상위 10개 사실을 고정 입력으로 썼다(원 논문 5.2절; TAPAS 비교군은 상위 15개 사실이 가장 많이 든 표 하나를 입력으로 썼다)." |
| B9 | 03_method.md:21 | "표가 HTML로만 주어지고" | 인용 없음 | [R][H][P] | [R] README L39–40; [H] train/dev/test.json, table_description_generation.py; [P] 3.1절 p.6591 | README: "tables: the list of tables in HTML format", "table_description: the list of table descriptions for each data cell in tables. Generated by the pre-processing script". table_description은 세 분할 모든 문항(7,830·1,044·1,566)에 있음. 원문: "we use a pre-processing script to extract the hierarchical structure of each HTML-format table"(3.1절). | 부정확 | B10과 함께: "MultiHiertt의 표는 머리글 칸 표시(<th>)가 없는 HTML로 주어지고, 머리글 행·열의 수나 층 구조를 따로 담은 항목이 없다. 데이터셋에는 원 저자의 전처리 스크립트가 데이터 셀마다 머리글을 이어 붙여 만든 문장(table_description)이 함께 있다. 본 연구는 이 문장을 쓰지 않고 머리글 행 수와 머리글 열 수를 규칙으로 추정한다." 쓰지 않은 이유를 한 문장 덧붙인다(이유는 저자가 확인해 적는다). |
| B10 | 03_method.md:21 | "머리글 계층 정보가 없다" | 인용 없음 | [H][P] | [H] 세 분할의 tables HTML, table_description; [P] 4장 p.6592 | HTML 태그는 table·tr·td(와 sup·sub·i)뿐이고 th는 0건, 머리글 범위 필드도 없음. 그러나 table_description이 셀마다 행 머리글·절 제목 행·여러 층 열 머리글을 이어 씀. 예(train 첫 문항): "Table 0 shows Mortgages and notes payable of December 31, 2007 Carrying Value (In thousands) is $584,795 .". 원문: "turn each cell into a sentence, along with its hierarchical row and column headers"(4장). | 부정확 | B9의 수정안과 같음. |
| B11 | 04_setup.md:21 | "공개 test 파일(test.json, 1,566문항)" | 각주 [^1](HF 커밋) | [P][H] | [P] 표 3 p.6592; [H] test.json | "Test Set Size 1,566 (15%)"(표 3). test.json 항목 1,566개, sha256이 각주 값과 일치. | 정확 | — |
| B12 | 04_setup.md:21 | "모든 문항의 qa 항목에 질문(question)만 포함하며, 정답·풀이·정답 근거를 포함하지 않는다" | 각주 [^1](HF 커밋) | [H][R] | [H] test.json; [R] README L13 | test.json 1,566건 모두 qa 키가 question 하나뿐(train·dev는 question·answer·program·text_evidence·table_evidence·question_type). README: "The leaderboard for the private test data is held on CodaLab". | 정확 | — |
| B13 | 04_setup.md:21 | "공식 저장소는 test 예측을 CodaLab 리더보드에 제출해 점수를 받도록 안내한다" | 공식 저장소 URL | [R] | README L13, L108; evaluate.py L74–76 | README: "submit test_predictions.zip to CodaLab to get the final score". evaluate.py: "Please submit the test prediction file to CodaLab to get the results". | 정확 | — |
| B14 | 04_setup.md:21 | "7,830 질의"(train 분할) | 인용 없음 | [P][H] | [P] 표 3 p.6592; [H] train.json | "Training Set Size 7,830 (75%)", "Development Set Size 1,044 (10%)"(표 3). train.json 7,830건, dev.json 1,044건. | 정확 | — |
| B15 | 04_setup.md:21 | "본문 문장 근거가 필요한 4,922건 … 남은 2,908건" (범위 밖 보조 확인) | 인용 없음 | [H] | [H] train.json | text_evidence가 있는 문항 4,922건(표 근거도 있음 4,209 + 본문만 713), 없는 문항 2,908건. | 정확 | — |
| B16 | 04_setup.md:21 | "(이 구분은 공식 `question_type` 필드와 100% 일치한다)" | 인용 없음 | [H][R] | [H] train.json qa.question_type; [R] README L36–47 | train.json 전체에서 program 있음 = arithmetic 6,306/6,306, program 없음 = span_selection 1,524/1,524. 필드는 데이터 파일에 있으나 README 형식 목록에는 없음. | 정확 | — |
| B17 | 04_setup.md:53 | "MultiHiertt 원 논문이 주 결과표를 운영점 하나로 보고하는 관행" | Zhao et al.(2022)(본문 표기만) | [P] | 표 4 p.6594; 5.2절 p.6594; 5.4절 p.6594 | 표 4는 모델마다 한 설정의 EM/F1만 싣고 검색 개수를 바꾼 값은 없음. MT2Net은 "top-10 retrieving facts"(5.2절)로 고정, top-10/15 재현율은 본문 한 문장뿐. (TAPAS 행의 top-15 조건은 B8 참조.) | 정확 | — |
| B18 | 04_setup.md:55 | "공식 `evaluate.py`의 문항 유형별 채점 규칙(조회는 문자열 비교, 산술은 수치 비교)" | 인용 없음(공식 코드) | [R] | evaluate.py L25–35, L88–99; span_selection_utils.py `get_span_selection_metrics`; utils/utils.py L6–21 | 분기는 정답 program 유무. 예측에 program이 없을 때: 정답 program 없음 → DROP식 정규화(소문자·구두점·관사 제거, 숫자 표기 통일) 후 문자열 일치; 정답 program 있음 → `str_to_num` 후 `math.isclose`(abs_tol=min(작은 값/1000, 0.1)), 숫자로 못 읽으면 문자열 비교로 넘어감. | 정확 | — |
| B19 | 04_setup.md:55 | "공식 규칙상 정답이 음수인 산술 문항은 텍스트 답으로 맞힐 수 없는데" | 인용 없음(공식 코드) | [R] | utils/utils.py L6–10; evaluate.py L25–35; span_selection_utils.py `_tokenize` | `str_to_num`이 `text.replace("-", "")`로 예측의 '-'를 지움. 공식 함수 실행: 정답 -5.0에 예측 "-5.0"·"-5"·"5"는 0점. 숫자로 못 읽히는 "the 5"·"5.0 ."은 1점(문자열 비교 경로에서 '-'가 토큰 구분자라 정답이 "5.0"으로 정규화됨). 정답을 숫자 그대로 넣은 상한 계산에는 영향 없음. | 부정확 | "공식 규칙은 숫자로 읽히는 예측에서 '-' 기호를 지운 뒤 정답과 비교하므로(utils/utils.py의 str_to_num), 정답이 음수인 산술 문항은 숫자 형태의 답으로는 맞힐 수 없다." |
| B20 | 06_discussion.md:79 | "MultiHiertt 주석 지침은 문항마다 표와 본문의 근거를 모두 표시하도록 했으나" | Zhao et al.(2022) | [P] | 3.2절 p.6591; 3.3절 p.6591 | "They are also required to mark all the supporting facts from tabular and textual content for each question"(3.2절). "annotation guideline"은 3.3절에 언급. '모두'는 all의 뜻이며 두 종류가 다 있어야 한다는 뜻은 아님(3.4절: 본문만 10.24%). | 정확 | — |
| B21 | 06_discussion.md:79 | "원 논문의 100문항 사람 평가" | Zhao et al.(2022), Table 2 | [P] | 표 2 설명 p.6591 | "Human evaluation over 100 samples of MultiHiertt. Four internal evaluators are asked to rate the samples on a scale of 1 to 5." | 정확 | — |
| B22 | 06_discussion.md:79 | "근거 정확성(Support Facts Correctness) 평균 점수가 4점 이상인 문항" | Zhao et al.(2022), Table 2 | [P] | 표 2 p.6591 | 행 이름 "Support Facts Correctness", 열 "%S ≥ 4" = "percent of samples that have average score ≥ 4"(1–5점 척도, 평가자 4명). | 정확 | — |
| B23 | 06_discussion.md:79 | "84.9%" | Zhao et al.(2022), Table 2 | [P] | 표 2 p.6591 | "Support Facts Correctness 84.9 0.81 0.77 / [0.72, 0.82]"(%S≥4, Agree, Kappa/95% CI). | 정확 | — |
| B24 | 06_discussion.md:79 | "(Zhao et al., 2022, Table 2)" | Zhao et al.(2022) | [P] | 3.3절 p.6591 | "The human evaluation scores and inter-evaluator agreements are reported in Table 2." 표 2가 3.3절 Quality Control에 있음. | 정확 | — |
| B25 | 08_refs.md:25 | "Zhao, Y., Li, Y., Li, C., & Zhang, R. (2022). MultiHiertt: … In Proceedings of the 60th Annual Meeting of the ACL (ACL 2022), pp. 6588–6600." | 서지 | [B][P] | BibTeX; PDF 1쪽 머리 | 저자 Zhao, Yilun·Li, Yunxiang·Li, Chenying·Zhang, Rui, 2022, 제목, 60th ACL, pages 6588–6600 모두 일치. 권 "(Volume 1: Long Papers)"와 DOI 10.18653/v1/2022.acl-long.454는 항목에 없음(같은 목록의 ACL 항목 5·10행과 같은 형식이라 오류는 아님). | 정확 | — |
| C1 | 01_intro.md:28 | "TableRAG(Chen et al., 2024)의 셀 검색을 옮긴 재구현 두 가지" | Chen et al., 2024 | [C:N], [C:G] | 3.3절 p.4 "Cell Retrieval"; [C:G] agent/retriever.py:114–126 build_cell_corpus | 원 논문의 구성요소로 셀 검색이 있고 공식 코드에 셀 문서 생성 함수가 있다. leaf·path는 원고가 만든 두 변형이다(원고 04_setup.md:39–40이 밝힘). | 정확 | — |
| C2 | 01_intro.md:28 | "그 논문의 비교군 RowCol … RandRow" | Chen et al., 2024 | [C:N] | 4.2절 p.6; 표 2·3 p.7 | 비교군으로 "RandRowSampling", "RowColRetrieval"을 둔다. | 정확 | — |
| C3 | 01_intro.md:28 | "RowCol(표의 행과 열을 각각 한 단위로 색인해 검색)" | Chen et al., 2024 | [C:N], [C:G] | 4.2절 p.6; 그림 1(c) 설명 p.2; [C:G] retriever.py:162 | p.6 "retrieve the top K rows and columns … to form a sub-table"; p.2 "Only the intersection of these rows and columns is presented to the LM." | 부정확 (축소: 교집합 부분 표라는 핵심 조건 누락) | "RowCol(표의 행과 열을 각각 한 단위로 색인해 질문과 가까운 행·열을 찾고, 그 행과 열이 겹치는 셀만 전달)" |
| C4 | 01_intro.md:28 | "RandRow(질문의 표에서 행을 무작위 순서로 골라 셀 예산까지 채우는 방법, 검색 없음)" | Chen et al., 2024 | [C:N], [C:G] | 4.2절 p.6; [C:G] agent/agent.py:176 | "randomly selects rows from the table with equal probabilities. When the total number of rows exceeds K, we select K rows". 원 논문은 K=30행을 뽑고, 원고는 셀 예산까지 채운다고 괄호 안에서 밝힌다. | 정확 | — |
| C5 | 02_related.md:11 | "TableRAG(Chen et al., 2024)는 \"cell retrieval\" … 이라 부른다" | Chen et al., 2024 | [C:N] | 1장 p.2; 3.3절 p.4 | 절 제목 "Cell Retrieval", "Cell Retrieval with Encoding Budget". | 정확 | — |
| C6 | 02_related.md:13 | "백만 토큰 규모의 큰 표를 LLM이 다루도록" | Chen et al., 2024 | [C:N] | 제목; 초록 p.1 | 제목 "Million-Token Table Understanding"; 초록 "two new million-token benchmarks". | 정확 | — |
| C7 | 02_related.md:13 | "스키마(열 이름과 자료형·값 범위 요약)와 셀(열 이름과 셀 값의 쌍)을 따로 검색해" | Chen et al., 2024 | [C:N] | 3.3절 Schema Retrieval p.4; 알고리즘 1 p.14 | "for categorical columns, we present the three most frequent categories as example values." 값 범위(최솟값·최댓값)는 숫자·날짜 열에만 해당한다. 셀 쌍·별도 검색(알고리즘 1의 6·7행)은 맞다. | 부정확 (축소: 범주형 열의 예시값 3개를 "값 범위"로 뭉뚱그림) | "스키마(열 이름·자료형과, 숫자·날짜 열은 최솟값·최댓값, 범주형 열은 가장 자주 나오는 값 3개)와 셀(열 이름과 셀 값의 쌍)을 따로 검색해" |
| C8 | 02_related.md:13 | "주 실험은 토큰 예산 B=10,000" | Chen et al., 2024 | [C:N] | 4.3절 p.7; 3.3절 p.4 | p.7 "we set the cell encoding budget B = 10,000"; p.4 B는 셀 색인에 넣는 서로 다른 열-값 쌍 수의 상한("restrict our encoding to the B most frequently occurring pairs"). 알고리즘 1(p.14)과 4.8절(p.9)은 "token encoding budget"이라고도 적어 원문 용어가 섞여 있다. | 부정확 (경미, 용어: 값은 맞으나 "토큰 예산"은 리더 입력 토큰 예산으로 읽힌다) | "셀 인코딩 예산 B=10,000(셀 색인에 넣는 서로 다른 열-값 쌍의 최대 수)" |
| C9 | 02_related.md:13 | "검색 개수 K=5로 설정했고" | Chen et al., 2024 | [C:N] | 4.3절 p.7; 그림 2 설명 p.4; 3.4절 p.6 | p.7 "the retrieval limit K = 5"; p.4 "The top K candidates from each query are combined"; p.6 확장 질의 "approximately 3-5 queries". K는 확장 질의 하나당 개수다. | 부정확 (경미, 조건 누락: 질의당 K이고 질의 3~5개의 결과를 합침) | "검색 개수 K=5(LM이 만든 확장 질의 3~5개 각각에서 상위 5개를 찾아 합침)" |
| C10 | 02_related.md:13 | "비교군인 RandRowSampling과 RowColRetrieval은 K=30으로 늘렸다" | Chen et al., 2024 | [C:N] | 4.3절 p.7 | "For RandRowSampling and RowColRetrieval, we increase the retrieval limit to K = 30." | 정확 | — |
| C11 | 02_related.md:13 | "이 논문은 검색 품질을 재현율·정밀도로 따로 보고한다" | Chen et al., 2024 | [C:N] | 4.5절, 표 3 p.7 | 표 3 "Evaluation of retrieval performance … R: recall, P: precision." (F1도 함께 보고). | 정확 | — |
| C12 | 02_related.md:13 | "TableRAG의 셀·스키마 텍스트 표현을 공식 코드와 대조해 이식하고(leaf, path 두 변형)" | Chen et al., 2024 (공식 코드) | [C:G] | retriever.py:102–126; utils/utils.py:50–67 | 범주형 셀 문서 문자열 `{"column_name": …, "cell_value": …}`은 같다. 다른 점: 숫자 열 판정 규칙(아래 R4), 스키마 문서의 임베딩 대상(원 코드 `page_content=col_name`, 111행; 원고는 요약 문자열 전체), 숫자 요약 문서의 dtype 표기 고정("float64"), 행 라벨 문서(stub) 추가. 원고 본문은 이 차이를 적지 않는다. | 부정확 (조건 누락) | 문장 끝에 "다만 숫자 열 판정 규칙, 스키마 문서의 임베딩 텍스트, 행 라벨 문서 추가는 공식 코드와 다르다(4.2절)"를 붙이고 4.2절에 R2~R4를 적는다. |
| C13 | 02_related.md:13 | "논문의 비교군 둘을 함께 구현했다" | Chen et al., 2024 | [C:N] | 4.2절 p.6 | RandRowSampling, RowColRetrieval. | 정확 | — |
| C14 | 02_related.md:13 | "다만 검색기는 본 방법과 같은 것을 쓴다(4.2절)" | Chen et al., 2024 | [C:N] | 4.3절 p.7; 표 5 p.8 | 원 논문은 "OpenAI's text-embedding-3-large as the encoder for dense retrieval", 표 5에서 임베딩만 쓴 검색이 BM25·혼합보다 높다. 원고는 검색기를 바꿨다고 밝힌다. | 정확 | — |
| C15 | 02_related.md:44 | "TableRAG … 큰 표가 주어짐 \| 스키마, (열, 값) 쌍 \| 재현율·정밀도(운영점 하나)" | Chen et al., 2024 | [C:N] | 3.2절 p.3; 표 3 p.7; 그림 5·6 p.8–9 | p.3 "we are presented with a table T"; 표 3은 4.3절의 한 설정(K=5, B=10,000)에서 R·P·F1. K·B를 바꾼 그림 5·6은 답변 정확도만 그린다. | 정확 | — |
| C16 | 04_setup.md:29 | "비교군의 텍스트 표현은 원 논문의 공개 코드와 대조해 이식했다." | Chen et al., 2024 (공식 코드) | [C:G] | agent/agent.py:176–179; retriever.py:131, 139 | RowCol의 "\|" 이음은 코드와 같다(131·139행). RandRowSampling은 원 코드에 색인 텍스트가 없고 리더에 `sampled_table.to_markdown(index=False)`(열 이름 줄 있음)를 준다. 원고 RandRow는 "제목 \| 행 라벨\|값…"(열 이름 없음, 03_method.md:69)을 준다. | 부정확 (RandRow 리더 입력이 원 코드와 다르고 밝히지 않음) | "RandRow는 원 코드가 뽑은 행을 열 이름이 있는 마크다운 표로 주지만, 본 연구는 행 값을 '\|'로 이은 텍스트(열 이름 없음)를 준다"를 4.2절에 적거나, 열 이름 줄을 붙여 RandRow 답변을 다시 실행한다. |
| C17 | 04_setup.md:29 | "모든 비교군은 본 방법과 같은 … 하이브리드 검색기(α=0.7) … 를 쓴다" | Chen et al., 2024 | [C:N], 원고 코드 | 4.2절 p.6; retrieval_accuracy.py:950 | 원 논문의 RandRowSampling은 무작위 추출이고, 원고 구현도 `np.random.default_rng(seed).permutation(sel)`로 순서를 정해 검색 점수를 쓰지 않는다. | 부정확 (경미: RandRow는 검색기를 쓰지 않음) | "RandRow(검색 없음)를 뺀 모든 비교군은 …" |
| C18 | 04_setup.md:37 | "RowCol \| Chen et al.(2024)의 비교군 \| 표의 행 하나 또는 열 하나 \| 논문과 대조" | Chen et al., 2024 | [C:N] | 4.2절 p.6 | "we encode rows and columns". 색인 단위 서술로는 맞다(교집합 선택 규칙 누락은 재구현 차이 R11). | 정확 | — |
| C19 | 04_setup.md:38 | "RandRow \| Chen et al.(2024)의 비교군 \| 무작위로 섞은 행 \| 논문과 대조" | Chen et al., 2024 | [C:N] | 4.2절 p.6 | "randomly selects rows from the table with equal probabilities". | 정확 | — |
| C20 | 04_setup.md:39 | "(열 잎 이름, 셀 값) 쌍 + 열 스키마 \| … LM 질의 확장·별도 스키마 검색 경로·풀이기 없음" | Chen et al., 2024 | [C:N] | 3.3절 p.3–5; 그림 2 p.4 | 원 논문의 핵심 구성요소는 Tabular Query Expansion, Schema Retrieval, Cell Retrieval, Program-Aided Solver다. 빠진 셋을 정확히 적었다. | 정확 | — |
| C21 | 04_setup.md:40 | "(열 전체 경로, 셀 값) 쌍 + 열 스키마 \| 위와 같음" | Chen et al., 2024 | [C:N] | 3.2절 p.3 | 원 논문의 표는 열마다 이름 하나(C_j)인 평평한 표다. path는 원고의 변형이며 원고가 그렇게 표시한다. | 정확 | — |
| C22 | 04_setup.md:43 | "(1) LM 질의 확장: 넣지 않았다. 질문 문장을 그대로 임베딩한다." | Chen et al., 2024 | [C:N], [C:G], 원고 코드 | 3.3절 p.3–4; 부록 D·E p.16; [C:G] rag_agent.py:71 | 원 논문은 스키마용·셀용 질의를 LM으로 따로 만든다(셀 질의에서 숫자는 코드가 버린다). 원고 코드는 `q["question"]` 하나를 인코딩한다. | 정확 | — |
| C23 | 04_setup.md:43 | "(2) 스키마 문서(열 이름과 자료형, 숫자 열은 최솟값·최댓값, 범주형 열은 자주 나오는 값 3개)는 만들지만, 원 논문처럼 별도 검색 경로로 찾지 않고 셀 문서와 한 색인에 섞어 같은 검색기로 찾는다." | Chen et al., 2024 | [C:N], [C:G] | 3.3절 p.4; 알고리즘 1 p.14 (11–16행); [C:G] retriever.py:111 | 내용 요약과 별도 경로는 원문과 맞다. 다만 원 논문은 스키마를 "matches them against the encoded column names"로 찾는다(열 이름만 임베딩, 요약은 결과로만 붙음). 원고는 요약 문자열 전체를 임베딩한다. | 부정확 (경미, 조건 누락) | 끝에 "원 논문은 스키마 DB에서 열 이름만 임베딩하고 요약은 검색 결과에 붙이지만, 본 구현은 요약 문자열 전체를 임베딩한다"를 붙인다. |
| C24 | 04_setup.md:43 | "(3) 열 이름-값 쌍 인코딩: 넣었다. 범주형 셀은 `{열 이름, 셀 값}` 문서가 되고 같은 쌍은 하나로 합친다." | Chen et al., 2024 | [C:N], [C:G] | 3.3절 p.4; [C:G] retriever.py:122–124 | p.4 "a database of distinct column-value pairs"; 코드 형식 `{"column_name": "…", "cell_value": "…"}`, Counter로 중복 합침. | 정확 | — |
| C25 | 04_setup.md:43 | "숫자 열은 개별 값을 색인하지 않고 요약 문서 하나로 접는다." | Chen et al., 2024 | [C:N], [C:G], 원고 코드 | 1장 p.2; [C:G] retriever.py:118–121, utils/utils.py:60–64 | p.2 "TableRAG only encodes distinct and the most frequent categorical values"; 코드는 object가 아닌 열에 최솟값·최댓값 문서 하나를 둔다. 그런데 어느 열이 숫자 열인지는 원 코드가 pandas 추론(`pd.to_numeric(errors='ignore')`, 실패하면 object로 남아 셀마다 색인)으로, 원고는 "쉼표를 지우고 비어 있지 않은 셀이 모두 숫자면 숫자 열"로 정한다. 이 규칙(본문 결과에 쓴 infer)에서 HiTab 단일 셀 조회 정답 셀 991개 중 890개가 접힌 열에 있고, 어떤 문서로도 전달될 수 없는 정답이 664건이다(전달 가능 상한 .3300, leaf·path 같음). 원고 코드 주석은 "infer 는 그만큼 이 비교군을 과소평가한다"고 적었으나 본문은 적지 않는다. | 부정확 (조건 누락: 숫자 열 판정 규칙과 그 결과인 상한) | 뒤에 "숫자 열 판정은 원 코드의 pandas 자료형 추론이 아니라 '비어 있지 않은 셀이 모두 숫자로 읽히는 열'로 했다. 이 규칙에서 HiTab 단일 셀 조회 991건 중 664건은 정답 셀이 어떤 문서로도 전달되지 않아, 이 비교군의 검색 정확도 상한은 .3300이다"를 붙인다. |
| C26 | 04_setup.md:43 | "원 논문의 풀이기(pandas 코드를 쓰는 에이전트)는 쓰지 않고, 검색 결과를 본 방법과 같은 리더에 넣는다." | Chen et al., 2024 | [C:N] | 3.3절 p.5; 부록 F p.17; 알고리즘 1 p.14 (8행) | "In this work, we consider ReAct"; 부록 F 프롬프트는 `python_repl_ast`로 pandas `df`를 다룬다. 풀이기에 표 T 전체가 들어간다. | 정확 | — |
| C27 | 04_setup.md:43 | "원 논문(Chen et al., 2024, \"TableRAG: Million-Token Table Understanding with Language Models\", arXiv:2410.04739)" | Chen et al., 2024 | [C:N], [C:v3] | 표지 p.1; arXiv 초록 페이지 | 제목·arXiv 번호가 같다. | 정확 | — |
| C28 | 04_setup.md:47 | "두 데이터셋 모두 원 논문처럼 행(열)의 값을 \"\|\"로 이은 텍스트로 색인한다." | Chen et al., 2024 (논문·공식 코드) | [C:N], [C:G], 원고 코드 | 4.2절 p.6; [C:G] retriever.py:131, 139; agent/agent.py:176–179; 원고 retrieval_accuracy.py:371 | 논문 본문은 "we encode rows and columns"만 적고 "\|" 형식은 공식 코드에만 있다. RandRowSampling은 원 코드에서 색인하지 않는다(무작위 추출 후 마크다운). 원고는 값 앞에 표 제목(행은 행 잎 라벨도)을 붙인다(`f"{title} \| {vals}"`). | 부정확 (출처가 논문이 아니라 코드이고, RandRow에는 해당하지 않으며, 제목 접두를 밝히지 않음) | "RowCol은 원 논문의 공개 코드(build_row_corpus·build_column_corpus)처럼 행(열)의 값을 '\|'로 이은 텍스트로 색인하되 앞에 표 제목을 붙였다(행은 행 잎 라벨도). RandRow는 검색 없이 행을 고르며, 고른 행을 같은 '\|' 텍스트(열 이름 없음)로 리더에 준다. 원 코드의 RandRow는 열 이름이 있는 마크다운 표를 준다." |
| C29 | 04_setup.md:53 | "이는 TableRAG … 원 논문이 주 결과표를 운영점 하나로 보고하는 관행과 같다." | Chen et al., 2024 | [C:N] | 4.3절·표 2·표 3 p.7; 4.8절 그림 5·6 p.8–9 | 주 결과표(표 2)와 검색 표(표 3)는 K=5, B=10,000(비교군 K=30) 한 설정이다. K·B를 바꾼 결과는 부록이 아니라 본문 4.8절 그림에 있으나, 원고의 이 문장은 주 결과표만 말한다. | 정확 | — |
| C30 | 05_results.md:50 | "leaf와 path 표현은 (열 이름, 셀 값) 쌍만 담고 행 머리글을 담지 않는다. … 같은 열의 셀들이 열 이름과 값만으로는 구별되지 않는다." (낮은 이유로 제시) | Chen et al., 2024 (재구현) | [C:G], 원고 코드 | [C:G] retriever.py:118–121; 원고 retrieval_accuracy.py:258–327 | 원고가 쓴 숫자 열 규칙에서 HiTab 단일 셀 조회 991건 중 664건은 정답 셀이 어떤 문서로도 전달되지 않는다(상한 .3300). 표 5-2의 관찰값 .2775(leaf)·.2916(path)은 이 상한 아래에 있다. 제시된 이유에는 이 접기가 빠졌다. | 부정확 (조건 누락: 구조적 상한 .3300) | 첫 문장 앞에 "숫자 열은 최솟값·최댓값 요약 문서 하나로 접히므로(4.2절), 단일 셀 조회 991건 중 664건은 정답 셀이 어떤 문서로도 전달되지 않는다(상한 .3300)."를 넣는다. |
| C31 | 05_results.md:50 | "이 표현은 원 논문이 다룬 평평한 대형 표를 위한 것이며" | Chen et al., 2024 | [C:N], [C:G] | 3.2절 p.3; 표 7 p.15; [C:G] rag_agent.py (DataFrame 생성) | 표를 열 이름 C_j 하나씩 가진 T={v_ij}로 정의하고, ArcadeQA 평균 79,376행·BirdQA 62,813행을 다룬다(표 7). 작은 WikiTableQA(4.7절)도 평가하지만 설계 대상은 큰 표다. | 정확 | — |
| C32 | 05_results.md:50 | "원 논문의 LM 질의 확장, 별도 스키마 검색 경로, 풀이기를 함께 쓰지 않았다는 점(4.2절)" | Chen et al., 2024 | [C:N] | 3.3절 p.3–5; 그림 2 p.4 | 원 논문의 구성요소 셋과 일치. | 정확 | — |
| C33 | 05_results.md:68 | "RandRow는 질문의 표 안에서 행을 무작위로 고르는 방법이라" | Chen et al., 2024 | [C:N], [C:G] | 4.2절 p.6; [C:G] agent/agent.py:176 | "randomly selects rows from the table"; 코드는 질문의 표 `df`에서 `df.sample`. | 정확 | — |
| C34 | 06_discussion.md:7 | "(가장 큰 차이가 나는 비교군은 …) RandRow다. 이 표현들은 … 행 단위로만 색인해 상위 머리글이 빠진다(RandRow)." | Chen et al., 2024 | [C:N], 원고 코드 | 4.2절 p.6; 원고 retrieval_accuracy.py:950 | RandRow는 원 논문과 원고 모두 행을 무작위로 고르므로 검색 정확도가 텍스트 표현과 무관하다. 같은 "\|" 행 텍스트를 점수로 고르는 RowCol은 .7639, RandRow는 .3744다(표 5-2). | 부정확 (RandRow의 차이를 머리글 누락으로 설명함; RandRow는 색인·검색을 하지 않음) | "이 표현들은 셀 값에 행 머리글을 붙이지 않는다(leaf, path). RandRow는 검색 없이 행을 무작위로 고르므로(Chen et al., 2024, 4.2절) 정답 행이 예산 안에 들어오는 비율 자체가 낮다." |
| C35 | 06_discussion.md:81 | "TableRAG 셀 검색 재구현에는 원 논문의 LM 질의 확장, 별도 스키마 검색 경로, 풀이기가 없고(4.2절)" | Chen et al., 2024 | [C:N] | 3.3절 p.3–5 | 구성요소 셋과 일치. | 정확 | — |
| C36 | 08_refs.md:4 | "Chen, S.-A., Miculicich, L., Eisenschlos, J. M., et al. (2024). TableRAG: … In *Advances in Neural Information Processing Systems (NeurIPS 2024)*. arXiv:2410.04739." | 서지 | [C:N] 초록 페이지, [C:v3] | proceedings.neurips.cc 초록 페이지 메타데이터 | 저자 10명(첫 셋 순서 일치), 2024, 제목, arXiv 번호 일치. 논문집 권 37, 쪽 74899–74921이 빠졌다(같은 목록의 Lewis et al. 2020 NeurIPS 항목은 권·쪽을 적음). | 부정확 (권·쪽 누락) | "… In *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*, pp. 74899–74921. arXiv:2410.04739." |
| D1 | 01_intro.md:15 | "부분 표가 원래 표 대신 어떤 표 질의응답 시스템의 입력으로도 쓰일 수 있다는 Lin et al.(2023)의 정의" | Lin et al., 2023 (ITR) | ACL 2023 PDF | §3.1, p.9911 | "Tsub can replace T as the input to virtually any existing TableQA system." 원문은 "virtually any"(사실상 어떤) | 정확 | — |
| D2 | 01_intro.md:15 | "이 구성은 … Lin et al.(2023)의 정의와 같다" (상위 20셀이 표 대신 리더 입력) | Lin et al., 2023 | ACL 2023 PDF | §3.1, p.9911; §3.2 Algorithm 1, p.9911; §3.3, p.9911 | 부분 표는 "the cells at the intersection of the selected rows and columns", 행·열 각 1개 이상 필요, 길이 예산 b 안의 부분 표 N개를 각각 모델에 넣음. 본 연구의 20셀 집합은 행·열 교차로 만든 부분 표가 아님 | 부정확 | "…원래 표 대신 리더 입력이 된다는 점에서 Lin et al.(2023)의 과제 정의(부분 표가 원래 표를 대신해 기존 표 질의응답 시스템의 입력이 됨)와 같은 취지다. 다만 ITR은 행·열을 검색해 그 교차로 부분 표를 만들고, 본 연구는 셀 하나를 단위로 고른다." |
| D3 | 01_intro.md:28 | "이종 문서용 TableRAG(Yu et al., 2025)" | Yu et al., 2025 | EMNLP PDF | 제목; 초록, p.14063 | 제목 "…Framework for Heterogeneous Document Reasoning"; "heterogeneous documents, comprising both textual and tabular components" | 정확 | — |
| D4 | 01_intro.md:28 | "TableRAG(Yu)의 청크 표현 … 같은 임베딩 모델, 같은 검색기, 같은 셀 예산 규칙(20셀)에 놓았다" | 원고 코드 | 결과 메타(rerun_20260926) | hitab_test_gold_trag_hetero.json, mh_train_trag_hetero.json | 두 파일 모두 encoder BAAI/bge-base-en-v1.5, alpha 0.7, budget 20, budget_policy whole_unit_stop_after_distinct_cells(본 방법과 같음). 청크 표현 자체의 차이는 아래 재구현 차이 표 | 정확 | — |
| D5 | 02_related.md:11 | "Lin et al.(2023)은 표 안 검색기(Inner Table Retriever, ITR)를 제안했다" | Lin et al., 2023 | ACL 2023 PDF | 초록, p.9909 | "We propose Inner Table Retriever (ITR), a general-purpose approach for handling long tables in TableQA" | 정확 | — |
| D6 | 02_related.md:11 | "질문에 답하는 데 필요한 정보를 가장 많이 담은 부분 표를 원래 표에서 찾고" | Lin et al., 2023 | ACL 2023 PDF | §3.1, p.9911 | "find one or more sub-tables Tsub containing the most relevant information from T needed to answer q" | 정확 | — |
| D7 | 02_related.md:11 | "그 부분 표가 원래 표 대신 기존의 어떤 표 질의응답 시스템에도 입력으로 쓰일 수 있게 했다" | Lin et al., 2023 | ACL 2023 PDF | §3.1, p.9911; §1, p.9910 | "can replace T as the input to virtually any existing TableQA system"; "integrated off-the-shelf into virtually any existing TableQA system" | 정확 | — |
| D8 | 02_related.md:11 | "본 연구의 과제 설정(표 안에서 셀을 골라 리더 입력을 대신함)은 이 정의와 같다" | Lin et al., 2023 | ACL 2023 PDF | §3.1, p.9911 | D2와 같음. ITR의 단위는 행 또는 열 전체("an item is either a complete row or a complete column")이고 결과는 그 교차로 만든 부분 표 | 부정확 | "…이 정의와 같은 취지다. 다만 ITR이 고르는 단위는 행·열이고 그 교차가 부분 표가 되며, 본 연구는 셀 하나를 고른다." |
| D9 | 02_related.md:11 | "ITR은 'inner table retriever'" 라 부른다 | Lin et al., 2023 | ACL 2023 PDF | 제목; 초록, p.9909 | 이름 "Inner Table Retriever (ITR)" | 정확 | — |
| D10 | 02_related.md:11 | "TAP4LLM(Sui et al., 2024)은 'table sampling'이라 부른다" | Sui et al., 2024 | Findings EMNLP 2024 PDF | 초록, p.10306; §2·§2.1, p.10308 | 모듈 이름 "(1) table sampling"; "Table Sampling: Decompose a large table T into a sub-table T′ with specific rows and columns." | 정확 | — |
| D11 | 02_related.md:11 | "표 안 셀을 고르는 단계는 논문마다 이름이 다르다" (ITR·TAP4LLM에 적용) | Lin et al., 2023; Sui et al., 2024 | 위 두 PDF | ITR §3.1, p.9911; TAP4LLM §2.1, p.10308 | 두 논문이 고르는 것은 셀이 아니라 행·열. TAP4LLM: "a subset of top-ranked rows and columns is selected to form a sub-table" | 부정확 | "표 안에서 질문에 필요한 부분(셀 또는 행·열)을 고르는 단계는 논문마다 이름이 다르다. … ITR은 행·열을 검색해 부분 표를 만드는 단계를 'inner table retriever', TAP4LLM은 상위 행·열로 부분 표를 만드는 단계를 'table sampling'이라 부른다." |
| D12 | 02_related.md:15 | "표와 본문이 섞인 이종 문서를 위한 RAG 프레임워크" | Yu et al., 2025 | EMNLP PDF | 초록, p.14063; §2, p.14064 | 과제 입력 "(T, D) where T denotes the textual contents and D refers to the tabular components" | 정확 | — |
| D13 | 02_related.md:15 | "문서의 표를 마크다운으로 바꾼 뒤 … 색인하고" | Yu et al., 2025 | EMNLP PDF | §3.2, p.14064 | "both the raw texts T and the Markdown-rendered form of each table … are segmented into chunks, which are then embedded". 본문 텍스트도 같은 색인에 들어감 | 정확 | — |
| D14 | 02_related.md:15 | "글자 수 기준으로 잘라(청크 1,000, 겹침 200)" | Yu et al., 2025 | EMNLP PDF; 공식 코드 | §5.1.2, p.14067; 코드 online_inference/tools/retriever.py:178 | 논문: "chunked into segments of 1000 tokens, with a 200-token overlap". 글자 기준은 코드(RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200), 길이 함수 기본값 len)에만 있음. 이 문장은 논문을 인용하면서 코드의 단위를 적음 | 부정확 | "…마크다운으로 바꾼 뒤 청크(원 논문은 1,000토큰·겹침 200토큰으로 적고, 공개 코드는 1,000자·겹침 200자로 나눈다. 4.2절)로 잘라 색인하고…" |
| D15 | 02_related.md:15 | "별도로 표를 관계형 데이터베이스에 넣어 SQL로 조회하는 경로를 둔다" | Yu et al., 2025 | EMNLP PDF | §3.2, p.14065; §3.3 "SQL Programming and Execution", p.14066 | "The tables are also ingested in a relational database (e.g., MySQL)"; "generates executable SQL programs and applies them to the pre-constructed MySQL database" | 정확 | — |
| D16 | 02_related.md:15 | "본 연구는 이 중 청크 표현만 재현했다" | 원고 코드 vs 공식 코드 | 공식 코드 f8d798f; 원고 코드 | tool_utils.py:134–155; retriever.py:222–227; 원고 chunks.py:10–26, 65–69 | 분할기·크기·겹침·머리말 형식은 같음. 그러나 마크다운 렌더링(원 코드: 첫 행 뒤 "\| --- \|" 구분 줄, 값 없는 칸 건너뜀 / 원고: 구분 줄 없음, 빈 칸 유지)과 머리말 이름(원 코드: 파일 이름 / 원고: HiTab 표 제목, MultiHiertt 빈 문자열)이 다름. 원고 본문에 이 차이 서술 없음 | 부정확 | "본 연구는 이 중 청크 표현을 공개 코드와 대조해 옮겼다(분할기·크기·겹침·머리말 형식은 같고, 마크다운 렌더링과 머리말 이름의 차이는 4.2절)." 그리고 4.2절에 차이 표의 1·3행 내용을 한두 문장으로 추가 |
| D17 | 02_related.md:15 | "SQL 경로와 전용 검색 구성은 재현하지 않았다" | 원고 코드 | 원고 코드; 결과 메타 | retrieval_accuracy.py:230–236 주석; 재실행 메타 | 원고 코드에 SQL·BGE-M3·재정렬기·질의 분해 없음. 검색기는 본 방법과 같은 bge-base-en-v1.5 하이브리드 | 정확 | — |
| D18 | 02_related.md:17 | "HiTab과 MultiHiertt에서 가져온 표를 본문과 함께 문서로 재구성한 코퍼스(2,178문서)" | Zhang et al., 2026 (MixRAG) | arXiv v3 | §3.2, p.3–4; 부록 A.2 표 6, p.11 | MultiHiertt는 "each document comprises textual content and hierarchical tables"라 문서를 그대로 가져옴. 재구성은 HiTab만("scrape contextual information utilizing webpage links provided in HiTab, reconstructing heterogeneous documents"). 2,178문서는 일치 | 부정확 | "…MultiHiertt 문서(본문과 계층형 표)와, HiTab 표의 원 웹페이지 본문을 모아 재구성한 문서로 이루어진 코퍼스(DocRAGLib, 2,178문서)에서…" |
| D19 | 02_related.md:17 | "질문에 맞는 문서를 찾는 코퍼스 전역 검색을 다룬다" | Zhang et al., 2026 | arXiv v3 | §3.1, p.3 | "our task is to retrieve the most relevant document D∗ and generate a comprehensive answer" (문서 모음 C 전체에서) | 정확 | — |
| D20 | 02_related.md:17 | "계층 구조는 행·열 단위 요약문으로 표현하고" | Zhang et al., 2026 | arXiv v3 | §4.1.2, p.4–5 | H-RCL 행·열 요약을 LLM으로 다듬어 만듦. 참고: BM25 단계는 표 단위 요약 코퍼스, 본문은 문장 단위(§4.1.3·§4.2, p.5–6) | 정확 | — |
| D21 | 02_related.md:17 | "LLM 재순위화를 쓰며" | Zhang et al., 2026 | arXiv v3 | 초록, p.1; §4.2 LLM-based Retrieval Stage, p.6 | "an ensemble retriever with LLM-based reranking" | 정확 | — |
| D22 | 02_related.md:17 | "문서 단위 적중률(HiT@K)을 보고한다" | Zhang et al., 2026 | arXiv v3 | 표 1, p.6; 부록 B, p.11 | "HiT@K: … proportion of questions for which the correct document is retrieved within the top-K candidates" | 정확 | — |
| D23 | 02_related.md:19 | "OHD는 표를 열 트리와 행 트리로 분해해 구조를 보존하는 LLM 입력 표현을 만든다" | Cao et al., 2026 | arXiv v2 | 초록, p.1 | "constructs structure-preserving input representations of complex tables for LLMs"; "decomposes irregular tables into a column tree and a row tree" | 정확 | — |
| D24 | 02_related.md:19 | "이것은 검색이 아니라 표 하나 전체의 표현 방법이다" | Cao et al., 2026 | arXiv v2 | §3.3 Algorithm 2, p.6; §3.4, p.6 | 모든 노드를 DFS로 순회해 R_col, R_row를 만들고 LLM이 합쳐 "a high-fidelity textual surrogate of the original semi-structured table"을 냄. 검색 단계 없음(관점 선택은 질문에 맞춤: "tailored to the specific question") | 정확 | — |
| D25 | 02_related.md:43 | ITR 설정 "표가 주어짐" | Lin et al., 2023 | ACL 2023 PDF | §3.1, p.9911 | "Given a question q and a table T" | 정확 | — |
| D26 | 02_related.md:43 | ITR 검색 단위 "부분 표" | Lin et al., 2023 | ACL 2023 PDF | §3.1–3.2, p.9911 | 검색 대상은 "an item is either a complete row or a complete column"; 부분 표는 검색한 항목을 조합한 결과 | 부정확 | 칸을 "행·열(교차로 부분 표 구성)"으로 |
| D27 | 02_related.md:43 | ITR 검색 품질의 보고 "—"(확인하지 않은 칸) | Lin et al., 2023 | ACL 2023 PDF | §4.1, p.9912; 그림 3, p.9916 | "evaluate ITR retrieval ability on WikiSQL using Recall@K, which measures whether all the gold rows/columns … are among the top-K retrieved items" | 부정확 | 칸을 "행·열 Recall@K(정답 행·열 전부가 상위 K 안, WikiSQL test)"로 채움. 본 연구 지표와 같은 '전부 포함' 기준이므로 본 연구 행의 굵은 글씨가 지표 자체의 새로움으로 읽히지 않게 할 것 |
| D28 | 02_related.md:45 | Yu 설정 "이종 문서 코퍼스" | Yu et al., 2025 | EMNLP PDF; 공식 코드 | §2, p.14064; 코드 retriever.py:190–206 | 과제 입력은 텍스트·표 문서 모음. 코드는 표(xlsx)와 문서(json) 전부를 한 색인에 넣음 | 정확 | — |
| D29 | 02_related.md:45 | Yu 검색 단위 "마크다운 청크 + SQL" | Yu et al., 2025 | EMNLP PDF | §3.2–3.3, p.14064–14066 | 마크다운 표·본문 청크 검색 + 스키마 매핑 후 SQL 실행 | 정확 | — |
| D30 | 02_related.md:45 | Yu 검색 품질의 보고 "—" | Yu et al., 2025 | EMNLP PDF | 본문·부록 전체 검색; 부록 E.2 그림 11, p.14077 | 검색 품질 지표(Recall·Hit 등) 보고 없음. top-k 민감도도 답 정확도로만 보고. 칸을 "보고 없음"으로 바꿔 적을 수 있음 | 정확 | — |
| D31 | 02_related.md:46 | MixRAG 설정 "코퍼스 전역" | Zhang et al., 2026 | arXiv v3 | §3.1, p.3; 부록 A.2, p.11 | 문서 모음 C에서 D∗ 검색; "the 2,178 heterogeneous documents are merged into a single collection" | 정확 | — |
| D32 | 02_related.md:46 | MixRAG 검색 단위 "행·열 요약문" | Zhang et al., 2026 | arXiv v3 | §4.2, p.5–6 | 임베딩 검색은 H-RCL 코퍼스 사용("we utilize the corpus constructed using the H-RCL method"). BM25는 표 단위 요약 코퍼스 | 정확 | — |
| D33 | 02_related.md:46 | "문서 적중률(HiT@K)" | Zhang et al., 2026 | arXiv v3 | 표 1, p.6; 부록 B, p.11 | D22와 같음 | 정확 | — |
| D34 | 04_setup.md:35 | "고정 청크 \| 업계 관행" (1,000자, 제목·머리글 블록을 청크마다 반복) | 인용 없음 | — | (참고) Yu et al., 2025 초록·§1, p.14063 | 출처가 없음. 가장 가까운 근거는 Yu 2025의 "The prevailing practice of flattening tables and chunking strategies"와 "Typically, chunking strategies are employed (Finardi et al., 2024)". 1,000자·머리글 반복은 어느 문헌에서도 오지 않은 본 연구의 설정 | 원문에 없음 | 출처 칸을 "널리 쓰이는 방식(Yu et al., 2025, p.14063)"으로 바꾸고, 1,000자와 머리글 반복은 본 연구가 정한 값이라고 적기 |
| D35 | 04_setup.md:36 | "\"File name/Table name\" 머리말" | Yu et al., 2025(공식 코드) | 공식 코드 | retriever.py:225; tool_utils.py:140 | 원 논문에는 없고 코드에만 있음: "File name: {key}"는 모든 청크 앞, "Table name: {name}"은 분할 전 표 맨 앞이라 첫 청크에만. 원고 코드도 같은 위치(chunks.py:11, 69) | 정확 | — |
| D36 | 04_setup.md:36 | "마크다운 표를 1,000자, 겹침 200자로 나눈 청크" | Yu et al., 2025(공식 코드) | 공식 코드; 원고 코드 | retriever.py:178; 원고 chunks.py:29–40; 재실행 메타 | 둘 다 LangChain RecursiveCharacterTextSplitter(1000, 200); 메타 chunk_measure "characters" | 정확 | — |
| D37 | 04_setup.md:36 | "청크 표현만. SQL 경로·전용 검색기 미재현" | 원고 코드 | 원고 코드 | retrieval_accuracy.py:230–236 | SQL·BGE-M3·bge-reranker-v2-m3 없음 | 정확 | — |
| D38 | 04_setup.md:49 | "원 논문 본문은 청크 크기를 '1,000토큰, 겹침 200토큰'이라 적지만" | Yu et al., 2025 | EMNLP PDF(+arXiv v1·v2) | §5.1.2, p.14067 | "the text is chunked into segments of 1000 tokens, with a 200-token overlap" (세 판본 같음) | 정확 | — |
| D39 | 04_setup.md:49 | "공개 코드는 글자 수 기준으로 나눈다" | 공식 코드 | f8d798f | retriever.py:13, 178; langchain_text_splitters base.py:51 (설치판 1.1.2) | RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200), length_function 지정 없음 → 기본값 len(글자 수) | 정확 | — |
| D40 | 04_setup.md:49 | "본 연구는 코드를 따랐다" | 원고 코드 | 원고 코드; 재실행 메타 | chunks.py split_chunks; hitab_test_gold_trag_hetero.json | chunk_measure "characters", chunk_size 1000, chunk_overlap 200, splitter RecursiveCharacterTextSplitter | 정확 | — |
| D41 | 06_discussion.md:81 | "TableRAG(Yu)의 SQL 경로도 재현하지 않았다" | Yu et al., 2025; 원고 코드 | EMNLP PDF; 원고 코드 | §3.3, p.14066; retrieval_accuracy.py:230–236 | 원 시스템에는 SQL 경로가 있고 원고 코드에는 없음 | 정확 | — |
| D42 | 08_refs.md:22 | Yu, X., Jian, P., & Chen, C. (2025). TableRAG: … In *Proceedings of EMNLP 2025*. arXiv:2506.10380. | — | ACL Anthology bib; arXiv abs | 2025.emnlp-main.710 | 저자 3명·연도·제목·학회·arXiv 번호 일치. 쪽은 14063–14082(선택해서 추가 가능) | 정확 | — |
| D43 | 08_refs.md:23 | Zhang, C., Chen, Q., & Zhang, M. (2026). Mixture-of-RAG … *Proceedings of the 32nd ACM SIGKDD … (KDD 2026)*, pp. 1880–1891. | — | arXiv v3; Crossref(ACM DL 못 엶) | arXiv v3 p.1 ACM Reference Format; Crossref 10.1145/3770854.3780171 | 저자·제목·학회 일치. 논문집 이름은 "…Knowledge Discovery and Data Mining **V.1**", 쪽 1880–1891(Crossref; arXiv v3 "12 pages"와 맞음). 권 "V.1" 빠짐, DOI 없음 | 부정확 | "…Data Mining V.1 (KDD 2026), pp. 1880–1891. https://doi.org/10.1145/3770854.3780171" |
| D44 | 08_refs.md:3 | Cao, B., Lu, H., Ma, C., Wang, T., Li, R., & Fan, J. (2026). Orthogonal Hierarchical Decomposition … *Proceedings of ICML 2026*. arXiv:2602.01969. | — | arXiv v2 | p.1 저자 목록·각주 | 저자 6명·제목 일치, "Proceedings of the 43rd International Conference on Machine Learning … PMLR 306, 2026" | 정확 | — |
| D45 | 08_refs.md:10 | Lin, W., Blloshmi, R., Byrne, B., de Gispert, A., & Iglesias, G. (2023). … *Proceedings of the 61st Annual Meeting of the ACL (ACL 2023)*, pp. 9909–9926. | — | ACL Anthology PDF·bib | 2023.acl-long.551 | 저자·제목·쪽 일치. 논문집 이름에 "(Volume 1: Long Papers)"가 있는데 빠짐 | 부정확 | "…Computational Linguistics (Volume 1: Long Papers), pp. 9909–9926." |
| D46 | 08_refs.md:17 | Sui, Y., Zou, J., Zhou, M., et al. (2024). TAP4LLM … *Findings of EMNLP 2024*. arXiv:2312.09039. | — | Findings PDF·bib; arXiv v3 | 2024.findings-emnlp.603 | 저자(7명 중 앞 3명+et al.)·제목·학회·arXiv 번호 일치. 쪽 10306–10323은 빠졌지만 틀린 것은 아님(다른 항목과 맞추려면 7명 전부와 쪽 추가) | 정확 | — |
| E1 | 01_intro.md:5 | "검색 증강 생성(Retrieval-Augmented Generation, RAG; Lewis et al., 2020)" (용어·개념 출처) | Lewis et al., 2020 | NeurIPS 2020 논문집 PDF | 초록 p.1(9459); §2.3 p.3(9461) | 초록 "a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG)"; §2.3 입력과 검색 문서를 "we simply concatenate them". 원문의 RAG는 DPR 검색기 + BART-large(400M)를 함께 미세조정하는 모델이라 원고가 쓰는 넓은 뜻(학습 없이 LLM 문맥에 넣음)보다 좁지만, 용어 출처로 인용하는 것은 맞음 | 정확 | — |
| E2 | 01_intro.md:5 | "대규모 언어 모델(LLM)은 … RAG 방식으로 널리 쓰인다" | Lewis et al., 2020 | NeurIPS 2020 논문집 PDF | 본문·부록 전체 | Lewis(2020)는 BART 기반 모델을 제안한 논문으로, LLM에서 RAG가 널리 쓰인다는 서술이 없음. 묶음 안에서는 Cuconasu arXiv v4 초록 p.1 "RAG has become increasingly important for Generative AI solutions, especially in enterprise settings"가 이 서술을 뒷받침함 | 원문에 없음 | "널리 쓰인다"에 별도 근거를 붙인다: "…(RAG; Lewis et al., 2020) 방식으로 널리 쓰인다(Cuconasu et al., 2024)." 또는 Lewis는 용어 출처로만 두고 "널리"를 뺀다 |
| E3 | 02_related.md:23 | "질문과 관련은 있으나 답이 아닌 정보가 섞이면 리더의 정확도가 떨어진다는 것은 확립된 현상이다" | 인용 없음 (뒤 문장의 Cuconasu 2024·CABINET 2024로 대조) | Cuconasu arXiv v4; CABINET arXiv v3 | Cuconasu 초록 p.1, §1 기여(1) p.2, Table 1 p.6; CABINET 초록 p.1 | 현상 자체는 두 문헌이 뒷받침함(Cuconasu Table 1: LLM 4개 모두 방해 문서를 넣으면 하락, Wilcoxon p<0.01). 그러나 Cuconasu는 이를 "One counter-intuitive finding of this work"라고 부르고 자기 연구를 "the first comprehensive study"라고 밝힘. 인용 문헌이 새 발견이라고 한 것을 "확립된 현상"이라 쓴 것은 과장 | 부정확 | "…정확도가 떨어진다는 결과가 여러 연구에서 보고되었다." ("확립된" 삭제) |
| E4 | 02_related.md:23 | "정답 문서가 있는 문맥에 주제상 가까운 방해 문서를 하나씩 더할수록 정확도가 단조롭게 떨어지는 것을 보였다" | Cuconasu et al., 2024 | arXiv 2401.14887v4 (ACM DL 판은 열지 못함) | §4.2 p.3–4; §5.1 p.5; Table 1 p.6 | 방해 문서의 정의는 "semantically similar to the query but do not contain the correct answer"(검색 점수가 높은 오답 문서)로 원고와 맞음. 원문 표현은 "progressive accuracy degradation", "adding just one distracting document causes a sharp reduction". 그러나 Table 1은 문서를 0·1·2·4·6·…·18개로 넣어(하나씩이 아님) 단조 감소가 아님: Llama2 Far 4개 .2745→6개 .2898, Llama2 Near 10개 .3716→12개 .3991→14개 .4118, MPT Far 2개 .1913→4개 .2209. 조건: NQ-open train 질의 10K개, Llama2-7B·MPT-7B·Phi-2·Falcon-7B(4bit) | 부정확 | "Cuconasu et al.(2024)은 정답 문서에 질문과 의미가 가깝지만 답은 없는 방해 문서를 더하면 1개만 넣어도 정확도가 크게 떨어지고, 개수가 늘수록 대체로 더 떨어진다는 것을 보였다(NQ-open, 7B급 LLM 4개)." |
| E5 | 02_related.md:23 | CABINET이 "표 전체 중 작은 부분만 질문과 관련이 있고, 나머지는 방해 정보로서 잡음으로 작용한다"고 보고함 | Patnaik et al., 2024 (CABINET) | arXiv 2402.01155v3 (OpenReview 판은 열지 못함) | 초록 p.1; §1 p.1 | "Typically, only a small part of the whole table is relevant to derive the answer for a given question. The irrelevant parts act as noise and are distracting information". 내용은 일치하지만, 따옴표 안 번역에서 한정어 "Typically(대개)"가 빠짐. 또 이것은 실험 결과가 아니라 문제 설정의 전제 서술임(§1은 Kumar et al. 2023, Chen et al. 2023a를 근거로 듦) | 부정확 | (경미) "…CABINET(Patnaik et al., 2024)은 \"대개 표 전체 중 질문의 답을 끌어내는 데 관련된 부분은 작고, 관련 없는 부분은 잡음이자 방해 정보로 작용한다\"고 지적했다." |
| E6 | 02_related.md:23 | "Liu et al.(2024)은 검색 문서 수를 늘릴 때 리더 성능이 검색 재현율보다 훨씬 먼저 포화한다는 것을 보였다" | Liu et al., 2024 | TACL 판 | §5 p.166, Figure 11; §1 p.159 | "reader model performance saturates long before retriever performance saturates". 조건: NaturalQuestions-Open 중 긴 답이 문단인 부분집합, Contriever(MS-MARCO 미세조정) 검색, 문서 수 k를 늘림. 20개에서 50개로 늘려도 GPT-3.5-Turbo는 약 1.5%, Claude-1.3은 약 1%만 오름 | 정확 | — |
| E7 | 02_related.md:31 | "LLM 추론 비용 때문에 전체 평가 집합 대신 일부 질의만 평가하는 것은 표 질의응답 연구에서 흔하다" | 인용 없음 (뒤 4편으로 대조) | Deng Findings; TableLlama NAACL; RealHiTBench Findings; Wang v1 | Deng §3.3 p.410; TableLlama Table 2·3 주석 p.6029–6030; RealHiTBench 부록 D.1 p.7122; Wang §3 p.3 | 비용을 이유로 든 문헌: Deng "Considering the limited access to LLMs' APIs and the scale of the comparison", TableLlama "due to limited budget", RealHiTBench "Considering the high computational cost of GPT-o1". Wang은 500건을 뽑은 이유를 밝히지 않음 | 정확 | — |
| E8 | 02_related.md:31 | "Deng et al.(2024)은 100건에서" | Deng et al., 2024 | Findings of ACL 2024 (arXiv v5 p.4 같은 문장) | §3.3 p.410 | "we randomly select 100 examples from the test set for each of these datasets". 데이터셋 6개(FinQA·LogicNLG·TabFact·WikiTQ는 정확도, ToTTo·E2E는 ROUGE)마다 test 분할에서 무작위 100건을 뽑아 모든 LLM·MLLM을 평가함(총 54,000개) | 정확 | — |
| E9 | 02_related.md:31 | "McNemar 검정 등을 썼고" | Deng et al., 2024 | Findings of ACL 2024 (arXiv v5 p.13 같은 문장) | 부록 B p.418–419 | "three statistical significance tests, including Fisher's Exact test, McNemar's Test, and proportion Z test". McNemar 검정은 GPT-4의 텍스트·이미지 입력 비교(FinQA·TabFact·WikiTQ에서 유의)와 vanilla·expert 프롬프트 비교에 씀. 비교 하나가 데이터셋 하나의 100건에 해당함 | 정확 | — |
| E10 | 02_related.md:31 | "Wang et al.(2026)은 500건" | Wang et al., 2026 | arXiv 2603.15402v1 | §3 p.3, Table 1 | "We randomly select 500 samples from 3 TQA benchmarks and 1 TFV benchmark". WTQ·HiTab·AIT-QA·TabFact에서 각 500건, 모두 2,000건. 어느 분할에서 뽑았는지는 밝히지 않음. 지표는 정확도(부록 Table 4) | 정확 | — |
| E11 | 02_related.md:31 | "(500건에) 부트스트랩 신뢰구간을 붙였으며" | Wang et al., 2026 | arXiv 2603.15402v1 | §7.2 Table 3·각주 1 p.9; 부록 D.1 p.15, Table 5 p.16 | "95% bootstrap confidence intervals (1,000 iterations, N=2,000 per condition)". 신뢰구간은 벤치마크 4개를 합친 2,000건 기준이고, 모델 5개 × Markdown/HTML × Direct/CoT 비교에만 붙음. 벤치마크별 500건 결과(Table 4)에는 신뢰구간이 없음 | 부정확 | "Wang et al.(2026)은 벤치마크 4개에서 500건씩 뽑은 2,000건으로 평가하고, 입력 형식·추론 방식 비교에 2,000건 기준 95% 부트스트랩 신뢰구간(재표집 1,000회)을 붙였으며," |
| E12 | 02_related.md:31 | "TableLlama(Zhang et al., 2024) … 도 500건을 썼다" | Zhang et al., 2024 (TableLlama) | NAACL 2024 (arXiv v3 p.6·7 같은 문구) | Table 2 주석 p.6029; Table 3 주석 p.6030; §3 p.6028 | "for GPT-4, we uniformly sample 500 examples from test set for each task due to limited budget". 500건은 GPT-4 비교군에만 쓰였고, 과제마다 test에서 균등 추출함. TableLlama·GPT-3.5는 test 전체로 평가함(열 유형 주석·개체 연결·행 채우기만 test 크기 때문에 줄인 test를 씀, §3) | 부정확 | "TableLlama(Zhang et al., 2024)는 GPT-4 비교군에 한해 과제마다 test에서 500건을 뽑아 썼고," |
| E13 | 02_related.md:31 | "RealHiTBench(Wu et al., 2025a)도 500건을 썼다" | Wu et al., 2025a (RealHiTBench) | Findings of ACL 2025 | 부록 D.1 p.7122, Table 6; §3.5 p.7109; §5.1 p.7110 | "we randomly selected 500 samples from benchmark". 500건은 추론 모델(GPT-o1·DeepSeek-R1·QwQ-32B, CoT/TreeThinker) 부록 실험에만 쓰였고, 이유는 GPT-o1 비용과 DeepSeek 서버 불안정임. 본 실험(Table 2, 모델 25개)에는 부분 표본 언급이 없음(벤치마크는 표 708개·질문 3,752개) | 부정확 | "RealHiTBench(Wu et al., 2025a)는 비용이 큰 추론 모델 부록 실험에서 500건을 뽑아 썼다." |
| E14 | 02_related.md:33 | "질의 유형마다 정확도가 크게 다르므로" | 인용 없음 (TableBench로 대조) | AAAI 2025 | Table 3 p.25502; 평가 지표 p.25501 | 예: GPT-4-Turbo TCoT에서 사실 확인 75.92, 수치 추론 53.01, 데이터 분석 41.03, 시각화 62.00. 다만 점수는 ROUGE-L(시각화는 pass@1)이며 정확도가 아님 | 정확 | — |
| E15 | 02_related.md:33 | "유형별 값과 개수를 함께 싣는 것이 관행이다" | 인용 없음 (TableBench로 대조) | AAAI 2025 | Table 1 p.25499; Table 3 p.25502 | TableBench는 범주별 값은 싣지만 범주별 문항 수는 싣지 않음(Table 1에는 전체 "TableBench Size 886"과 하위 범주 18개의 이름만 있음). "값"은 뒷받침되고 "개수"는 뒷받침되지 않음 | 부정확 | "유형별 값을 따로 싣는 것이 관행이다(예: TableBench). 본 연구는 여기에 유형별 개수도 함께 싣는다."처럼 "개수"를 관행의 근거에서 떼어 낸다 |
| E16 | 02_related.md:33 | "TableBench(Wu et al., 2025b)는 유형별 정확도를 유형 크기로 가중 평균한다" | Wu et al., 2025b (TableBench) | AAAI 2025 | Table 3 캡션 p.25502; 평가 지표 p.25501 | "The overall results represent a weighted average of performance across different categories." 가중치(유형 크기)는 원문에 없음. 범주 점수도 ROUGE-L·pass@1이며 정확도가 아님. (참고: Table 3의 모델 29개 × 방법 3개 = 87행으로 역산한 가중치는 .108/.447/.388/.056, 886건으로 환산하면 약 96/396/343/50, 최대 잔차 0.18. 크기 가중과 맞지만 원문이 밝힌 내용은 아님) | 부정확 | "TableBench(Wu et al., 2025b)는 전체 점수를 범주별 점수의 가중 평균으로 보고한다(Table 3 주석, 가중치는 밝히지 않음)." "유형 크기"를 남기려면 가중치를 밝힌 근거(공식 코드 등)를 따로 확인해 인용한다 |
| E17 | 08_refs.md:9 | Lewis, P., Perez, E., Piktus, A., et al. (2020). … NeurIPS 2020, 33, pp. 9459–9474. | 서지 | NeurIPS 논문집 PDF + BibTeX | p.1; BibTeX | 저자 12명 중 앞의 3명이 일치함. 제목 일치. booktitle "Advances in Neural Information Processing Systems", volume 33, pages 9459--9474, 2020 | 정확 | — |
| E18 | 08_refs.md:6 | Cuconasu, F., Trappolini, G., Siciliano, F., et al. (2024). … SIGIR 2024, pp. 719–729. | 서지 | arXiv v4 (ACM DL 판은 열지 못함) + Crossref DOI 메타데이터 | arXiv v4 p.1 ACM Reference Format | 저자 8명 중 앞의 3명 일치. 제목 일치. SIGIR '24(47th), DOI 10.1145/3626772.3657834, "11 pages". 쪽 719–729는 Crossref로 확인했고 11쪽 분량과 맞음. ACM DL 판에서 직접 확인하지는 못함 | 정확 | — |
| E19 | 08_refs.md:13 | Patnaik, S., Changwal, H., Aggarwal, M., Bhatia, S., Kumar, Y., & Krishnamurthy, B. (2024). CABINET: Content Relevance based … ICLR 2024. arXiv:2402.01155. | 서지 | arXiv v3 + arXiv 초록 페이지 + OpenReview 검색 API 기록 | p.1; arXiv Comments | 저자 6명 일치(논문 본문 표기는 "Yaman Kumar Singla", arXiv·OpenReview 메타데이터는 "Yaman Kumar"). arXiv 판 "Accepted at ICLR 2024 (spotlight)". 제목은 arXiv 판이 "Content Relevance based", ICLR(OpenReview) 판이 "Content Relevance-based"로 하이픈만 다름 | 정확 | — |
| E20 | 08_refs.md:11 | Liu, N. F., Lin, K., Hewitt, J., et al. (2024). … TACL, 12, 157–173. | 서지 | TACL 판 + Anthology 메타데이터 | p.1 | 저자 7명 중 앞의 3명 일치. TACL 12권 157–173쪽, DOI 10.1162/tacl_a_00638 | 정확 | — |
| E21 | 08_refs.md:7 | Deng, N., Sun, Z., He, R., et al. (2024). … Findings of ACL 2024. arXiv:2402.12424. | 서지 | Findings 판 + arXiv 초록 페이지 | p.1 | 저자 8명 중 앞의 3명 일치. 제목 일치. Findings of ACL 2024, pp. 407–426(원고에는 쪽이 없음). arXiv 판은 v1–v5가 있고 최신은 v5 | 정확 | — |
| E22 | 08_refs.md:18 | Wang, J., Qin, C., Zheng, M., Si, Q., Li, P., & Lin, Z. (2026). … arXiv:2603.15402. | 서지 | arXiv v1 | p.1; arXiv 초록 페이지 | 저자 6명의 이름과 순서가 모두 일치. 제목 일치. v1(2026-03-16)이 유일한 버전 | 정확 | — |
| E23 | 08_refs.md:24 | Zhang, T., Yue, X., Li, Y., & Sun, H. (2024). TableLlama … NAACL 2024. arXiv:2311.09206. | 서지 | NAACL 판 + arXiv 초록 페이지 | p.1 | 저자 4명 일치. 제목 일치. NAACL 2024 Long, pp. 6024–6044(원고에는 쪽이 없음). arXiv 판 최신은 v3, "NAACL 2024 long paper" | 정확 | — |
| E24 | 08_refs.md:19 | Wu, P., Yang, Y., Zhu, G., et al. (2025a). RealHiTBench … Findings of ACL 2025, pp. 7105–7137. | 서지 | Findings 판 + Anthology 메타데이터 | p.1 | 저자 13명 중 앞의 3명 일치. 제목 일치. pp. 7105–7137 | 정확 | — |
| E25 | 08_refs.md:20 | Wu, X., Yang, J., Chai, L., et al. (2025b). TableBench … AAAI 2025, 39(24), pp. 25497–25506. | 서지 | AAAI OJS PDF + 논문 페이지 메타데이터 | p.1; OJS 메타데이터 | 저자 13명 중 앞의 3명 일치. 제목 일치. Proceedings of the AAAI Conference on AI 39권 24호, 25497–25506쪽, DOI 10.1609/aaai.v39i24.34739 | 정확 | — |
| F1 | 03_method.md:77 | "`BAAI/bge-base-en-v1.5`(Xiao et al., 2024)" | Xiao et al., 2024 | arXiv v5 + 모델 카드 | 논문 1쪽 초록, 8쪽 표 5; 모델 카드 Evaluation(MTEB 표), Citation | 논문: "we release our data and models for English text embeddings"(1쪽). 표 5 "BGE (base) 768 63.55 53.25 …"가 모델 카드 bge-base-en-v1.5 행과 여덟 값 모두 같음. 모델 카드 Citation 블록이 arXiv 2309.07597을 인용 요청. 논문 본문에 "v1.5"라는 이름은 없음 | 정확 | — |
| F2 | 03_method.md:77 | "검색용 지시문(\"Represent this sentence for searching relevant passages: \")" | Xiao et al., 2024 (사용법은 모델 카드) | 모델 카드 | Model List 표 "query instruction for retrieval" 열 | bge-base-en-v1.5 행: "`Represent this sentence for searching relevant passages: `". 논문 3.4절(6쪽)에는 예시 "search relevant passages for the query"만 있고 이 문자열은 없음 | 정확 | — |
| F3 | 03_method.md:77 | "이 모델의 사용법대로 질문 앞에만 … 붙이고 문장에는 붙이지 않는다" | Xiao et al., 2024 (사용법은 모델 카드) | 모델 카드 + arXiv v5 | 모델 카드 Model List 각주 [1]; 논문 3.4절 6쪽 | 카드: "we suggest to add the instruction to the query … In all cases, no instruction needs to be added to passages." 논문: "a task specific instruction 𝐼𝑡 is attached to the query side" | 정확 | — |
| F4 | 03_method.md:77 | "벡터는 정규화해 코사인 유사도를 쓴다" | 인용 없음(모델 카드 사용법) | 모델 카드 | Usage > Using Langchain, Using Sentence-Transformers | "encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity" | 정확 | — |
| F5 | 03_method.md:77 | "입력 한도(512토큰)" | 인용 없음(모델 카드) | 모델 카드 + sentence_bert_config.json | Evaluation MTEB 표 "Sequence Length" 열; sentence_bert_config.json | bge-base-en-v1.5 행 Sequence Length "512"; config "max_seq_length": 512. 논문에는 이 한도 서술 없음 | 정확 | — |
| F6 | 03_method.md:79 | "BM25(Robertson and Zaragoza, 2009) 색인" | Robertson & Zaragoza, 2009 | 저자 공개본 | 3.4절 식 (3.15), 360쪽 | "This is the classic BM25 term-weighting and document-scoring function." | 정확 | — |
| F7 | 03_method.md:95 | "`Qwen2.5-7B-Instruct`(Qwen Team, 2024)" | Qwen Team, 2024 | arXiv v2 | 1쪽 초록; 5.2절 13쪽, 표 8(12쪽) | "instruction-tuned models in sizes of 0.5B, 1.5B, 3B, 7B, …"(1쪽); "the Qwen2.5-7B-Instruct model significantly outperforms its competitors"(13쪽). 4비트 양자화는 원고 자체 설정이며 보고서에 기대는 주장 아님 | 정확 | — |
| F8 | 03_method.md:96 | "`Qwen3-8B`(Qwen Team, 2025)" | Qwen Team, 2025 | arXiv v1 | 2절 3쪽, 표 1 | "6 dense models, namely Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B, …" | 정확 | — |
| F9 | 03_method.md:96 | "비생각(non-thinking) 모드" | Qwen Team, 2025 | arXiv v1 | 1쪽 초록; 4.3절 11쪽 | "integration of thinking mode … and non-thinking mode (for rapid, context-driven responses) into a unified framework"; "/think and /no think flags … retain an empty thinking block" | 정확 | — |
| F10 | 01_intro.md:28 | "Holm(1979) 보정 p=.0040" (Holm 인용 부분만) | Holm, 1979 | 본문 못 엶 | — | JSTOR 본문 유료. 참고로 JSTOR 공식 초록(RIS)만 확인: "a simple and widely applicable multiple test procedure of the sequentially rejective type". 초록만으로는 판정하지 않음 | 확인 불가 | 합법적 본문(도서관 JSTOR 접속 등) 확보 후 재대조 |
| F11 | 04_setup.md:69 | "정확 McNemar 검정(McNemar, 1947 …)" — 검정의 출처 | McNemar, 1947 | 본문 못 엶 | — | Cambridge Core 본문 유료. 출판사 초록만 확인 | 확인 불가 | 합법적 본문 확보 후 재대조 |
| F12 | 04_setup.md:69 | "(McNemar, 1947; 불일치 쌍에 대한 양측 이항 검정)" — 정확 형태를 1947 논문에 귀속 | McNemar, 1947 | 본문 못 엶 | — | 출판사 초록: "Two formulas are presented for judging the significance … The chi square equivalent of one of the developed formulas is pointed out." 초록상 원 논문은 근사 공식·카이제곱 형태이고, 정확 이항 형태가 들어 있는지는 본문 없이는 알 수 없음 | 확인 불가 | 본문 확보 전에는 귀속을 좁혀 쓴다: "McNemar(1947) 검정의 정확 형태(불일치 쌍에 대한 양측 이항 검정)로 판정한다." |
| F13 | 04_setup.md:69 | "여러 비교를 함께 판정하는 곳은 비교 묶음 단위 Holm 보정값을 함께 적는다" | 인용 없음 (Holm 1979는 01_intro.md:28에만 인용) | 본문 못 엶 | — | Holm 본문 미열람(F10과 같음) | 확인 불가 | 방법을 정의하는 이 줄 첫 언급에 "(Holm, 1979)" 추가 |
| F14 | 04_setup.md:75 | "continuous batching(끝난 요청 자리에 다음 요청을 바로 넣는 생성 방식)" | 인용 없음(transformers 문서) | transformers v5.12.1 문서 원본 | continuous_batching.md 19행(첫 문단); continuous_batching_architecture.md 21행 | "As requests finish, new ones join immediately instead of waiting for the whole batch to complete."; "the scheduler checks for finished requests and replaces them immediately with waiting ones" | 정확 | — |
| F15 | 04_setup.md:75 | "이 방식은 한 번에 한 건씩 생성하는 방식과 부동소수 연산 순서가 달라 출력 문자열이 바이트 단위로 같지 않다" | 인용 없음(transformers 문서) | transformers v5.12.1 문서 원본 3개 | continuous_batching.md, continuous_batching_architecture.md, serve-cli/serving_optims.md 전체 | 세 문서에서 determin·identical·floating·numeric·precision·differ·reproduc·invarian 검색: 출력이 한 건씩 생성과 달라진다거나 그 원인이 연산 순서라는 서술 없음. 바이트 불일치 자체는 원고 자체 측정(부록 C, 문자열 일치 72/120)이고, 원인은 측정한 것이 아님 | 원문에 없음 | 원인 단정을 빼고 관측만 적는다: "이 방식의 출력은 한 건씩 생성한 출력과 바이트 단위로 같지 않았다(문자열 일치 72/120, 부록 C)." 같은 원인 문구가 있는 09_appendix.md:36("부동소수 연산 순서가 달라")도 같이 고친다 |
| F16 | 06_discussion.md:73 | "교차 인코더(bge-reranker-v2-m3)" | 인용 없음(모델 카드) | bge-reranker-v2-m3 모델 카드 + bge-base-en-v1.5 모델 카드 | reranker 카드 첫 문단, Usage > Using Huggingface transformers; bge 카드 Model List 각주 [2] | reranker 카드: "reranker uses question and document as input and directly output similarity instead of embedding"(질문·문서 쌍을 한 입력으로 AutoModelForSequenceClassification에 넣음). bge 카드: "cross-encoder is widely used to re-rank top-k documents" | 정확 | — |
| F17 | 08_refs.md:21 | Xiao, Liu, Zhang, Muennighoff, Lian, Nie (2024). C-Pack: Packed Resources For General Chinese Embeddings. SIGIR 2024. arXiv:2309.07597 | 서지 | arXiv v5 + Crossref DOI 레코드 | arXiv v5 1쪽 ACM Reference Format; arXiv 초록 페이지 | 저자 6명·순서, 제목(v5 제목), SIGIR '24, 2024, arXiv 번호 모두 일치. SIGIR 논문집 쪽 641–649(Crossref)는 목록에 없음(같은 목록 Cuconasu SIGIR 항목은 쪽을 적음). 모델 카드 Citation 블록의 옛 제목 "Packaged Resources To Advance General Chinese Embedding"(v1)과는 다르나 목록은 최신·학회판 제목을 따름 | 정확 | — |
| F18 | 08_refs.md:16 | Robertson & Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond. FnTIR, 3(4), 333–389 | 서지 | 저자 공개본 | 1쪽(표지) | "Vol. 3, No. 4 (2009) 333–389 … DOI: 10.1561/1500000019 … By Stephen Robertson and Hugo Zaragoza" | 정확 | — |
| F19 | 08_refs.md:14 | Qwen Team: Yang, A., Yang, B., Zhang, B., et al. (2024). Qwen2.5 Technical Report. arXiv:2412.15115 | 서지 | arXiv v2 | arXiv 초록 페이지 저자 목록; PDF 1쪽; 7절 18쪽 | arXiv 저자 "Qwen: An Yang, Baosong Yang, Beichen Zhang, …", PDF 저자 표기 "Qwen Team", v1 2024-12-19 | 정확 | — |
| F20 | 08_refs.md:15 | Qwen Team: Yang, A., Li, A., Yang, B., et al. (2025). Qwen3 Technical Report. arXiv:2505.09388 | 서지 | arXiv v1 | arXiv 초록 페이지; PDF 1쪽; 6절 | arXiv 저자 "An Yang, Anfeng Li, Baosong Yang, …", PDF "Qwen Team", 2025-05-14 | 정확 | — |
| F21 | 08_refs.md:8 | Holm (1979). A Simple Sequentially Rejective Multiple Test Procedure. Scand. J. Stat., 6(2), 65–70 | 서지 | JSTOR 서지 레코드(본문 미열람) | jstor.org/citation/ris/4615733 | "TI A Simple Sequentially Rejective Multiple Test Procedure; AU Holm, Sture; VL 6; IS 2; SP 65; EP 70; PY 1979" | 정확 | — |
| F22 | 08_refs.md:12 | McNemar (1947). Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages. Psychometrika, 12(2), 153–157 | 서지 | Cambridge Core 기사 페이지 메타데이터 + Crossref(본문 미열람) | citation_* 메타태그; DOI 10.1007/BF02295996 | 제목 일치, 저자 Quinn McNemar, 1947/06, 권 12, 호 2, 153–157쪽 | 정확 | — |
| G1 | 01_intro.md:7; 03_method.md:17 | "값 `52.1`은 '지역 = 동부 온타리오 > 프랑스어 사용 노동자', '산업 = 퍼센트 > 음식 서비스'라는 두 경로" / 표 3-1 "행 경로 percent > food service, 열 경로 eastern ontario > french-language workers" | 인용 없음(HiTab 데이터) | HiTab 공식 저장소 로컬 사본 github.com/microsoft/HiTab 커밋 d179602(묶음 A의 [A:R]과 같은 커밋) | data/test_samples.jsonl id c26dd7d31e73948ab9deb29ca1fc89ef; data/tables/raw/100.json texts 6행 1열, left_root·top_root | 질문 "in eastern ontario, what percent of french-language workers have worked in the restaurant and food services sector?", 답 [52.1]; 트리 경로 left "percent > food service", top "eastern ontario > french-language workers"; 표 제목 "agri-food industry sub-groups … two agricultural regions of ontario, 2011" | 정확 | — ("지역 =", "산업 ="은 원고가 붙인 설명어이며 원자료에는 없다. 'percent'는 HiTab 트리에서 단위 머리글이 행 트리의 상위 노드로 들어간 것이다.) |
| G2 | 01_intro.md:9 | "흔한 방식은 표를 마크다운이나 쉼표 구분 텍스트로 펴는 것이다" | 인용 없음 | (참고) Zhou et al., 2026 arXiv:2510.09671v2 — 묶음 A의 [A:Z] | 3.1절(Complex Table 절 끝) | "When using more fine-grained textual representations such as JSON or Markdown, there seems to be no optimal format" — 텍스트 표현의 예로 JSON·Markdown을 들 뿐, 마크다운·쉼표 구분 텍스트가 '흔한 방식'이라는 서술은 없다 | 원문에 없음 | 인용할 근거를 붙이거나(예: "표를 JSON·마크다운 같은 텍스트로 바꿔 넣는다(Zhou et al., 2026, 3.1절)"), "흔한"이라는 일반화를 빼고 본 연구의 비교군(고정 청크·표 전체)이 마크다운 직렬화를 쓴다는 사실로 한정한다. |

## 4. 연 판본

### 묶음 A. HiTab(Cheng et al., 2022)·표 QA 서베이(Zhou et al., 2026)

- **[A:H]** Cheng et al. (2022) HiTab, ACL Anthology 논문집 판 2022.acl-long.78, https://aclanthology.org/2022.acl-long.78.pdf (Proceedings of the 60th Annual Meeting of the ACL, Volume 1: Long Papers, pp. 1094–1110, 17쪽). 원문 파일: src/A_hitab.pdf / .txt. 아래 쪽 번호는 논문집 인쇄 쪽(PDF 1쪽 = p.1094).
- **[A:Z]** Zhou et al. (2026) 서베이, arXiv:2510.09671 **v2**(2026-04-18, 최신), https://arxiv.org/pdf/2510.09671v2 (29쪽). 원문 파일: src/A_zhou.pdf / .txt. 대조용으로 ACL 2026 논문집 판 2026.acl-long.557(https://aclanthology.org/2026.acl-long.557.pdf, pp. 12161–12189, src/A_zhou_acl.pdf)도 열어 아래 인용 부분이 두 판에서 같음을 확인했다. 쪽은 "arXiv PDF 쪽 / ACL 인쇄 쪽"으로 적는다.
- **[A:R]** HiTab 공식 저장소 github.com/microsoft/HiTab, 커밋 d179602662b490249baf068a76fbe4137029126e (2025-12-16). README, data/*_samples.jsonl, data/tables.zip, qa/table/utils.py, qa/datadump/utils.py, qa/table/experiments.py 를 이 커밋에서 직접 받아 확인(src/A_repo/). 로컬 사본(rag-agent/data/hitab)도 같은 커밋이며 test_samples.jsonl MD5 일치(dafdb896…).
- **[A:T]** ToTTo 공식 데이터 https://storage.googleapis.com/totto-public/totto_data.zip. 로컬 사본(/mnt/c/.../recheck_20260913/sources/totto_data.zip)의 MD5 6e3418e92745df6bd7fe3a535011df2e 가 공식 파일 ETag와 같아 공식 파일로 보고 사용(03_method.md:37 확인용).

### 묶음 B. MultiHiertt(Zhao et al., 2022)·공식 저장소

- [P] 논문집 PDF: https://aclanthology.org/2022.acl-long.454.pdf — Proceedings of the 60th Annual Meeting of the ACL, Volume 1: Long Papers, pp. 6588–6600 (PDF 13쪽; 아래 쪽 번호는 논문집 인쇄 쪽 = PDF 쪽 + 6587). 텍스트: src/B_multihiertt.txt
- [B] 논문집 BibTeX: https://aclanthology.org/2022.acl-long.454.bib (src/B_anthology.bib)
- [R] 공식 저장소 github.com/psunlpgroup/MultiHiertt, main 최신 커밋 45bd9ccdf3142ea059bd5e69c0afb83437fa539c (2024-10-22): README.md, evaluate.py, utils/utils.py, utils/span_selection_utils.py (src/B_repo_*)
- [H] 저자 Hugging Face 저장소 datasets/yilunzhao/MultiHiertt 커밋 f18473da528dede3d9ce2274366d9ee8102ea0fd (README L51이 가리키는 저장소, 원고 각주 [^1]의 커밋): multihiertt_data/{train,dev,test}.json, table_description_generation.py (src/B_hf_*). test.json sha256 = 15bfe9cc1241e29050a5bcaf7b9639d6907e7895cd07a1f1b874c2fda905ad01 (각주 값과 일치). README의 Google Drive 판은 열지 않음.

### 묶음 C. TableRAG(Chen et al., 2024)·RowCol·RandRow 재구현

- **[C:N] NeurIPS 논문집 판(우선 판본)**: https://proceedings.neurips.cc/paper_files/paper/2024/file/88dd7aa6979e352fda7c4952ca8eac59-Paper-Conference.pdf
  (Advances in Neural Information Processing Systems 37, pp. 74899–74921, DOI 10.52202/079017-2382, 23쪽). 아래 쪽 번호는 이 PDF의 쪽 번호다(논문집 쪽 = 74898 + PDF 쪽).
  저장 위치: src/C_TableRAG_neurips.pdf / .txt
- **[C:v3] arXiv:2410.04739v3** (2024-12-26), https://arxiv.org/pdf/2410.04739v3 — [C:N]과 본문·표·수치가 같다(글자 사이 띄어쓰기와 arXiv 날인만 다름, 쪽 구성 같음).
- **[C:v1] arXiv:2410.04739v1** (2024-10-07, 17쪽) — 본문·표 수치(B, K, 표 2·3)는 같다. 차이: 초록 끝의 코드 URL 문장, 감사의 글, NeurIPS 체크리스트가 없고, 4.8절이 "Table 6 compares different retrieval approaches"로 표 번호를 잘못 적었다(v3·[C:N]은 Table 5). v2는 2024-12-24 제출(크기 같음, 열지 않음).
- **[C:G] 공식 코드**: github.com/google-research/google-research/tree/master/table_rag. 원고 코드 주석이 적은 커밋 08a8d6736475776f42ffac23b2c13111a28e5795(2026-09-10)과, 내려받은 5b09c22d73a9d35eb6c5d2a99b95677a45053466(table_rag 폴더를 마지막으로 건드린 커밋, 2026-02-12)의 agent/retriever.py·agent/agent.py·agent/rag_agent.py·utils/utils.py는 라이선스 머리말을 뺀 내용이 같다. 첫 공개 8369f2d(2024-11-10). 저장 위치: src/C_table_rag_code/
- **원고 코드**(읽기만, 저장소 HEAD 2dd7e9e): scripts/retrieval_accuracy.py, rag_agent/serialization/tablerag_unit.py, scripts/mh_arms.py

### 묶음 D. TableRAG(Yu et al., 2025)·MixRAG·Cao·ITR·TAP4LLM

| 약칭 | 연 판본(URL·버전) | 로컬 파일(scratchpad/cite/src/) |
|---|---|---|
| Yu 2025 | EMNLP 2025 논문집 PDF https://aclanthology.org/2025.emnlp-main.710.pdf (pp. 14063–14082). 대조용: arXiv:2506.10380 v1(2025-06-12)·v2(2025-09-30), 청크 문장 세 판본 모두 같음 | D_yu_emnlp.pdf/.txt, D_yu_v1/.v2 |
| Yu 코드 | 논문이 링크한 https://github.com/yxh-y/TableRAG , 커밋 f8d798f(2025-07-23) | D_yu_code/ |
| MixRAG | ACM DL(https://dl.acm.org/doi/10.1145/3770854.3780171)은 403(Cloudflare 차단)으로 못 엶. 저자 공개본 arXiv:2504.09554 **v3**(2025-12-16, "Accepted to SIGKDD 2026", ACM Reference Format·DOI·"V.1 (KDD 2026) … 12 pages" 포함)으로 대신함. 쪽 번호는 Crossref DOI 메타데이터(api.crossref.org/works/10.1145/3770854.3780171)로 확인 | D_mixrag_v3.pdf/.txt |
| Cao 2026 | arXiv:2602.01969 **v2**(2026-06-24, 목록에 버전 없어 최신판). 1쪽 각주 "Proceedings of the 43rd ICML … PMLR 306, 2026". PMLR v306 페이지는 404(미게시) | D_cao_v2.pdf/.txt |
| ITR | ACL 2023 논문집 PDF https://aclanthology.org/2023.acl-long.551.pdf (pp. 9909–9926) | D_itr.pdf/.txt |
| TAP4LLM | Findings of EMNLP 2024 PDF https://aclanthology.org/2024.findings-emnlp.603.pdf (pp. 10306–10323), 서지용으로 arXiv:2312.09039 v3(2024-10-10) | D_tap4llm_findings.*, D_tap4llm_v3.* |
| 원고 코드 | /home/user/T2-1/rag-agent: scripts/retrieval_accuracy.py(trag_hetero_chunks L223–251, budget_select L459), rag_agent/serialization/chunks.py, scripts/mh_arms.py; 결과 메타 results/rerun_20260926/{hitab,mh}/…trag_hetero.json(읽기만) | — |

### 묶음 E. RAG·방해 정보·평가 관행(Lewis, Cuconasu, CABINET, Liu, Deng, Wang, TableLlama, RealHiTBench, TableBench)

| 약칭 | 연 판본 (URL, 버전) | 로컬 파일 (scratchpad/cite/src/) |
|---|---|---|
| Lewis 2020 | NeurIPS 2020 논문집 PDF https://proceedings.neurips.cc/paper_files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf (PDF 1쪽 = 논문집 9459쪽), 논문집 BibTeX | E_lewis_neurips.pdf/.txt, E_lewis.bib |
| Cuconasu 2024 | **ACM DL 판(https://dl.acm.org/doi/10.1145/3626772.3657834)은 curl·WebFetch 모두 403(Cloudflare)으로 못 열어 arXiv 판으로 대신함**: arXiv 2401.14887v4 (2024-05-01, SIGIR '24 머리글·ACM Reference Format "11 pages" 포함). 쪽 번호는 Crossref DOI 메타데이터(https://api.crossref.org/works/10.1145/3626772.3657834)로 확인 | E_cuconasu_arxiv_v4.pdf/.txt |
| CABINET 2024 | **OpenReview 판(https://openreview.net/forum?id=SQrHpTllXa)은 PDF·API 모두 챌린지(403)로 못 열어 arXiv 판으로 대신함**: arXiv 2402.01155v3 (2024-02-13, 1쪽 머리글 "Published as a conference paper at ICLR 2024"). OpenReview 검색 API 기록(forum SQrHpTllXa, venue "ICLR 2024 spotlight")으로 학회·제목 확인 | E_cabinet_arxiv_v3.pdf/.txt |
| Liu 2024 | TACL 판 https://aclanthology.org/2024.tacl-1.9.pdf (PDF 1쪽 = 157쪽) + 논문 페이지 메타데이터 | E_liu_tacl.pdf/.txt |
| Deng 2024 | Findings of ACL 2024 https://aclanthology.org/2024.findings-acl.23.pdf (pp. 407–426) + arXiv 2402.12424v5 (2024-10-17, 최신) — 대조한 문장은 두 판이 같음 | E_deng_findings.*, E_deng_arxiv_v5.* |
| Wang 2026 | arXiv 2603.15402v1 (2026-03-16, 유일한 버전) | E_wang_arxiv_v1.pdf/.txt |
| TableLlama 2024 | NAACL 2024 Long https://aclanthology.org/2024.naacl-long.335.pdf (pp. 6024–6044) + arXiv 2311.09206v3 (2024-04-04, 최신) — 대조한 문장은 두 판이 같음 | E_tablellama_naacl.*, E_tablellama_arxiv_v3.* |
| RealHiTBench 2025a | Findings of ACL 2025 https://aclanthology.org/2025.findings-acl.371.pdf (pp. 7105–7137) | E_realhitbench_findings.* |
| TableBench 2025b | AAAI 2025 OJS https://ojs.aaai.org/index.php/AAAI/article/download/34739/36894 (39(24), pp. 25497–25506) + 논문 페이지 메타데이터 | E_tablebench_aaai.pdf/.txt, E_aaai_34739.html |

### 묶음 F. 임베딩·BM25·리더 모델·통계 검정·도구(C-Pack, BM25, Qwen2.5, Qwen3, Holm, McNemar, bge-reranker, transformers)

| 약칭 | 연 판본 (URL·버전) | 로컬 파일 |
|---|---|---|
| C-Pack | arXiv:2309.07597v5 (2024-09-24, 최신; v1–v5 있음) https://arxiv.org/pdf/2309.07597v5 — 1쪽에 SIGIR '24 ACM Reference Format·DOI 10.1145/3626772.3657878 표기. ACM DL PDF는 403으로 못 열어 arXiv 판으로 대신함. SIGIR 쪽 번호는 Crossref DOI 레코드(api.crossref.org/works/10.1145/3626772.3657878)로 확인 | src/F_cpack.pdf/.txt |
| bge 모델 카드 | https://huggingface.co/BAAI/bge-base-en-v1.5 README.md (커밋 a5beb1e3, lastModified 2024-02-21, 2026-09-28 열람), sentence_bert_config.json, config.json | src/F_bge_readme.md |
| BM25 | 저자(Robertson) 소속 기관 공개본 https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf (출판사 조판본 inr-019, 59쪽 = 학술지 333–389쪽). nowpublishers 페이지는 Emerald로 넘어가 403 | src/F_bm25.pdf/.txt |
| Qwen2.5 | arXiv:2412.15115v2 (2025-01-03, 최신; v1 2024-12-19) | src/F_qwen25.pdf/.txt |
| Qwen3 | arXiv:2505.09388v1 (2025-05-14, 유일 버전) | src/F_qwen3.pdf/.txt |
| Holm 1979 | 본문 못 엶(JSTOR 유료·봇 차단, OpenAlex oa_status=closed). 서지만 JSTOR 인용 내보내기 https://www.jstor.org/citation/ris/4615733 로 확인 | — |
| McNemar 1947 | 본문 못 엶(Cambridge Core 유료, PDF 링크가 초록 페이지로 돌아감). 서지·출판사 초록만 Cambridge Core 기사 페이지 메타데이터와 Crossref DOI 10.1007/BF02295996 레코드로 확인 | src/F_mcnemar_crossref.json, src/F_mcnemar_abs.html |
| bge-reranker 모델 카드 | https://huggingface.co/BAAI/bge-reranker-v2-m3 README.md (커밋 953dc6f6, lastModified 2024-06-24, 2026-09-28 열람) | src/F_reranker_readme.md |
| transformers 문서 | 설치본과 같은 태그 v5.12.1의 docs/source/en/continuous_batching.md, continuous_batching_architecture.md, serve-cli/serving_optims.md (raw.githubusercontent.com). 렌더링 페이지 https://huggingface.co/docs/transformers/main/en/continuous_batching 도 같은 문장 확인(main 커밋 07338b6, 2026-09-27) | src/F_tf_*.md |

## 5. 재구현 비교와 묶음별 비고

### 묶음 A

#### 비고 (판정 대상 아님)

- 04_setup.md:19 의 제외 3건(주석 없음 1·좌표 대응 불가 1·계산식 1), 1,245/336, 다중 셀 조회 38건, 셀 문장 67,664개는 본 연구가 직접 센 값이라 원문 대조 대상이 아니다. 참고로 [A:R] 원자료만으로 단순 재계산하면 [ANSWER] 머리글 340·다중 셀 37(원고 336·38), test 표 데이터 영역 셀 73,359개(빈칸 제외 66,763개)로, 제외 규칙과 색인 규칙 차이만큼 다르다.
- A16: HiTab 표 id `<묶음>_totto<N>-<k>` 를 ToTTo train N번째 줄에 대응시키는 규칙은 HiTab 논문·README 어디에도 적혀 있지 않다(본 연구가 추정한 규칙). [A:T]로 확인하면 1,851표 중 1,835표의 `title` 이 그 줄의 절 제목과 정확히 같아 규칙 자체는 맞다. 원고나 부록에 이 대응 규칙과 확인 결과를 한 줄 적어 두면 재현에 도움이 된다.
- A18: [A:R]에는 2025-10 품질 점검으로 test 231건(질문만 165, 답만 9, 둘 다 57)을 고친 test_samples_qualitycheck.jsonl 이 따로 있다. 원고는 원래 test_samples.jsonl(MD5 dafdb896…)을 썼으므로 4.1절에 사용 파일을 밝혀 두면 좋다.
- A21: [A:R] README 에 따르면 공개 데이터는 논문 이후 계층 추출 개선·QA 쌍 수정이 반영된 최신판이다("performance will be slightly higher than the results reported in the paper").

#### 판정별 개수

| 판정 | 개수 |
|---|---:|
| 정확 | 19 |
| 부정확 | 5 (A6, A9, A15, A22, A24) |
| 원문에 없음 | 0 |
| 확인 불가 | 0 |
| 합계 | 24 |

### 묶음 B

#### 재구현 차이
해당 없음. 이 묶음에는 재구현 비교군이 없다. 원고의 MultiHiertt 채점기 이식 코드는 대조 범위 밖이라 열지 않았고, 공식 채점 규칙 자체만 B18·B19에서 대조했다.

#### 판정별 개수
- 정확: 19 (B1, B4–B7, B11–B18, B20–B25)
- 부정확: 6 (B2, B3, B8, B9, B10, B19)
- 원문에 없음: 0
- 확인 불가: 0

### 묶음 C

#### 재구현 차이

| 항목 | 원 논문([C:N]) | 공식 코드([C:G]) | 원고 구현(코드) | 차이 | 원고가 밝히는가 |
|---|---|---|---|---|---|
| R1 LM 질의 확장 | 스키마용·셀용 질의를 LM이 따로 생성, 약 3~5개(3.3절 p.3–4, 3.4절 p.6, 부록 D·E p.16) | rag_agent.py retrieve_schema_by_prompt / retrieve_cell_by_prompt, 숫자 셀 질의 제거(71행) | 질문 원문 하나를 인코딩(bge 질의 접두어) | 있음 | 밝힘: 04_setup.md:43 (1), 05_results.md:50, 06_discussion.md:81 |
| R2 스키마 검색 | 별도 스키마 DB, 열 이름을 인코딩해 대조, 결과로 자료형·예시값 제공(p.4, 알고리즘 1 11–16행) | build_schema_corpus: `page_content=col_name`, 요약은 metadata(111행) | 요약 JSON 전체를 셀 문서와 한 색인에 넣고 같은 하이브리드 검색. 숫자 요약 문서는 최솟값·최댓값 셀을 전달, 범주형 스키마 문서는 셀 0개 | 경로 합침, 임베딩 텍스트 다름 | 경로 합침은 밝힘(04_setup.md:43 (2)); 임베딩 텍스트 차이는 안 밝힘(코드 docstring에만) |
| R3 열 이름-값 쌍 인코딩 | 서로 다른 열-값 쌍, 범주형 값만(p.2, p.4) | build_cell_corpus: `{"column_name","cell_value"}`, Counter 중복 합침; 숫자 열은 최솟값·최댓값 문서 하나(118–124행) | 같은 문자열·중복 합침. 숫자 요약 문서의 dtype을 "float64"로 고정. 행 라벨 문서 `{"column_name": <좌상단 머리글>, "cell_value": <행 잎 라벨>}` 추가(셀 0개 전달). leaf/path 열 이름 | 셀 문서 같음; dtype 표기·행 라벨 문서·leaf/path는 원고 적응 | leaf/path 밝힘(04_setup.md:39–40); 행 라벨 문서와 dtype 표기는 안 밝힘(표 3-1 leaf 예시가 행 라벨 문서인데 설명 없음) |
| R4 숫자 열 판정 | 명시 없음("convert columns to integer, float, or datetime data types when feasible", p.4) | utils.infer_dtype: `pd.to_numeric(errors='ignore')` 뒤 `to_datetime`; 쉼표 든 숫자 문자열·중복 열 이름(pandas 2.3.3에서 TypeError 확인)은 object로 남아 셀마다 색인 | `_is_numeric_column`: 쉼표 제거 후 비어 있지 않은 셀이 모두 숫자면 숫자 열, 날짜 판정 없음. 본문 결과는 `tablerag_dtype=infer`(results/rerun_20260926/hitab/*_tablerag_*.json; MH는 mh_arms.py 기본값 infer) | 판정 규칙 다름. HiTab 단일 셀 조회 991건: 정답 셀 890개가 접힌 열, 전달 가능 327건(상한 .3300, leaf·path 같음); all_object면 991건 모두 전달 가능 | 안 밝힘(원고 코드 주석만 "infer 는 그만큼 이 비교군을 과소평가한다. 둘 다 보고한다"라 적었고 본문은 infer만 보고) |
| R5 셀 빈도 예산 B | 빈도순 상위 B개 쌍만 인코딩, B=10,000(p.4, p.7) | `most_common(self.max_encode_cell - len(docs))`(124행) | tablerag_units는 B를 적용하지 않음(tablerag_unit.py의 기본값 10000은 이 경로에서 안 쓰임) | 실측: HiTab 표 최대 1,025셀(TableRAG 문서 최대 460개), MultiHiertt train 표 최대 602셀(`<td>` 수) → B가 걸리는 표 없음, 실질 차이 없음 | 안 밝힘(영향 없음) |
| R6 검색 개수 K와 예산 | 질의당 상위 K=5, 질의 3~5개 결과 합침(p.4 그림 2, p.6, p.7) | top_k=5 | 점수순 문서를 전달 셀이 서로 다른 20개에 이를 때까지 넣음(셀 0개 문서는 예산을 쓰지 않음), `--max-units` 안 씀 | 있음 | 밝힘(04_setup.md:29 "같은 셀 예산(20)", 3.5절) |
| R7 검색기·인코더 | text-embedding-3-large, 임베딩만 쓴 검색이 가장 높음(p.7, 표 5 p.8) | 기본 `retrieve_mode='embed'`, FAISS | BAAI/bge-base-en-v1.5 + BM25, α=0.7 | 있음 | 밝힘(04_setup.md:29, 06_discussion.md:81; 원 인코더 이름은 적지 않음) |
| R8 검색 범위 | 표마다 DB를 만들어 그 표 안에서 검색(알고리즘 1 2–3행) | init_retriever(table_id, df) | 표 5-2: 질문의 표 안(dense는 정확히 마스크, BM25 IDF는 코퍼스 전체). 표 5-3: 538개 표 한 색인이며, TableRAG 문서에는 표 제목이 없고 RowCol·RandRow·청크 단위에는 제목을 붙임 | 표 5-3은 원 설정 밖; TableRAG만 제목 없음 | 원 설정("큰 표가 주어짐")은 02_related.md:44에 밝힘; 538개 색인에서 TableRAG 문서만 표 제목이 없다는 점은 안 밝힘(표 3-1 예시에서만 보임) |
| R9 풀이기 | ReAct 기반 PyReAct, pandas 코드로 표 전체에 접근, 10회 다수결(p.5, p.7, 부록 F) | solver_loop(df, prompt) | 검색 결과만 Qwen 리더에 넣음 | 있음 | 밝힘(04_setup.md:43, 06_discussion.md:81) |
| R10 RowCol 색인 텍스트 | "encode rows and columns"만 적음(p.6) | 행·열 값을 `'\|'.join`(131·139행); 행은 평평하게 읽은 표라 행 라벨 값 포함, 열은 열 이름 없음 | 행 "제목 \| 행 잎 라벨\|값…", 열 "제목 \| 값…" | 표 제목 접두 추가 | "\|" 형식은 밝힘(04_setup.md:47, 다만 "원 논문처럼"은 코드가 출처); 제목 접두는 안 밝힘 |
| R11 RowCol 선택 | 상위 K 행 ∩ 상위 K 열 부분 표, K=30(p.6, p.7, 그림 1(c) p.2) | `self.df.iloc[row_ids, col_ids]`(162행) | 행·열 순위를 K=1부터 함께 늘려 교집합의 서로 다른 셀이 20개 이상이면 멈춤(최대 200, rowcol_select 547행) | K 고정 대신 셀 예산으로 K를 정함 | 안 밝힘: 본문에 교집합 서술이 없고, 3.5절은 누적 규칙이 "모든 방법에 똑같이 적용된다"고 적음 |
| R12 RowCol 행 절단 | 표를 B/2M 행으로 자름(p.6) | `max_row = max_encode_cell // 2 // len(columns)`(61행) | 자르지 않음 | 표 크기 최대 1,025셀 < 5,000(=B/2)이라 절단이 걸리는 표 없음 | 안 밝힘(영향 없음) |
| R13 RowCol 리더 입력 | 부분 표(그림 1(c)) | 부분 표 `to_markdown(index=False)`(agent.py:183) | "# 제목" + 열 잎 라벨 머리 줄 + 행 잎 라벨 열(subtable_context) | 제목·행 라벨 열 추가(작음) | 안 밝힘(표 3-1 예시에만 보임) |
| R14 RandRowSampling 선택 | 균등 확률로 K=30행(p.6, p.7) | `df.sample(n=self.top_k).sort_index()`(176행) | 질의 id 시드의 무작위 순열로 행을 넣어 서로 다른 셀 20개까지; 표 순서로 다시 정렬하지 않음 | 행 수 대신 셀 예산 | 밝힘(01_intro.md:28 "셀 예산까지 채우는") |
| R15 RandRow 리더 입력 | 표에서 뽑은 행(p.6) | 색인 없음, `sampled_table.to_markdown(index=False)`(열 이름 줄 있음, 179행) | "제목 \| 행 잎 라벨\|값…"(열 이름 없음, 03_method.md:69) | 리더가 열 이름을 받지 못함 → 답변 정확도(표 5-4 RandRow .1533)에 영향 가능, 측정 안 함 | 안 밝힘(04_setup.md:47은 "원 논문처럼 … 색인"이라 적음) |
| R16 RandRow와 검색기 | 검색 없음(p.6) | 검색 없음 | 검색 점수 안 씀(950행) | 차이 없음 | 01_intro.md:28은 "검색 없음"으로 밝힘; 04_setup.md:29 "모든 비교군은 … 같은 하이브리드 검색기"는 예외를 적지 않음(C17) |

#### 판정별 개수

- 정확: 23 (C1, C2, C4, C5, C6, C10, C11, C13, C14, C15, C18, C19, C20, C21, C22, C24, C26, C27, C29, C31, C32, C33, C35)
- 부정확: 13 (C3, C7, C8, C9, C12, C16, C17, C23, C25, C28, C30, C34, C36)
- 원문에 없음: 0
- 확인 불가: 0
- 합계: 36

#### 재현용 실측 명령 메모

- 전달 가능 상한: `retrieval_accuracy.load_queries("data/hitab","test",{})`의 단일 셀 조회 991건마다 `tablerag_units(tab, t, mode, dtype)`가 전달하는 셀 집합에 정답 셀이 있는지 셈. (leaf, infer) 327, (path, infer) 327, (leaf, all_object) 991. 숫자 열(`_is_numeric_column`)에 정답 셀이 있는 질의 890.
- 표 크기: HiTab `hg.table_ids("data/hitab")` 3,597표 중 최대 n_rows×n_cols = 1,025, 표당 TableRAG 문서 최대 460개. MultiHiertt(bevaya/MultiHiertt train 캐시) 표당 `<td>` 최대 602.
- pandas 2.3.3에서 중복 이름 열에 `pd.to_numeric`을 부르면 TypeError, 원 코드의 `except: pass`로 object 유지 확인.

### 묶음 D

#### 재구현 차이: TableRAG(Yu) 청크 비교군

| 항목 | 원 논문(EMNLP 2025) | 공식 코드(yxh-y/TableRAG @f8d798f) | 원고 코드 | 원고가 밝히는가(위치) |
|---|---|---|---|---|
| 1. 마크다운 변환 | "Markdown-rendered form of each table"(§3.2, p.14064). 세부 없음 | tool_utils.py:134–155 excel_to_markdown: 행마다 " \| a \| b \| ", **첫 행 뒤 "\| --- \|" 구분 줄**, 값이 None인 칸은 **건너뜀**(열 위치가 밀림), 워크북 시트 전부 이어 붙임 | chunks.py:10–26: 원 격자 행마다 "\| a \| b \|", **구분 줄 없음**, **빈 칸 유지**, 전부 빈 행은 건너뜀 | 안 밝힘(04_setup.md:36은 "마크다운 표"라고만 씀. 코드 주석에만 "explicit adaptation") |
| 2. 크기·겹침 단위 | 1,000**토큰**, 겹침 200**토큰**(§5.1.2, p.14067) | RecursiveCharacterTextSplitter(1000, 200), 길이 함수 기본값 → **글자** | 같은 클래스, 1,000자·200자(메타 chunk_measure=characters) | 밝힘(04_setup.md:49). 다만 02_related.md:15는 논문 인용 아래 글자 기준으로 적음(D14) |
| 3. "File name/Table name" 머리말 | 없음 | "File name: {파일 이름}"을 모든 청크 앞에, "Table name: {파일 이름에서 .xlsx 뺀 것}"을 표 맨 앞에(첫 청크에만) | 같은 형식·위치. 이름 값은 HiTab=표 제목(고유 라벨), **MultiHiertt=빈 문자열**(label_rule none; 재실행 기록 11,627개 문맥 청크 전부 첫 줄이 "File name: ") | 형식은 밝힘(04_setup.md:36, 표 3-1 예시 03_method.md:67). 이름 출처가 파일 이름이 아니라 표 제목이라는 점, MultiHiertt에서 이름이 비어 있다는 점, Table name이 첫 청크에만 붙는다는 점은 안 밝힘 |
| 4. 색인 대상 | 본문 T와 마크다운 표 D̂를 한 색인에(§3.2) | 표(xlsx)와 문서(json) 전부 | 표만(본문 문단 없음) | 밝힘(02_related.md:15 "청크 표현만", 04_setup.md:21 본문 근거 문항 제외) |
| 5. 임베딩 | BGE-M3(§5.1.2) | bge-m3, CLS 벡터, FAISS 내적 | bge-base-en-v1.5 + 하이브리드 α=0.7(본 방법과 같음) | 밝힘(04_setup.md 4.2절 첫 문단, 표 4-2 "전용 검색기 미재현") |
| 6. 임베딩 입력 절단 | 언급 없음 | 토크나이저 truncation=True(모델 한도) | 512토큰 한도, truncate. HiTab 질문의 표 안 색인 1,108개 청크 중 26개(2.3%) 절단, MultiHiertt 0개(embedding_input_audit) | 안 밝힘(06_discussion.md:83의 입력 절단 설명은 표 단위 색인만 다룸) |
| 7. 재정렬·전달 개수 | 상위 30개 → 재정렬 → 상위 3개(§5.1.2) | retrieve(q, 30, 5): bge-reranker-v2-m3로 상위 5개 → set()으로 중복 제거 → 앞 3개(main.py:169–171). 논문(3개)과 코드(5개에서 3개)가 조금 다름 | 재정렬 없음. 서로 다른 셀 20개가 찰 때까지 청크 단위로 전달(평균 전달 셀: HiTab 질문의 표 안 52.85, MultiHiertt 문서 안 45.6) | 밝힘(4.2절 "같은 셀 예산(20)", 결과표 전달 셀 칸) |
| 8. 질의 | LLM 질의 분해 → 하위 질의, 최대 5회 반복(§3.3, §5.1.2) | 처음 검색은 질문 + "The given table is in {table_id}"로 1순위 표를 고름(main.py:131–133). 그 뒤 하위 질의마다 검색 | 질문을 그대로 한 번 검색 | 따로 밝히지 않음("전용 검색 구성은 재현하지 않았다", "원 논문 시스템 전체와의 비교가 아니다"로만 포괄. 02_related.md:15, 06_discussion.md:81) |
| 9. SQL 경로 | 표를 MySQL에 적재, 청크→스키마 매핑, LLM이 SQL 생성·실행, 텍스트 검색 결과와 교차 확인(§3.2–3.3, p.14065–14066) | offline_data_ingestion_and_query_interface/(MySQL 8.0.24), sql_tool.py | 없음 | 밝힘(02_related.md:15, 04_setup.md:36, 06_discussion.md:81) |
| 10. 검색 범위 | 전체 문서 모음 | 전체 색인 | HiTab 질문의 표 안 / 538개 표 한 색인 / MultiHiertt 문서 안 | 밝힘(4장 조건 설명) |
| 11. 리더 | 백본 LLM(Claude-3.5-Sonnet, DeepSeek-V3 등)이 반복 추론 | 같음 | 본 방법과 같은 리더(Qwen3-8B) | 밝힘(4.2절 "같은 리더") |

#### 판정별 개수 (46행)

- 정확: 35
- 부정확: 10 (D2, D8, D11, D14, D16, D18, D26, D27, D43, D45)
- 원문에 없음: 1 (D34)
- 확인 불가: 0 (MixRAG는 ACM DL을 못 열어 저자 공개본 arXiv v3와 Crossref로 대신함)

### 묶음 E

재구현 차이: 이 묶음에는 재구현 비교군이 없어 해당 없음.

판정에서 뺀 부분 (이 묶음 문헌에 관한 주장이 아님):
- 01_intro.md:5 둘째 문장 "기업·기관의 보고서와 통계 자료는 상당 부분이 표로 되어 있으므로 … RAG의 주요 적용처다": 인용이 없는 저자 전제임. 이 묶음 9편 어디에도 이를 뒷받침하는 서술은 없음(참고로 적어 둠).
- 02_related.md:31 마지막 문장(본 연구의 seed 고정·id 목록 커밋), 02_related.md:33 마지막 문장(본 연구의 그룹별 보고·가중): 본 연구 자체에 관한 서술임.

#### 판정별 개수

| 판정 | 개수 | 번호 |
|---|---|---|
| 정확 | 16 | E1, E6, E7, E8, E9, E10, E14, E17–E25 |
| 부정확 | 8 | E3, E4, E5, E11, E12, E13, E15, E16 |
| 원문에 없음 | 1 | E2 |
| 확인 불가 | 0 | — |
| 합계 | 25 | |

### 묶음 F

재구현 비교군 없음(이 묶음은 해당 없음).

참고(판정 대상 아님): Qwen3 보고서 4.6절(13쪽)의 비생각 모드 평가 설정은 "temperature = 0.7, top-p = 0.8, top-k = 20"이다. 원고(03_method.md:96)는 greedy 디코딩을 쓰는데, 이는 보고서를 따른다고 주장하지 않으므로 인용 오류는 아니다.

#### 판정별 개수

- 정확: 17 (F1–F9, F14, F16–F22)
- 부정확: 0
- 원문에 없음: 1 (F15)
- 확인 불가: 4 (F10–F13)
- 합계: 22

## 6. (나) 인용 표기 없이 다른 연구·데이터셋·코드를 서술한 문장

원고 줄에 인용 표기가 없거나, 인용이 같은 문단의 다른 문장에만 있거나, 근거가 논문이 아니라 공식 코드·README·모델 카드인 행 60개다.

| 번호 | 위치 | 원고 문장 | 근거로 연 자료 | 구분 | 판정 |
|---|---|---|---|---|---|
| A9 | 02_related.md:5 | "본 연구는 이 계층 트리를 그대로 써서 셀의 머리글 경로를 만들고" | 인용 없음(데이터셋 [A:R]) | 문장에 인용 없음(같은 문단 다른 문장에만 인용) | 부정확 |
| A12 | 03_method.md:19 | "데이터셋이 표마다 행·열 머리글의 계층 트리를 준다" | 인용 없음(Cheng et al., 2022 데이터) | 줄에 인용 표기 없음 | 정확 |
| A13 | 03_method.md:19 | "병합 머리글 때문에 대응이 빠지는 열" | 인용 없음(데이터셋 [A:R]) | 줄에 인용 표기 없음 | 정확 |
| A14 | 03_method.md:37 | "HiTab의 표 제목" (데이터셋이 준 제목) | 인용 없음(데이터셋 [A:R]) | 줄에 인용 표기 없음 | 정확 |
| A15 | 03_method.md:37 | "데이터셋이 준 표의 절 제목 앞에" | 인용 없음(데이터셋 [A:R], [A:T]) | 줄에 인용 표기 없음 | 부정확 |
| A16 | 03_method.md:37 | "표의 원 출처인 위키백과 페이지의 제목(ToTTo 페이지 제목)이 있으면 그것을 붙인다" | 인용 없음(Cheng et al., 2022 데이터; ToTTo) | 줄에 인용 표기 없음 | 정확 |
| A17 | 04_setup.md:12 | "질의 수 1,584" (HiTab) | 인용 없음(데이터셋 [A:R]) | 줄에 인용 표기 없음 | 정확 |
| A18 | 04_setup.md:19 | "test 분할 1,584 질의를 쓴다" | 인용 없음(데이터셋 [A:R]) | 줄에 인용 표기 없음 | 정확 |
| A19 | 04_setup.md:19 | "HiTab test 분할의 표는 538개" | 인용 없음(데이터셋 [A:R]) | 줄에 인용 표기 없음 | 정확 |
| A20 | 04_setup.md:19 | "HiTab이 붙인 집계 라벨과 정답 근거 셀 수로 세 유형으로 나눈다" | 인용 없음(데이터셋 [A:R]) | 줄에 인용 표기 없음 | 정확 |
| A21 | 04_setup.md:55 | "HiTab은 데이터셋의 공식 채점 규칙 …" (공식 채점 코드의 존재·내용) | 인용 없음([A:R]) | 줄에 인용 표기 없음 | 정확 |
| A22 | 04_setup.md:55 | "…공식 채점 규칙을 이식한 채점기로" | 인용 없음([A:R] 대 원고 코드) | 줄에 인용 표기 없음 | 부정확 |
| B1 | 01_intro.md:15 | "질문과 짝지어진 … 문서(MultiHiertt)가 직접 주어진다" | 인용 없음(문장의 인용 Zhou 2026·Cheng 2022는 다른 묶음) | 문장에 인용 없음(같은 문단 다른 문장에만 인용) | 정확 |
| B9 | 03_method.md:21 | "표가 HTML로만 주어지고" | 인용 없음 | 줄에 인용 표기 없음 | 부정확 |
| B10 | 03_method.md:21 | "머리글 계층 정보가 없다" | 인용 없음 | 줄에 인용 표기 없음 | 부정확 |
| B11 | 04_setup.md:21 | "공개 test 파일(test.json, 1,566문항)" | 각주 [^1](HF 커밋) | 줄에 인용 표기 없음 | 정확 |
| B12 | 04_setup.md:21 | "모든 문항의 qa 항목에 질문(question)만 포함하며, 정답·풀이·정답 근거를 포함하지 않는다" | 각주 [^1](HF 커밋) | 줄에 인용 표기 없음 | 정확 |
| B13 | 04_setup.md:21 | "공식 저장소는 test 예측을 CodaLab 리더보드에 제출해 점수를 받도록 안내한다" | 공식 저장소 URL | 줄에 인용 표기 없음 | 정확 |
| B14 | 04_setup.md:21 | "7,830 질의"(train 분할) | 인용 없음 | 줄에 인용 표기 없음 | 정확 |
| B15 | 04_setup.md:21 | "본문 문장 근거가 필요한 4,922건 … 남은 2,908건" (범위 밖 보조 확인) | 인용 없음 | 줄에 인용 표기 없음 | 정확 |
| B16 | 04_setup.md:21 | "(이 구분은 공식 `question_type` 필드와 100% 일치한다)" | 인용 없음 | 줄에 인용 표기 없음 | 정확 |
| B17 | 04_setup.md:53 | "MultiHiertt 원 논문이 주 결과표를 운영점 하나로 보고하는 관행" | Zhao et al.(2022)(본문 표기만) | 줄에 인용 표기 없음 | 정확 |
| B18 | 04_setup.md:55 | "공식 `evaluate.py`의 문항 유형별 채점 규칙(조회는 문자열 비교, 산술은 수치 비교)" | 인용 없음(공식 코드) | 줄에 인용 표기 없음 | 정확 |
| B19 | 04_setup.md:55 | "공식 규칙상 정답이 음수인 산술 문항은 텍스트 답으로 맞힐 수 없는데" | 인용 없음(공식 코드) | 줄에 인용 표기 없음 | 부정확 |
| C12 | 02_related.md:13 | "TableRAG의 셀·스키마 텍스트 표현을 공식 코드와 대조해 이식하고(leaf, path 두 변형)" | Chen et al., 2024 (공식 코드) | 근거가 논문이 아니라 코드·README·모델 카드 | 부정확 |
| C16 | 04_setup.md:29 | "비교군의 텍스트 표현은 원 논문의 공개 코드와 대조해 이식했다." | Chen et al., 2024 (공식 코드) | 줄에 인용 표기 없음 | 부정확 |
| C17 | 04_setup.md:29 | "모든 비교군은 본 방법과 같은 … 하이브리드 검색기(α=0.7) … 를 쓴다" | Chen et al., 2024 | 줄에 인용 표기 없음 | 부정확 |
| C28 | 04_setup.md:47 | "두 데이터셋 모두 원 논문처럼 행(열)의 값을 \"\|\"로 이은 텍스트로 색인한다." | Chen et al., 2024 (논문·공식 코드) | 줄에 인용 표기 없음 | 부정확 |
| C29 | 04_setup.md:53 | "이는 TableRAG … 원 논문이 주 결과표를 운영점 하나로 보고하는 관행과 같다." | Chen et al., 2024 | 줄에 인용 표기 없음 | 정확 |
| C30 | 05_results.md:50 | "leaf와 path 표현은 (열 이름, 셀 값) 쌍만 담고 행 머리글을 담지 않는다. … 같은 열의 셀들이 열 이름과 값만으로는 구별되지 않는다." (낮은 이유로 제시) | Chen et al., 2024 (재구현) | 줄에 인용 표기 없음 | 부정확 |
| C31 | 05_results.md:50 | "이 표현은 원 논문이 다룬 평평한 대형 표를 위한 것이며" | Chen et al., 2024 | 줄에 인용 표기 없음 | 정확 |
| C32 | 05_results.md:50 | "원 논문의 LM 질의 확장, 별도 스키마 검색 경로, 풀이기를 함께 쓰지 않았다는 점(4.2절)" | Chen et al., 2024 | 줄에 인용 표기 없음 | 정확 |
| C33 | 05_results.md:68 | "RandRow는 질문의 표 안에서 행을 무작위로 고르는 방법이라" | Chen et al., 2024 | 줄에 인용 표기 없음 | 정확 |
| C34 | 06_discussion.md:7 | "(가장 큰 차이가 나는 비교군은 …) RandRow다. 이 표현들은 … 행 단위로만 색인해 상위 머리글이 빠진다(RandRow)." | Chen et al., 2024 | 줄에 인용 표기 없음 | 부정확 |
| C35 | 06_discussion.md:81 | "TableRAG 셀 검색 재구현에는 원 논문의 LM 질의 확장, 별도 스키마 검색 경로, 풀이기가 없고(4.2절)" | Chen et al., 2024 | 줄에 인용 표기 없음 | 정확 |
| D4 | 01_intro.md:28 | "TableRAG(Yu)의 청크 표현 … 같은 임베딩 모델, 같은 검색기, 같은 셀 예산 규칙(20셀)에 놓았다" | 원고 코드 | 근거가 논문이 아니라 코드·README·모델 카드 | 정확 |
| D16 | 02_related.md:15 | "본 연구는 이 중 청크 표현만 재현했다" | 원고 코드 vs 공식 코드 | 근거가 논문이 아니라 코드·README·모델 카드 | 부정확 |
| D17 | 02_related.md:15 | "SQL 경로와 전용 검색 구성은 재현하지 않았다" | 원고 코드 | 근거가 논문이 아니라 코드·README·모델 카드 | 정확 |
| D34 | 04_setup.md:35 | "고정 청크 \| 업계 관행" (1,000자, 제목·머리글 블록을 청크마다 반복) | 인용 없음 | 줄에 인용 표기 없음 | 원문에 없음 |
| D35 | 04_setup.md:36 | "\"File name/Table name\" 머리말" | Yu et al., 2025(공식 코드) | 근거가 논문이 아니라 코드·README·모델 카드 | 정확 |
| D36 | 04_setup.md:36 | "마크다운 표를 1,000자, 겹침 200자로 나눈 청크" | Yu et al., 2025(공식 코드) | 근거가 논문이 아니라 코드·README·모델 카드 | 정확 |
| D37 | 04_setup.md:36 | "청크 표현만. SQL 경로·전용 검색기 미재현" | 원고 코드 | 근거가 논문이 아니라 코드·README·모델 카드 | 정확 |
| D38 | 04_setup.md:49 | "원 논문 본문은 청크 크기를 '1,000토큰, 겹침 200토큰'이라 적지만" | Yu et al., 2025 | 줄에 인용 표기 없음 | 정확 |
| D39 | 04_setup.md:49 | "공개 코드는 글자 수 기준으로 나눈다" | 공식 코드 | 줄에 인용 표기 없음 | 정확 |
| D40 | 04_setup.md:49 | "본 연구는 코드를 따랐다" | 원고 코드 | 줄에 인용 표기 없음 | 정확 |
| D41 | 06_discussion.md:81 | "TableRAG(Yu)의 SQL 경로도 재현하지 않았다" | Yu et al., 2025; 원고 코드 | 줄에 인용 표기 없음 | 정확 |
| E3 | 02_related.md:23 | "질문과 관련은 있으나 답이 아닌 정보가 섞이면 리더의 정확도가 떨어진다는 것은 확립된 현상이다" | 인용 없음 (뒤 문장의 Cuconasu 2024·CABINET 2024로 대조) | 문장에 인용 없음(같은 문단 다른 문장에만 인용) | 부정확 |
| E7 | 02_related.md:31 | "LLM 추론 비용 때문에 전체 평가 집합 대신 일부 질의만 평가하는 것은 표 질의응답 연구에서 흔하다" | 인용 없음 (뒤 4편으로 대조) | 문장에 인용 없음(같은 문단 다른 문장에만 인용) | 정확 |
| E14 | 02_related.md:33 | "질의 유형마다 정확도가 크게 다르므로" | 인용 없음 (TableBench로 대조) | 문장에 인용 없음(같은 문단 다른 문장에만 인용) | 정확 |
| E15 | 02_related.md:33 | "유형별 값과 개수를 함께 싣는 것이 관행이다" | 인용 없음 (TableBench로 대조) | 문장에 인용 없음(같은 문단 다른 문장에만 인용) | 부정확 |
| F2 | 03_method.md:77 | "검색용 지시문(\"Represent this sentence for searching relevant passages: \")" | Xiao et al., 2024 (사용법은 모델 카드) | 근거가 논문이 아니라 코드·README·모델 카드 | 정확 |
| F3 | 03_method.md:77 | "이 모델의 사용법대로 질문 앞에만 … 붙이고 문장에는 붙이지 않는다" | Xiao et al., 2024 (사용법은 모델 카드) | 근거가 논문이 아니라 코드·README·모델 카드 | 정확 |
| F4 | 03_method.md:77 | "벡터는 정규화해 코사인 유사도를 쓴다" | 인용 없음(모델 카드 사용법) | 문장에 인용 없음(같은 문단 다른 문장에만 인용) | 정확 |
| F5 | 03_method.md:77 | "입력 한도(512토큰)" | 인용 없음(모델 카드) | 문장에 인용 없음(같은 문단 다른 문장에만 인용) | 정확 |
| F13 | 04_setup.md:69 | "여러 비교를 함께 판정하는 곳은 비교 묶음 단위 Holm 보정값을 함께 적는다" | 인용 없음 (Holm 1979는 01_intro.md:28에만 인용) | 문장에 인용 없음(같은 문단 다른 문장에만 인용) | 확인 불가 |
| F14 | 04_setup.md:75 | "continuous batching(끝난 요청 자리에 다음 요청을 바로 넣는 생성 방식)" | 인용 없음(transformers 문서) | 줄에 인용 표기 없음 | 정확 |
| F15 | 04_setup.md:75 | "이 방식은 한 번에 한 건씩 생성하는 방식과 부동소수 연산 순서가 달라 출력 문자열이 바이트 단위로 같지 않다" | 인용 없음(transformers 문서) | 줄에 인용 표기 없음 | 원문에 없음 |
| F16 | 06_discussion.md:73 | "교차 인코더(bge-reranker-v2-m3)" | 인용 없음(모델 카드) | 줄에 인용 표기 없음 | 정확 |
| G1 | 01_intro.md:7; 03_method.md:17 | "값 `52.1`은 '지역 = 동부 온타리오 > 프랑스어 사용 노동자', '산업 = 퍼센트 > 음식 서비스'라는 두 경로" / 표 3-1 "행 경로 percent > food service, 열 경로 eastern ontario > french-language workers" | 인용 없음(HiTab 데이터) | 줄에 인용 표기 없음 | 정확 |
| G2 | 01_intro.md:9 | "흔한 방식은 표를 마크다운이나 쉼표 구분 텍스트로 펴는 것이다" | 인용 없음 | 줄에 인용 표기 없음 | 원문에 없음 |

## 7. (다) 수치 대조표의 "연도·인용"·"외부 문헌" 수치 157개와 대조 행

참고문헌 목록(08_refs.md)의 연도·권·쪽·arXiv 번호는 문헌별 서지 행에서, 본문 수치는 같은 줄의 행에서 대조했다. 대조 행이 없는 수치는 0개다.

| 위치 | 수치 | 분류 | 대조 행 |
|---|---|---|---|
| 01_intro.md:5 | 2020 | 연도·인용 | E1, E2 |
| 01_intro.md:15 | 2026 | 연도·인용 | A1, A2, A3, B1, D1, D2 |
| 01_intro.md:15 | 2022 | 연도·인용 | A1, A2, A3, B1, D1, D2 |
| 01_intro.md:15 | 2023 | 연도·인용 | A1, A2, A3, B1, D1, D2 |
| 01_intro.md:28 | 2024 | 연도·인용 | C1, C2, C3, C4, D3, D4, F10 |
| 01_intro.md:28 | 2025 | 연도·인용 | C1, C2, C3, C4, D3, D4, F10 |
| 01_intro.md:28 | 1979 | 연도·인용 | C1, C2, C3, C4, D3, D4, F10 |
| 02_related.md:5 | 2022 | 연도·인용 | A4, A5, A6, A7, A8, A9 |
| 02_related.md:7 | 2022 | 연도·인용 | B2, B3, B4, B5, B6, B7, B8 |
| 02_related.md:7 | 2 | 연도·인용 | B2, B3, B4, B5, B6, B7, B8 |
| 02_related.md:7 | 10 | 연도·인용 | B2, B3, B4, B5, B6, B7, B8 |
| 02_related.md:7 | 76.4% | 연도·인용 | B2, B3, B4, B5, B6, B7, B8 |
| 02_related.md:7 | 15 | 연도·인용 | B2, B3, B4, B5, B6, B7, B8 |
| 02_related.md:7 | 80.8% | 연도·인용 | B2, B3, B4, B5, B6, B7, B8 |
| 02_related.md:7 | 10 | 연도·인용 | B2, B3, B4, B5, B6, B7, B8 |
| 02_related.md:11 | 2023 | 연도·인용 | C5, D5, D6, D7, D8, D9, D10, D11 |
| 02_related.md:11 | 2024 | 연도·인용 | C5, D5, D6, D7, D8, D9, D10, D11 |
| 02_related.md:11 | 2024 | 연도·인용 | C5, D5, D6, D7, D8, D9, D10, D11 |
| 02_related.md:13 | 2024 | 연도·인용 | C6, C7, C8, C9, C10, C11, C12, C13, C14 |
| 02_related.md:13 | 10,000 | 연도·인용 | C6, C7, C8, C9, C10, C11, C12, C13, C14 |
| 02_related.md:13 | 5 | 연도·인용 | C6, C7, C8, C9, C10, C11, C12, C13, C14 |
| 02_related.md:13 | 30 | 연도·인용 | C6, C7, C8, C9, C10, C11, C12, C13, C14 |
| 02_related.md:15 | 2025 | 연도·인용 | D12, D13, D14, D15, D16, D17 |
| 02_related.md:15 | 1,000 | 연도·인용 | D12, D13, D14, D15, D16, D17 |
| 02_related.md:15 | 200 | 연도·인용 | D12, D13, D14, D15, D16, D17 |
| 02_related.md:17 | 2026 | 연도·인용 | D18, D19, D20, D21, D22 |
| 02_related.md:17 | 2,178 | 연도·인용 | D18, D19, D20, D21, D22 |
| 02_related.md:19 | 2026 | 연도·인용 | D23, D24 |
| 02_related.md:23 | 2024 | 연도·인용 | E3, E4, E5, E6 |
| 02_related.md:23 | 2024 | 연도·인용 | E3, E4, E5, E6 |
| 02_related.md:23 | 2024 | 연도·인용 | E3, E4, E5, E6 |
| 02_related.md:29 | 2026 | 연도·인용 | A10, A11 |
| 02_related.md:31 | 2024 | 연도·인용 | E7, E8, E9, E10, E11, E12, E13 |
| 02_related.md:31 | 100 | 연도·인용 | E7, E8, E9, E10, E11, E12, E13 |
| 02_related.md:31 | 2026 | 연도·인용 | E7, E8, E9, E10, E11, E12, E13 |
| 02_related.md:31 | 500 | 연도·인용 | E7, E8, E9, E10, E11, E12, E13 |
| 02_related.md:31 | 2024 | 연도·인용 | E7, E8, E9, E10, E11, E12, E13 |
| 02_related.md:31 | 2025 | 연도·인용 | E7, E8, E9, E10, E11, E12, E13 |
| 02_related.md:31 | 500 | 연도·인용 | E7, E8, E9, E10, E11, E12, E13 |
| 02_related.md:33 | 2025 | 연도·인용 | E14, E15, E16 |
| 02_related.md:43 | 2023 | 연도·인용 | D25, D26, D27 |
| 02_related.md:44 | 2024 | 연도·인용 | C15 |
| 02_related.md:45 | 2025 | 연도·인용 | D28, D29, D30 |
| 02_related.md:46 | 2026 | 연도·인용 | D31, D32, D33 |
| 03_method.md:77 | 2024 | 연도·인용 | F1, F2, F3, F4, F5 |
| 03_method.md:79 | 2009 | 연도·인용 | F6 |
| 03_method.md:95 | 2024 | 연도·인용 | F7 |
| 03_method.md:96 | 2025 | 연도·인용 | F8, F9 |
| 04_setup.md:36 | 2025 | 연도·인용 | D35, D36, D37 |
| 04_setup.md:37 | 2024 | 연도·인용 | C18 |
| 04_setup.md:38 | 2024 | 연도·인용 | C19 |
| 04_setup.md:39 | 2024 | 연도·인용 | C20 |
| 04_setup.md:40 | 2024 | 연도·인용 | C21 |
| 04_setup.md:43 | 2024 | 연도·인용 | C22, C23, C24, C25, C26, C27 |
| 04_setup.md:43 | arXiv:2410.04739 | 연도·인용 | C22, C23, C24, C25, C26, C27 |
| 04_setup.md:49 | 1,000 | 연도·인용 | D38, D39, D40 |
| 04_setup.md:49 | 200 | 연도·인용 | D38, D39, D40 |
| 06_discussion.md:79 | 100 | 외부 문헌 | B20, B21, B22, B23, B24 |
| 06_discussion.md:79 | 4 | 외부 문헌 | B20, B21, B22, B23, B24 |
| 06_discussion.md:79 | 84.9% | 외부 문헌 | B20, B21, B22, B23, B24 |
| 06_discussion.md:79 | 2022 | 연도·인용 | B20, B21, B22, B23, B24 |
| 06_discussion.md:79 | 2 | 연도·인용 | B20, B21, B22, B23, B24 |
| 08_refs.md:3 | 2026 | 연도·인용 | D44 |
| 08_refs.md:3 | 2026 | 연도·인용 | D44 |
| 08_refs.md:3 | arXiv:2602.01969 | 연도·인용 | D44 |
| 08_refs.md:4 | 2024 | 연도·인용 | C36 |
| 08_refs.md:4 | 2024 | 연도·인용 | C36 |
| 08_refs.md:4 | arXiv:2410.04739 | 연도·인용 | C36 |
| 08_refs.md:5 | 2022 | 연도·인용 | A23 |
| 08_refs.md:5 | 60 | 연도·인용 | A23 |
| 08_refs.md:5 | 2022 | 연도·인용 | A23 |
| 08_refs.md:5 | 1094 | 연도·인용 | A23 |
| 08_refs.md:5 | 1110 | 연도·인용 | A23 |
| 08_refs.md:6 | 2024 | 연도·인용 | E18 |
| 08_refs.md:6 | 2024 | 연도·인용 | E18 |
| 08_refs.md:6 | 719 | 연도·인용 | E18 |
| 08_refs.md:6 | 729 | 연도·인용 | E18 |
| 08_refs.md:7 | 2024 | 연도·인용 | E21 |
| 08_refs.md:7 | 2024 | 연도·인용 | E21 |
| 08_refs.md:7 | arXiv:2402.12424 | 연도·인용 | E21 |
| 08_refs.md:8 | 1979 | 연도·인용 | F21 |
| 08_refs.md:8 | 6 | 연도·인용 | F21 |
| 08_refs.md:8 | 2 | 연도·인용 | F21 |
| 08_refs.md:8 | 65 | 연도·인용 | F21 |
| 08_refs.md:8 | 70 | 연도·인용 | F21 |
| 08_refs.md:9 | 2020 | 연도·인용 | E17 |
| 08_refs.md:9 | 2020 | 연도·인용 | E17 |
| 08_refs.md:9 | 33 | 연도·인용 | E17 |
| 08_refs.md:9 | 9459 | 연도·인용 | E17 |
| 08_refs.md:9 | 9474 | 연도·인용 | E17 |
| 08_refs.md:10 | 2023 | 연도·인용 | D45 |
| 08_refs.md:10 | 61 | 연도·인용 | D45 |
| 08_refs.md:10 | 2023 | 연도·인용 | D45 |
| 08_refs.md:10 | 9909 | 연도·인용 | D45 |
| 08_refs.md:10 | 9926 | 연도·인용 | D45 |
| 08_refs.md:11 | 2024 | 연도·인용 | E20 |
| 08_refs.md:11 | 12 | 연도·인용 | E20 |
| 08_refs.md:11 | 157 | 연도·인용 | E20 |
| 08_refs.md:11 | 173 | 연도·인용 | E20 |
| 08_refs.md:12 | 1947 | 연도·인용 | F22 |
| 08_refs.md:12 | 12 | 연도·인용 | F22 |
| 08_refs.md:12 | 2 | 연도·인용 | F22 |
| 08_refs.md:12 | 153 | 연도·인용 | F22 |
| 08_refs.md:12 | 157 | 연도·인용 | F22 |
| 08_refs.md:13 | 2024 | 연도·인용 | E19 |
| 08_refs.md:13 | 2024 | 연도·인용 | E19 |
| 08_refs.md:13 | arXiv:2402.01155 | 연도·인용 | E19 |
| 08_refs.md:14 | 2024 | 연도·인용 | F19 |
| 08_refs.md:14 | Qwen2.5 | 연도·인용 | F19 |
| 08_refs.md:14 | arXiv:2412.15115 | 연도·인용 | F19 |
| 08_refs.md:15 | 2025 | 연도·인용 | F20 |
| 08_refs.md:15 | Qwen3 | 연도·인용 | F20 |
| 08_refs.md:15 | arXiv:2505.09388 | 연도·인용 | F20 |
| 08_refs.md:16 | 2009 | 연도·인용 | F18 |
| 08_refs.md:16 | BM25 | 연도·인용 | F18 |
| 08_refs.md:16 | 3 | 연도·인용 | F18 |
| 08_refs.md:16 | 4 | 연도·인용 | F18 |
| 08_refs.md:16 | 333 | 연도·인용 | F18 |
| 08_refs.md:16 | 389 | 연도·인용 | F18 |
| 08_refs.md:17 | 2024 | 연도·인용 | D46 |
| 08_refs.md:17 | TAP4LLM | 연도·인용 | D46 |
| 08_refs.md:17 | 2024 | 연도·인용 | D46 |
| 08_refs.md:17 | arXiv:2312.09039 | 연도·인용 | D46 |
| 08_refs.md:18 | 2026 | 연도·인용 | E22 |
| 08_refs.md:18 | arXiv:2603.15402 | 연도·인용 | E22 |
| 08_refs.md:19 | 2025 | 연도·인용 | E24 |
| 08_refs.md:19 | 2025 | 연도·인용 | E24 |
| 08_refs.md:19 | 7105 | 연도·인용 | E24 |
| 08_refs.md:19 | 7137 | 연도·인용 | E24 |
| 08_refs.md:20 | 2025 | 연도·인용 | E25 |
| 08_refs.md:20 | 2025 | 연도·인용 | E25 |
| 08_refs.md:20 | 39 | 연도·인용 | E25 |
| 08_refs.md:20 | 24 | 연도·인용 | E25 |
| 08_refs.md:20 | 25497 | 연도·인용 | E25 |
| 08_refs.md:20 | 25506 | 연도·인용 | E25 |
| 08_refs.md:21 | 2024 | 연도·인용 | F17 |
| 08_refs.md:21 | 2024 | 연도·인용 | F17 |
| 08_refs.md:21 | arXiv:2309.07597 | 연도·인용 | F17 |
| 08_refs.md:22 | 2025 | 연도·인용 | D42 |
| 08_refs.md:22 | 2025 | 연도·인용 | D42 |
| 08_refs.md:22 | arXiv:2506.10380 | 연도·인용 | D42 |
| 08_refs.md:23 | 2026 | 연도·인용 | D43 |
| 08_refs.md:23 | 32 | 연도·인용 | D43 |
| 08_refs.md:23 | 2026 | 연도·인용 | D43 |
| 08_refs.md:23 | 1880 | 연도·인용 | D43 |
| 08_refs.md:23 | 1891 | 연도·인용 | D43 |
| 08_refs.md:24 | 2024 | 연도·인용 | E23 |
| 08_refs.md:24 | 2024 | 연도·인용 | E23 |
| 08_refs.md:24 | arXiv:2311.09206 | 연도·인용 | E23 |
| 08_refs.md:25 | 2022 | 연도·인용 | B25 |
| 08_refs.md:25 | 60 | 연도·인용 | B25 |
| 08_refs.md:25 | 2022 | 연도·인용 | B25 |
| 08_refs.md:25 | 6588 | 연도·인용 | B25 |
| 08_refs.md:25 | 6600 | 연도·인용 | B25 |
| 08_refs.md:26 | 2026 | 연도·인용 | A24 |
| 08_refs.md:26 | 2026 | 연도·인용 | A24 |
| 08_refs.md:26 | arXiv:2510.09671 | 연도·인용 | A24 |

## 8. (라) 참고문헌 목록과 본문 인용의 대응

- 목록 24편과 본문(초록 제외) 인용 24건이 모두 짝을 이룬다. **목록에만 있는 문헌: 없음. 본문에만 있는 인용: 없음.**
  (Wu et al. 2025a·2025b, Zhang et al. 2024·2026, Qwen Team 2024·2025 도 연도·첨자로 구별된다.)
- 목록에 없는 출처가 본문에 이름·URL로만 나오는 곳(서지 항목 없음):
  - 04_setup.md:21 MultiHiertt 공식 저장소 https://github.com/psunlpgroup/MultiHiertt (URL만)
  - 04_setup.md:23, 각주 1 Hugging Face `bevaya/MultiHiertt`, `yilunzhao/MultiHiertt` (데이터 저장소 이름만)
  - 03_method.md:37 ToTTo(페이지 제목의 출처) — 인용 없음(A16)
  - 04_setup.md:55 HiTab·MultiHiertt 공식 채점 코드 — 저장소 표기 없음(A21·A22, B18·B19)
  - 04_setup.md:69 Holm 보정 — 이 줄에는 인용 없음, Holm(1979)은 01_intro.md:28 에만 인용(F13)
  - 04_setup.md:75 transformers continuous batching — 인용 없음(F14·F15)
  - 06_discussion.md:73 bge-reranker-v2-m3 — 인용 없음(F16)
- 서지 사항 판정(문헌별 한 행, 24행): 정확 20, 부정확 4.
  - A24 Zhou et al. 2026: 논문집 전체 이름(64th Annual Meeting, Volume 1)과 쪽(12161–12189) 누락.
  - C36 Chen et al. 2024: NeurIPS 권·쪽(37, pp. 74899–74921) 누락.
  - D43 Zhang et al. 2026(MixRAG): 논문집 이름의 "V.1" 누락, DOI 권장.
  - D45 Lin et al. 2023(ITR): "(Volume 1: Long Papers)" 누락.
  - F21 Holm 1979·F22 McNemar 1947 은 본문을 열지 못해(유료) 출판사 서지 기록(JSTOR·Cambridge Core)으로만 서지 사항을 대조해 정확으로 두었다. 본문 주장(F10~F12)은 확인 불가다.

## 9. 판정 재확인 기록

묶음 결과 가운데 수치·핵심 근거는 원문 추출 텍스트나 원고 코드로 다시 확인했다.

- MultiHiertt(B): Table 2 "Support Facts Correctness 84.9"(6591쪽), 3.2절 "mark all the supporting facts", 5.4절 "76.4% recall for the top-10 … 80.8% recall for the top-15"(6594쪽), README 의 `table_description` 필드 설명을 원문에서 확인.
- HiTab(A): 각주 2 "For samples with XLOOKUP or IF formulas, we didn't explicitly provide the formulas"(1096쪽), 서베이 부록 표의 Closed 행에 Cheng et al.(2022) 확인.
- Cuconasu(E4): 표 1 Llama2 Near 열 10개 .3716 → 12개 .3991 → 14개 .4118(비단조) 확인. Wang(E11): 부트스트랩 신뢰구간 "1,000 iterations, N=2,000" 확인.
- TableRAG 재구현 상한(C25·C30): 원고 코드(`scripts/retrieval_accuracy.py: tablerag_units`, dtype infer)로 HiTab 단일 셀 조회 991건 중 정답 셀이 전달 가능한 셀 집합에 드는 질의를 다시 셌다 — leaf 327, path 327(.3300), all_object 991. 결과 파일의 검색 정확도 leaf .2775, path .2916 은 이 상한 아래다(모델 추론 없는 결정적 계산).
- RowCol .7639 · RandRow .3744(C34): 재실행 결과 파일(HiTab 단일 셀 조회)과 같다.
- 서론 52.1 예시(G1): HiTab 원자료(test_samples.jsonl, 표 100)와 머리글 트리 경로로 확인.
