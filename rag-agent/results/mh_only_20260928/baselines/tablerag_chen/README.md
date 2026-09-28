# tablerag_chen — Google TableRAG(Chen et al., NeurIPS 2024) 원방법, MultiHiertt 같은 조건

- 원논문: Chen et al., "TableRAG: Million-Token Table Understanding with Language Models", NeurIPS 2024.
- 공식 코드: <https://github.com/google-research/google-research/tree/master/table_rag>,
  commit `08a8d6736475776f42ffac23b2c13111a28e5795` (`vendor/table_rag/VENDOR_COMMIT`, git sparse-checkout 2026-09-29).
- 실행 코드: `scripts/baselines/tablerag_chen.py`, 실행: `run.sh` (아래). 결과: 이 디렉터리의 `tablerag_chen_records.jsonl`(검색 2,885),
  `tablerag_chen_answer.jsonl`(답변 882), 요약 `tablerag_chen.json` / `tablerag_chen_answer.json`. 점검 실행은 `smoke3/`.
- 기존 `tablerag_path`/`tablerag_leaf`(색인 글 형식만 옮긴 이식, `rag_agent/serialization/tablerag_unit.py`)는 이 비교군으로 대체한다.

## 파이프라인 (원본 `agent/rag_agent.py: TableRAGAgent.run` 의 단계 그대로)

1. **질의 확장** — `extract_column_prompt`(스키마 질의 목록), `extract_cell_prompt`(셀 키워드 목록), wtq 프롬프트 원문. LLM 이 JSON 목록을 내고,
   파싱 실패는 원본대로 최대 3회 재시도, 숫자 키워드는 원본대로 버린다(`is_numeric`).
2. **스키마 검색 + 셀 검색** — 원본 `Retriever.build_schema_corpus`/`build_cell_corpus`(열마다 스키마 문서 1개; 숫자·날짜 열은 `{column_name, dtype, min, max}`
   요약 문서 1개, 범주 열은 `{column_name, cell_value}` 문서를 (열, 값) 로 중복 제거해 빈도순 B=10,000 개까지), 질의마다 top_k=5, 결과는 원본대로 `set()` 으로 합침.
   `retrieve_mode='embed'`(원본 기본), 하이브리드/BM25 없음.
3. **풀이** — 원본 `solver_loop`(PyReAct): `Thought/Action/Observation` 을 최대 `max_depth=5` 회, `Action` 의 코드를 원본 `python_repl_ast` 로 실행(pandas·numpy),
   호출당 `max_tokens=128`, 정지열 `Observation:`. 답 = 원본 규칙(마지막 `Answer:` 뒤 첫 줄) 그대로 — 추가 추출·정규화 없음.

## 원설정과 다른 것 (왼쪽 = 여기, 오른쪽 = 논문·공식 코드)

| # | 여기 | 논문·공식 코드 | 이유 |
|---|---|---|---|
| 1 | 모든 LLM 호출 = Qwen3-8B 4bit, 비생각, greedy(temperature 0), seed 42, 사용자 메시지 하나(시스템 메시지 없음 — 원본 `query_openai` 와 같음) | GPT-3.5-turbo-0125 / GPT-4 / Gemini, temperature 0.8, top_p 0.95, self-consistency sc=1(run.py 기본) | 같은 조건 규칙(PREREG-2026-09-29). `vendor/table_rag/agent/local_backend.py: LocalQwen` |
| 2 | 임베딩 = BAAI/bge-base-en-v1.5(로컬 캐시), sentence-transformers 인코딩(모델 자체 CLS 풀링+정규화), 질의 접두어 없음, `faiss.IndexFlatL2` 전수 L2 | OpenAI text-embedding-3-large, langchain_community `HuggingFaceEmbeddings`/`FAISS`(IndexFlatL2) | 같은 인코더 규칙. langchain/langchain_community 미설치 → 같은 호출면의 대체 클래스(`local_backend.py`). 정규화 벡터라 L2 순위 = 코사인 순위. 본 방법과 달리 bge 질의 접두어는 넣지 않았다(원본 `HuggingFaceEmbeddings` 가 넣지 않는다) |
| 3 | **(i) 계층 머리글**: 표 프레임 = `tablerag_unit.official_frame(table, mode="path")` — 열 이름 = 열 머리글 경로를 ` > ` 로 이은 것, 원본 `infer_dtype` 그대로 | 머리글 행 1개인 평면 표 `pd.DataFrame(table_text[1:], columns=table_text[0])` | MultiHiertt 표는 머리글이 여러 행. 기존 이식에서 원본 `infer_dtype` 와 대조된 프레임을 그대로 쓴다 |
| 3b | (i) 행 라벨 열: 프레임 0열에 행 라벨 열을 넣는다 — 이름 = 격자 모서리(머리글 행×머리글 열) 글, 비어 있으면 `row`(이미 있는 이름이면 ` (row)` 붙임); 값 = 행 머리글 경로를 ` > ` 로 이은 것. 이 열도 원본 `infer_dtype` 를 거친다 | 평면 표에서 행 라벨은 보통 첫 열(범주 열)이라 셀 문서·스키마 문서가 자동으로 생기고, 풀이가 `df[df[col]==...]` 로 행을 고른다 | 우리 파서(`MHTable.data`)는 행 머리글을 데이터 밖에 둔다. 빼면 풀이가 행을 고를 수 없고, 기존 이식(`retrieval_accuracy.tablerag_units`)도 같은 이유로 행 라벨 문서를 넣었다 |
| 4 | **(ii) 문서당 표 여러 개**: 표 k(문서 안 순서, 0부터)마다 프레임 `df_k`; 스키마·셀 코퍼스 = 표마다 원본 `build_*_corpus` 를 부른 뒤 합침(열 이름에 `df_k.` 접두어, B=10,000 은 표마다). 풀이 프롬프트의 문장 "The name of the dataframe is `df`." 만 "The document has N tables, loaded as pandas dataframes `df_0`, … Column names … are prefixed with the dataframe name as `df_k.<column>`; index a column as `df_k["<column>"]`." 로 바꾸고, REPL 지역변수로 `df_0…` 을 넘긴다(`df` 는 없음) | 질문당 표 1개, `df` 하나 | MultiHiertt 문서는 표가 여러 개(질문의 문서 안 검색 = 본 방법과 같은 범위). `DocRetriever.get_docs`, `DocAgent.get_prompt`, 패치 3 |
| 5 | **(iii) 표 읽기**: `scripts/mh_arms.py: build_tables(docs, "v3.3u", "none")` — 본 방법과 같은 표 집합(3행·2열 미만 표 제외), 본문 단락 없음 | 데이터셋이 준 `table_text` | 같은 조건 |
| 6 | 스키마·셀 DB 캐시 = `.cache/tablerag_chen/db/` (`vecs.npy`+`docs.json`, 문서 uid 별) | `db/` 에 FAISS `save_local` | 같은 역할, 형식만 다름 |
| 7 | `PYTHONHASHSEED=0` 고정 | 없음 | 원본이 검색 결과를 `set()` 으로 합쳐 문서 순서가 문자열 해시에 달렸다. 결과 재현을 위해 고정 |
| 8 | 풀이 LLM 호출의 프롬프트 토큰 수(`n_tok`) = 원본 `get_token_count`(채팅 템플릿 없이 프롬프트 원문 토큰) 합 | 같은 계산 | 원본 계측 그대로. 본 방법 `n_tok`(템플릿 포함)과 정의가 다르므로 비교 시 주의 |

리더 프롬프트 `cot`·최대 384토큰(본 방법)은 이 비교군에 적용하지 않는다 — 원방법의 풀이가 프롬프트·토큰 한도(128/호출)·반복(≤5)을 정의한다.

## 채점

- 답변: `pred` = 원본 `solver_loop` 의 answer 문자열 그대로. EM = `rag_agent/eval/multihiertt_em.mh_exact_match`(공식 포팅, 문항 유형 = program 유무),
  F1 = `rag_agent/eval/multihiertt_f1.mh_f1`(공식 `get_span_selection_metrics`·`evaluate_span_program_result` 이식, 이미 저장소에 있음 — 새로 옮기지 않음).
  정답 = 공식 파일(`yilunzhao/MultiHiertt@f18473d`, `answer_accuracy_mh.py` 와 같은 `official_answers`).
- 검색 레코드(`tablerag_chen_records.jsonl`): `ranked_units` = 합쳐진 셀 검색 문서를 가장 가까운 셀 질의와의 L2 거리 순(상위 30 또는 전부), 문서마다
  `cells`(배달 셀 id "표-행-열", 펼친 격자 좌표 = 본 방법 `gold_ids` 와 같은 규약), `text`, `n_tok`(Qwen3 토크나이저), `dist`.
  셀 매핑 = `retrieval_accuracy.tablerag_units` 규칙: 숫자·날짜 요약 문서 → 그 열의 min·max 행, 범주 문서 → 그 열에서 값이 같은 셀 전부; 행 라벨 열 문서·빈 값 문서는 배달 없음.
  `schema_hits` = 스키마 검색 결과(`result_text`, 배달 셀 없음), `context` = 풀이 프롬프트에 들어간 스키마+셀 검색 글 그대로, `correct` = 정답 셀 전부가 `ranked_units` 의 셀 합집합에 있는가.
  표준 Recall@k 등은 `final/rpm.py` 가 이 레코드에서 계산한다.

## 공식 코드 패치 (vendor/table_rag, 전부 `T2 PATCH` 주석으로 표시)

1. `agent/model.py`: `tiktoken`/`vertexai` import 를 try/except 로 감쌈(미설치, openai/google 제공자에서만 쓰임); 제공자 `local`(`'local:<hf id>'`) 추가 —
   `get_provider`/`get_context_limit`/`__init__`/`query`/`get_token_count` 각 한 갈래. 생성은 `agent/local_backend.py: LocalQwen`(새 파일).
2. `agent/retriever.py`: langchain 계열 import 를 try/except 로 감싸고, 실패하면 `agent/local_backend.py` 의 `Document`(langchain_core 원본 클래스)·`FAISS`·`HuggingFaceEmbeddings` 대체를 쓴다.
   `Retriever` 의 메서드는 손대지 않았다.
3. `agent/agent.py: solver_loop`: `custom_locals={'df': df}` 한 줄 → `df` 가 dict 이면 그 dict 를 REPL 지역변수로(표 여러 개), DataFrame 이면 원본과 같음.
4. 새 파일 `agent/local_backend.py`, `VENDOR_COMMIT`. 프롬프트(`prompts/wtq.py`)·`utils/execute.py`·`utils/utils.py`·`rag_agent.py` 는 손대지 않았다.

## 실행

```
bash results/mh_only_20260928/baselines/tablerag_chen/run.sh --limit 3   # 점검 → smoke3/
bash results/mh_only_20260928/baselines/tablerag_chen/run.sh             # 전체 (검색 2,885 + 답변 882) → 이 디렉터리
bash results/mh_only_20260928/baselines/tablerag_chen/run.sh --resume    # 끊기면 이어 쓰기
```
실행 순서: 882 부분집합(query_id 순, 검색+답변) → 나머지 2,003 문항(검색만). `--limit N` = 882 부분집합 앞 N 건만.

## 점검 실행 (smoke3, 882 부분집합 앞 3건)

실행 2026-09-29, `smoke3/` (`run.sh --limit 3`, 총 141초, 모델 적재 제외; GPU 사용 약 6.4 GB — Qwen3-8B 4bit + bge-base).
첫 실행에서 `df_k` 를 REPL 에 정수 키로 넘겨 5회 모두 `NameError` 가 난 결함을 고치고(코드 한 줄) 다시 돌린 결과다.

| query_id | 유형 | 정답 | 예측(`pred`) | EM | F1 | LLM 호출 | 풀이 반복 | 코드 오류 | Final Answer 도달 | 검색(정답 셀 전부/일부) | 초 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 007a44f1 | lookup | no | 'The data does not provide direct values for "Investment Banking Fees" '… | 0 | 0.0 | 7 | 5 | 0 | False | 0/0 | 58.429 |
| 00fa4ad4 | lookup | Fixed maturities, Commercial mortgage and other loans | 'Thought: The values in the "Amout" column contain commas, which need t'… | 0 | 0.07 | 7 | 5 | 1 | False | 0/1 | 31.124 |
| 01208501 | arith | 429192.5 | 'Thought: I need to correctly extract the values for the American Tower'… | 0 | 0.0 | 7 | 5 | 5 | False | 0/1 | 51.729 |

- 3건 모두 반복 5회 안에 `Final Answer:` 를 내지 못했다 → 원본 규칙대로 마지막 생각의 첫 줄이 `pred` 가 되어 EM 0. 셋째 문항은 greedy 생성이 같은 미완성 리스트 컴프리헨션을 5회 반복(SyntaxError ×5).
  이 결과는 실측이며 코드 결함이 아니다(첫째 문항은 코드 오류 0, pandas 로 행 라벨 검색까지 성공). 전체 실행 뒤 `terminal_rate`·`code_errors_mean` 을 표에 같이 적는다.
- 검색: LLM 이 낸 셀 키워드는 범주 값("Investment Banking", "Fees", "Year")이라 정답인 숫자 셀 문서(`"cell_value": "$ 41,879"`)는 검색되지 않음 — 원본의 설계(숫자 키워드 제거, 숫자는 코드로 읽음).
  문항당 스키마·셀 문서 수 16~22 / 135~285(B=10,000 에 훨씬 못 미쳐 잘림 없음). LLM 호출 = 확장 2 + 풀이 5 = 7, 프롬프트 토큰 합 6,946~9,188.
- 첫째 문항 풀이 흔적(프롬프트의 검색 결과 부분부터, 원문 그대로):

```
Since you cannot view the table directly, here are some schemas and cell values retrieved from the table.

Schema Retrieval Queries: Year, Investment_Banking_Fees
Schema Retrieval Results:
{"column_name": "df_0.2006", "dtype": "object", "cell_examples": ['$41,895', '32,983', '8,122']}
{"column_name": "df_1.Year Ended > % Incr. (Decr.) > column 4", "dtype": "object", "cell_examples": ['', '46%', '-34%']}
{"column_name": "df_0.2007", "dtype": "object", "cell_examples": ['$45,296', '32,238', '8,402']}
{"column_name": "df_2.Year Ended > % Incr. (Decr.) > column 4", "dtype": "object", "cell_examples": ['', '88%', '108%']}
{"column_name": "df_1.Year Ended > % Incr. (Decr.) > column 2", "dtype": "object", "cell_examples": ['', '-4%', '48%']}
{"column_name": "df_2.Year Ended > September 30, 2007 > ($ in 000's)", "dtype": "object", "cell_examples": ['', '$ 68,685', '59,276']}
{"column_name": "df_1.Year Ended > September 30, 2007 > ($ in 000's)", "dtype": "object", "cell_examples": ['', '$ 41,879', '2,846']}

Cell Retrieval Queries: Investment Banking, Fees, Year
Cell Retrieval Results:
{"column_name": "df_1.row", "cell_value": "Revenues > Securities Commissions and > Interest Expense"}
{"column_name": "df_1.row", "cell_value": "Revenues > Securities Commissions and > Investment Banking Fees"}
{"column_name": "df_1.row", "cell_value": "Revenues > Securities Commissions and > Interest Income"}
{"column_name": "df_1.Year Ended > September 30, 2006", "cell_value": ""}
{"column_name": "df_2.Year Ended > September 30, 2005", "cell_value": ""}
{"column_name": "df_1.row", "cell_value": "Revenues > Securities Commissions and > Investment Advisory Fees"}
{"column_name": "df_2.Year Ended > September 30, 2006", "cell_value": ""}
{"column_name": "df_0.row", "cell_value": "Recurring leasing costs"}
{"column_name": "df_1.Year Ended > % Incr. (Decr.) > column 2", "cell_value": ""}
{"column_name": "df_1.row", "cell_value": "Revenues > Non-Interest Expense > Other Expense"}
{"column_name": "df_2.Year Ended > % Incr. (Decr.) > column 4", "cell_value": ""}
{"column_name": "df_1.row", "cell_value": "Revenues > Non-Interest Expense > Compensation Expense"}
{"column_name": "df_2.row", "cell_value": "Interest Income and Expense > Net Interest Income"}

Strictly follow the given format to respond:
Thought: you should always think about what to do
Action: the single line Python command to execute
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: before giving the final answer, you should think about the observations
Final Answer: the final answer to the original input question (Answer1, Answer2, ...)

Notes:
- Do not use markdown or any other formatting in your responses.
- Ensure the last line is only "Final Answer: Answer1, Answer2, ..." form, no other form.
- Directly output the Final Answer rather than outputting by Python.
- Ensure to have a concluding thought that verifies the table, observations and the question before giving the final answer.

Now, please use ``python_repl_ast` with the column names and cell values above to answer the question: Does Investment Banking Fees keeps increasing each year between 2006 and 2007 ?

Begin!
Thought: Thought: I need to determine if Investment Banking Fees increased from 2006 to 2007. The relevant data is likely in the columns related to "Investment Banking Fees" and the years 2006 and 2007. I will check the cell values for these years and compare them.  
Action: print([cell for cell in df_1["row"] if "Investment Banking Fees" in cell] + [cell for cell in df_2["row"] if "Investment Banking Fees" in cell])
Observation: ['Revenues > Securities Commissions and > Investment Banking Fees']
Thought: The relevant row is "Revenues > Securities Commissions and > Investment Banking Fees". I will now check the values for the years 2006 and 2007.  
Action: print([cell for cell in df_1["Year Ended > September 30, 2006"] if "Investment Banking Fees" in cell] + [cell for cell in df_2["Year Ended > September 30, 2006"] if "Investment Banking Fees" in cell])
Observation: []
Thought: The values for the year 2006 are not directly available in the provided data. I will now check the values for the year 2007.  
Action: print([cell for cell in df_1["Year Ended > September 30, 2007 > ($ in 000's)"] if "Investment Banking Fees" in cell] + [cell for cell in df_2["Year Ended > September 30, 2007 > ($ in 000's)"] if "Investment Banking Fees" in cell])
Observation: []
Thought: The values for the years 2006 and 2007 are not directly available in the provided data. I will now check the values for the year 2006 in the "df_0.2006" column and the year 2007 in the "df_0.2007" column.  
Action: print([cell for cell in df_0["2006"] if "Investment Banking Fees" in cell] + [cell for cell in df_0["2007"] if "Investment Banking Fees" in cell])
Observation: []
Thought: The data does not provide direct values for "Investment Banking Fees" for the years 2006 and 2007. I will now check the percentage changes between these years to determine if the fees increased.  
Action: print([cell for cell in df_1["Year Ended > % Incr. (Decr.) > column 2"] if "Investment Banking Fees" in cell] + [cell for cell in df_2["Year Ended > % Incr. (Decr.) > column 4"] if "Investment Banking Fees" in cell])
Observation: []

```


## 전체 실행 예상 시간

3건 평균 47초/문항(확장 2호출 + 풀이 5호출, 4비트 greedy, 호출당 ≤128토큰) → 답변 882건 ≈ 11.5시간; 검색만 하는 2,003건은 확장 2호출 + 임베딩 ≈ 7~8초/문항 ≈ 4시간.
합계 **≈ 15~16시간**(3건 기준 추정, 문서 크기·반복 횟수에 따라 달라짐; GPU 를 다른 실행과 나누면 더 걸림). 끊기면 `run.sh --resume`.
