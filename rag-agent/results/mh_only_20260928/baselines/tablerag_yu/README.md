# `tablerag_yu` — TableRAG (Yu, Jian & Chen, EMNLP 2025) 원 방법 비교군

논문: *A Retrieval Augmented Generation Framework for Heterogeneous Document Reasoning* (arXiv 2506.10380).
공식 코드: <https://github.com/yxh-y/TableRAG> → `vendor/tablerag_yu`, 커밋 **`f8d798fc60fb44736d7285ca3ca1dbcb7b3e984d`** (2026-09-29 clone, vendored 파일은 수정하지 않음 — 프롬프트·`Embedder`·`Reranker` 를 파일 경로로 그대로 불러 쓴다).
구현: `scripts/baselines/tablerag_yu.py`. 실행: `run.sh` (아래).

## 파이프라인 (공식 코드 대응)

| 단계 | 공식 코드 | 이 구현 |
|---|---|---|
| 오프라인 표 → 텍스트 | `utils/tool_utils.py: excel_to_markdown` ("Table name: …" + 파이프 행) → `RecursiveCharacterTextSplitter(1000, 200)` → 청크마다 `"File name: {key}\n"` 접두 (`tools/retriever.py`) | 기존 `trag_hetero` arm 과 같은 `rag_agent/serialization/chunks.py: table_chunks` (같은 렌더러·분할기·접두, 셀 범위 매핑 재사용). 표 제목이 없어 이름은 빈 문자열 |
| 오프라인 표 → DB | `src/data_persistent.py`: `pd.read_excel` → `infer_and_convert` → `transfer_df_columns` → MySQL `to_sql`, 스키마 `{table_name, column_list=[[열, 형, 'sample values:[…]'], …]}` | 아래 적응 (i)(ii) |
| 온라인 1. 표 선택 | `_run`: 질문(+`"The given table is in {table_id}"`)으로 `retrieve(30, 5)` → 상위 1 청크의 표 = SQL 대상 표·초기 프롬프트의 표 | 같음. 접미는 붙이지 않음(HybridQA 는 표 id 가 주어지는 설정; MultiHiertt 는 문서 안 표 여러 개 중 어느 표인지 모름) |
| 온라인 2. 질의 분해 | `SYSTEM_EXPLORE_PROMPT` + 도구 `solve_subquery` (OpenAI tool-calling), 최대 5회, 1회차에는 답을 받지 않음, 도구 호출이 없으면 `"ERROR: Did not call tool with a suquery!"` | 프롬프트·도구 정의·규칙 그대로. tool-calling 은 Qwen3 채팅 템플릿의 `tools` 지원(`<tool_call>` JSON) |
| 온라인 3. 텍스트 검색 | 하위 질의로 회수 30(내적, `IndexFlatIP`) → 재정렬 상위 5 → `list(set(...))[:3]` 을 `"\n"` 으로 이음 | 같음(검색 범위 = 질문의 문서). `set` 의 순서는 비결정적이라 재정렬 순서를 지키며 중복만 제거 |
| 온라인 4. NL2SQL + 실행 | `NL2SQL_SYSTEM_PROMPT`/`NL2SQL_USER_PROMPT`(상위 1 표 스키마) → ```` ```sql ```` 추출 → 실행, 결과 json 1000자 절단, 실패 문구 `"SQL execution failed: …"` | 같음(SQLite, 패치 1) |
| 온라인 5. 중간 답 | `COMBINE_PROMPT`(청크 3 + 스키마 + SQL + 실행 결과) → `"Subquery Answer: …"` 을 tool 메시지로 | 같음 |
| 최종 답 | `"<Answer>"` 뒤 전부 | 같음 + 채점용 정리(아래) |

## 같은 조건 치환 (논문 값 → 이 실행)

| 항목 | 논문·공식 코드 | 이 실행 |
|---|---|---|
| LLM(분해·NL2SQL·중간 답) | GPT-4o / DeepSeek-V3 / Qwen-72B(temperature 0.1, 0.01) | Qwen3-8B 4bit 비생각·greedy, seed 42 — 모든 호출 |
| 생성 상한 | API 기본값(무제한) | 호출마다 512 토큰 (`--max-tokens`) |
| 임베딩 | BGE-M3 (CLS, 정규화 없음, 내적) | bge-base-en-v1.5, 공식 `Embedder` 클래스 그대로(CLS, 정규화 없음, 내적) |
| 재정렬 | bge-reranker-v2-m3 (GPU, fp32) | bge-reranker-v2-m3 **CPU fp32**(기본, 점수 계산은 같은 모델·정밀도). 8GB GPU 에 리더와 같이 올리면(`--reranker-device cuda:0`, fp16) 메모리 7.9/8 GB 로 생성이 3배 느려짐(아래 점검) |
| 검색 범위 | 코퍼스 전체(HybridQA 표+문단 파일) | 질문의 문서, **표만**(본문 단락 없음) — 본 방법·기존 청크 arm 과 같음 |
| 평가 | LLM 심판(`EVALUATION_PRONPT`) | MultiHiertt 공식 EM(`multihiertt_em.mh_exact_match`) + 공식 F1(`multihiertt_f1.mh_f1`, 45bd9cc 이식) |

## 불가피한 적응 (논문 가정 → 여기)

(i) **MySQL → SQLite**(stdlib `sqlite3`, 문서마다 인메모리 DB, 표 = `table_{문서 안 번호}`).
  - 패치 1: `NL2SQL_USER_PROMPT` 의 `"MySQL syntax"` → `"SQLite syntax"` (실행기가 SQLite 이므로 방언 이름만 바꿈; 나머지 문구·백틱 규칙 동일. SQLite 는 백틱 식별자를 받는다).
  - 스키마의 형 이름은 공식 `pandas_to_mysql_dtype` 그대로(`BIGINT`, `DOUBLE`, `VARCHAR(255)`…) — LLM 에게 보이는 글자이고 SQLite 는 형 이름을 가리지 않는다.
  - 결과 절단 1000자·실패 문구는 `sql_alchemy_helper.fetchall`/`service.py` 그대로.
(ii) **계층 머리글 → SQL 열**: 논문 가정은 머리글 한 줄의 평면 표(첫 열 = 행 이름 데이터 열). 여기서는 `official_frame(table, mode="path")`(열 이름 = `" > "` 로 결합한 열 머리글 경로, 값·수치 변환은 그 함수 그대로)에 **첫 열 `row_header` = 결합한 행 머리글 경로**를 붙였다(행 머리글은 데이터 열이 아니라 이 열이 없으면 행을 고를 수 없다). 열 이름은 공식 `transfer_name`·중복 `_1/_2` 규칙으로 정리하고, 스키마 `column_list` 항목 4번째에 `"original header: <원 경로>"` 를 덧붙여 원 경로를 보인다(공식은 3원소).
  - 표본 값(`sample values`)은 공식 `random.sample` 을 seed 42 로 고정.
(iii) **표 로딩**: `scripts/mh_arms.py: build_tables(header_rule="v3.3u")`(unique_doc 포함) — 본 방법과 같은 표. 표만, 문단 없음.
(iv) **공식 코드의 결함 두 곳을 의도대로 실행**: `main.py` 의 `schema = get_excel_rag_response['nl2sql_prompt']…`(함수 객체를 인덱싱 → 항상 예외 → 문자열을 3변수로 언패킹해 다시 예외) → 의도(스키마 JSON = `nl2sql_prompt` 의 "Based on the schemas above" 앞부분)대로 스키마 JSON 을 넣는다. `list(set(reranked_docs))[:3]` 의 비결정 순서 → 재정렬 순서 유지.
(v) **답 추출(같은 추출 규칙, 모든 비교군 공통)**: 최종 `<Answer>` 를 내는 프롬프트(`SYSTEM_EXPLORE_PROMPT`) 끝에 `scripts/answer_accuracy_mh.py` 의 마커 문장 "On the last line write 'Final answer: ' followed by only the final value, with no units or explanation." 을 한 문장 덧붙이고(공식 문구는 그대로), `<Answer>` 뒤 내용에 같은 `answer_accuracy_mh.extract` 를 적용해 `Final answer:` 뒤를 채점한다. 마커가 없으면 `<Answer>` 내용 전체(앞의 `:`·감싼 `[ ]`·끝 `.` 만 벗김). 마커 문장을 붙였으므로 `<Answer>` 없이 `Final answer:` 만 쓴 출력도 최종 답으로 받는다(공식은 `<Answer>` 만). 원문은 `raw.rounds[*].answer_raw`, 마커 유무는 `marker_found`. 5회 안에 최종 답이 없으면 `pred=null`(공식 `tablerag_answer=""`) → 오답. 논문 값: 마커 없음, `<Answer>` 뒤 문장을 LLM 심판이 읽음. 이유: 모든 비교군에 같은 추출 규칙.

## 출력 (`--stage`)

- `--stage retrieve` → `tablerag_yu_records.jsonl` (2,885 문항, 882 답변 집합이 앞): 질문 자체로 1회차 텍스트 검색만(회수 30 → 재정렬; LLM·SQL 없음). `ranked_units` = 재정렬 상위 3 먼저, 이어 나머지 회수 청크(회수 순), 최대 30, 각 `{cells, text, n_tok(Qwen3 토크나이저), rerank_score}` — Recall@k 목록. `context` = 상위 3 청크 원문. 재정렬기는 GPU fp32(리더 없음).
- `--stage answer` → `tablerag_yu_answer.jsonl` (882): `pred, answer, answer_correct(EM), f1, sub_queries, sql(실행한 SQL 전부와 결과/오류), raw(전체 흔적: init_retrieval·rounds·sql·messages·round1), n_tok(모든 호출 프롬프트 토큰 합), n_llm_calls, n_rounds, n_sql_errors, seconds`. 재정렬기는 CPU fp32(기본; GPU 를 리더와 같이 쓰면 3배 느림).
- gold 좌표 규약은 두 단계 모두 실행 시 검증: 레코드 `gold_ids`("표-행-열", 펼친 격자 좌표) 를 `resolve_gold` 로 다시 풀어 전부 일치해야 진행.

## 실행

```
bash results/mh_only_20260928/baselines/tablerag_yu/run.sh --stage retrieve            # 2,885 검색
bash results/mh_only_20260928/baselines/tablerag_yu/run.sh --stage answer              # 882 전체 파이프라인
bash results/mh_only_20260928/baselines/tablerag_yu/run.sh --stage answer --limit 3    # 점검
```
`--resume` 로 이어 쓴다(저장된 query_id 는 건너뜀). GPU 잠금 `flock .gpu.lock` 은 run.sh 안에 있다.

## 점검 실행(--limit 3, 882 답변 집합의 앞 3문항, 2026-09-29) — 산출물 `smoke2/` (마커 문장 추가 후; 추가 전 첫 점검은 `smoke_limit3/`, EM 0/3·평균 452초, 재정렬기 GPU fp16)

`--stage retrieve --limit 3`: 3문서 16청크, 검색 1.3초(문항당 0.44초, 재정렬기 GPU fp32). 준비(데이터 로드·표 구성·임베딩·모델 로드) 약 20초.

`--stage answer --limit 3` (재정렬기 CPU fp32, GPU allocated 6.08 GiB):

| # | query_id | 유형 | 정답 | 예측(pred) | EM | F1 | 마커 | 회차 | LLM 호출 | SQL 오류 | 프롬프트 토큰 합 | 초 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `007a44f1` | lookup_m2+ | no | 'No, Investment Banking Fees decreased by 4% from 2006 to 2007' | 0 | 0.17 | True | 3 | 7 | 0 | 8191 | 44 |
| 2 | `00fa4ad4` | lookup_m2+ | Fixed maturities, Commercial mortgage and other loans | '2734' | 0 | 0.0 | True | 3 | 7 | 0 | 9417 | 79 |
| 3 | `01208501` | arith_m2+ | 429192.5 | '793000 65385' | 0 | 0.0 | True | 2 | 6 | 1 | 6327 | 38 |

합계 EM 0/3, 평균 54초/문항, 평균 LLM 호출 6.7회. 1회차 검색: 세 문항 모두 재정렬 상위 3 청크가 정답 셀 전부 포함.

### 전체 실행 시간 추정

- `--stage retrieve` 2,885: 검색 0.44초/문항 × 2,885 ≈ 21분 + 준비(2,885문서 표 구성·약 1.5만 청크 임베딩) 수 분 → **약 30분**.
- `--stage answer` 882: 54초/문항 × 882 ≈ **13시간** (3문항 근거; 회차 수·생성 길이에 따라 흔들림, GPU 단독 기준).

### 문항 1 전체 흔적 (`raw`)

```json
{
 "query_id": "007a44f1744249da9b20a6c846787596",
 "kind": "lookup",
 "m": 2,
 "layer": "lookup_m2+",
 "question": "Does Investment Banking Fees keeps increasing each year between 2006 and 2007 ?",
 "gold_ids": [
  "1-5-1",
  "1-5-3"
 ],
 "pred": "No, Investment Banking Fees decreased by 4% from 2006 to 2007",
 "answer": "no",
 "answer_correct": 0,
 "f1": 0.17,
 "sub_queries": [
  "What is the value of Investment Banking Fees for the years 2006 and 2007?",
  "What is the percentage change in Investment Banking Fees from 2006 to 2007?"
 ],
 "sql": [
  {
   "sub_query": "What is the value of Investment Banking Fees for the years 2006 and 2007?",
   "nl2sql_response": "```sql\nSELECT `row_header`, `year_ended_september_30_2007_in_000_s`, `year_ended_september_30_2006` \nFROM `table_2` \nWHERE `row_header` = 'Investment Banking Fees';\n```",
   "sql": "SELECT `row_header`, `year_ended_september_30_2007_in_000_s`, `year_ended_september_30_2006` FROM `table_2` WHERE `row_header` = 'Investment Banking Fees';",
   "result": "[]"
  },
  {
   "sub_query": "What is the percentage change in Investment Banking Fees from 2006 to 2007?",
   "nl2sql_response": "```sql\nSELECT `year_ended_incr` \nFROM `table_2` \nWHERE `row_header` = 'Revenues > Non-Interest Expense > Investment Banking Fees';\n```",
   "sql": "SELECT `year_ended_incr` FROM `table_2` WHERE `row_header` = 'Revenues > Non-Interest Expense > Investment Banking Fees';",
   "result": "[]"
  }
 ],
 "raw": {
  "init_retrieval": {
   "top1_table": "table_2",
   "reranked": [
    1,
    3,
    0,
    2
   ],
   "recall": [
    1,
    0,
    3,
    2
   ]
  },
  "rounds": [
   {
    "step": 1,
    "response": "<tool_call>\n{\"name\": \"solve_subquery\", \"arguments\": {\"subquery\": \"What is the value of Investment Banking Fees for the years 2006 and 2007?\"}}\n</tool_call>",
    "sub_queries": [
     "What is the value of Investment Banking Fees for the years 2006 and 2007?"
    ],
    "tool_runs": [
     {
      "sub_query": "What is the value of Investment Banking Fees for the years 2006 and 2007?",
      "reranked_top3": [
       1,
       3,
       0
      ],
      "recall": [
       1,
       3,
       0,
       2
      ],
      "sub_answer": "The value of Investment Banking Fees for the years 2006 and 2007 is as follows:\n\n- **2007**: $41,879  \n- **2006**: $43,703  \n\nThese values are directly taken from the first table in Content 1. The SQL execution result was empty, so we rely on the table content provided in Content 1 for the answer."
     }
    ]
   },
   {
    "step": 2,
    "response": "<tool_call>\n{\"name\": \"solve_subquery\", \"arguments\": {\"subquery\": \"What is the percentage change in Investment Banking Fees from 2006 to 2007?\"}}\n</tool_call>",
    "sub_queries": [
     "What is the percentage change in Investment Banking Fees from 2006 to 2007?"
    ],
    "tool_runs": [
     {
      "sub_query": "What is the percentage change in Investment Banking Fees from 2006 to 2007?",
      "reranked_top3": [
       1,
       3,
       2
      ],
      "recall": [
       1,
       0,
       3,
       2
      ],
      "sub_answer": "The percentage change in Investment Banking Fees from 2006 to 2007 is **-4%**.\n\nThis is directly taken from the first table in Content 1, where the percentage change for \"Investment Banking Fees\" from 2006 to 2007 is listed as **-4%**. The SQL query returned an empty result, likely due to an issue with the schema mapping or query logic, but the correct answer can be validated from Content 1."
     }
    ]
   },
   {
    "step": 3,
    "response": "Final answer: No, Investment Banking Fees decreased by 4% from 2006 to 2007.",
    "sub_queries": null,
    "answer_raw": "Final answer: No, Investment Banking Fees decreased by 4% from 2006 to 2007.",
    "marker_found": true
   }
  ],
  "sql": [
   {
    "sub_query": "What is the value of Investment Banking Fees for the years 2006 and 2007?",
    "nl2sql_response": "```sql\nSELECT `row_header`, `year_ended_september_30_2007_in_000_s`, `year_ended_september_30_2006` \nFROM `table_2` \nWHERE `row_header` = 'Investment Banking Fees';\n```",
    "sql": "SELECT `row_header`, `year_ended_september_30_2007_in_000_s`, `year_ended_september_30_2006` FROM `table_2` WHERE `row_header` = 'Investment Banking Fees';",
    "result": "[]"
   },
   {
    "sub_query": "What is the percentage change in Investment Banking Fees from 2006 to 2007?",
    "nl2sql_response": "```sql\nSELECT `year_ended_incr` \nFROM `table_2` \nWHERE `row_header` = 'Revenues > Non-Interest Expense > Investment Banking Fees';\n```",
    "sql": "SELECT `year_ended_incr` FROM `table_2` WHERE `row_header` = 'Revenues > Non-Interest Expense > Investment Banking Fees';",
    "result": "[]"
   }
  ],
  "messages": [
   {
    "role": "user",
    "content": "Next, you will complete a table-related question answering task. Based on the provided materials such as the table content (in Markdown format), you need to analyze the User Query.\nAnd try to decide whether the User Input Query should be broken down into subqueries. You are provided with \"solve_subquery\" tool that can get answer for the subqueries.\nAfter you have collected sufficient information, you need to generate comprehensive answers.\n\nTable Contet: |  | Year Ended |  |  |  |  |\n|  | September 30, 2007 | % Incr. (Decr.) | September 30, 2006 | % Incr. (Decr.) | September 30, 2005 |\n|  | ($ in 000's) |  |  |  |  |\n| Revenues |  |  |  |  |  |\n| Securities Commissions and |  |  |  |  |  |\n| Investment Banking Fees | $ 41,879 | -4% | $ 43,703 | 46% | $ 29,928 |\n| Investment Advisory Fees | 2,846 | 48% | 1,919 | -34% | 2,890 |\n| Interest Income | 4,042 | 11% | 3,647 | 90% | 1,919 |\n| Trading Profits | 5,254 | 41% | 3,720 | 18% | 3,141 |\n| Other | 5,062 | 123% | 2,274 | 156% | 890 |\n| Total Revenues | 59,083 | 7% | 55,263 | 43% | 38,768 |\n| Interest Expense | 1,075 | -27% | 1,467 | 195% | 497 |\n| Net Revenues | 58,008 | 8% | 53,796 | 41% | 38,271 |\n| Non-Interest Expense |  |  |  |  |  |\n| Compensation Expense | 28,071 | -4% | 29,185 | 48% | 19,758 |\n| Other Expense | 23,302 | 17% | 19,867 | 93% | 10,294 |\n| Total Non-Interest Expense | 51,373 | 5% | 49,052 | 63% | 30,052 |\n| Minority Interest | 2,995 |  | 1,887 |  | 2,292 |\n| Pre-tax Earnings | $ 3,640 | 27% | $ 2,857 | -52% | $ 5,927 |\n\nInstructions:\n1. Carefully analyze each user query through step-by-step reasoning.\n2. If the query needs information more than the given table content：\n    - Decompose the query into subqueries.\n    - Process one subquery at a time.\n    - Use \"solve_subquery\" tool to get answers for each subquey.\n3. If a query can be answered by table content, do not decompose it. And directly put the orignal query into the \"solve_subquery\" tool.\n    The \"solve_subquery\" tool utilizes SQL execution inside, it can solve complex subquery on table through one tool call.\n4. Generate exactly ONE subquery at a time.\n5. Write out all terms completely - avoid using abbreviations.\n6. When you have sufficient information, provide the final answer in the following format:\n    <Answer>: [your complete response]\n\nUser Input Query: Does Investment Banking Fees keeps increasing each year between 2006 and 2007 ?\nPlease start!\nOn the last line write 'Final answer: ' followed by only the final value, with no units or explanation.\n"
   },
   {
    "role": "assistant",
    "content": "",
    "tool_calls": [
     {
      "type": "function",
      "function": {
       "name": "solve_subquery",
       "arguments": {
        "subquery": "What is the value of Investment Banking Fees for the years 2006 and 2007?"
       }
      }
     }
    ]
   },
   {
    "role": "tool",
    "content": "Subquery Answer: The value of Investment Banking Fees for the years 2006 and 2007 is as follows:\n\n- **2007**: $41,879  \n- **2006**: $43,703  \n\nThese values are directly taken from the first table in Content 1. The SQL execution result was empty, so we rely on the table content provided in Content 1 for the answer."
   },
   {
    "role": "assistant",
    "content": "",
    "tool_calls": [
     {
      "type": "function",
      "function": {
       "name": "solve_subquery",
       "arguments": {
        "subquery": "What is the percentage change in Investment Banking Fees from 2006 to 2007?"
       }
      }
     }
    ]
   },
   {
    "role": "tool",
    "content": "Subquery Answer: The percentage change in Investment Banking Fees from 2006 to 2007 is **-4%**.\n\nThis is directly taken from the first table in Content 1, where the percentage change for \"Investment Banking Fees\" from 2006 to 2007 is listed as **-4%**. The SQL query returned an empty result, likely due to an issue with the schema mapping or query logic, but the correct answer can be validated from Content 1."
   }
  ],
  "round1": {
   "sub_query": "What is the value of Investment Banking Fees for the years 2006 and 2007?",
   "context": "File name: \nTable name: \n|  | Year Ended |  |  |  |  |\n|  | September 30, 2007 | % Incr. (Decr.) | September 30, 2006 | % Incr. (Decr.) | September 30, 2005 |\n|  | ($ in 000's) |  |  |  |  |\n| Revenues |  |  |  |  |  |\n| Securities Commissions and |  |  |  |  |  |\n| Investment Banking Fees | $ 41,879 | -4% | $ 43,703 | 46% | $ 29,928 |\n| Investment Advisory Fees | 2,846 | 48% | 1,919 | -34% | 2,890 |\n| Interest Income | 4,042 | 11% | 3,647 | 90% | 1,919 |\n| Trading Profits | 5,254 | 41% | 3,720 | 18% | 3,141 |\n| Other | 5,062 | 123% | 2,274 | 156% | 890 |\n| Total Revenues | 59,083 | 7% | 55,263 | 43% | 38,768 |\n| Interest Expense | 1,075 | -27% | 1,467 | 195% | 497 |\n| Net Revenues | 58,008 | 8% | 53,796 | 41% | 38,271 |\n| Non-Interest Expense |  |  |  |  |  |\n| Compensation Expense | 28,071 | -4% | 29,185 | 48% | 19,758 |\n| Other Expense | 23,302 | 17% | 19,867 | 93% | 10,294 |\n| Total Non-Interest Expense | 51,373 | 5% | 49,052 | 63% | 30,052 |\n| Minority Interest | 2,995 |  | 1,887 |  | 2,292 |\nFile name: \nTable name: \n|  | Year Ended |  |  |  |  |\n|  | September 30, 2007 | % Incr. (Decr.) | September 30, 2006 | % Incr. (Decr.) | September 30, 2005 |\n|  | ($ in 000's) |  |  |  |  |\n| Interest Income and Expense |  |  |  |  |  |\n| Interest Income | $ 68,685 | 15% | $ 59,947 | 88% | $ 31,876 |\n| Interest Expense | 59,276 | 25% | 47,593 | 108% | 22,873 |\n| Net Interest Income | 9,409 | -24% | 12,354 | 37% | 9,003 |\n| Expenses | 4,406 | 1% | 4,353 | 43% | 3,041 |\n| Pre-tax Earnings | $ 5,003 | -37% | $ 8,001 | 34% | $ 5,962 |\nFile name: \nTable name: \n|  | 2007 | 2006 | 2005 |\n| Recurring tenant improvements | $45,296 | $41,895 | $60,633 |\n| Recurring leasing costs | 32,238 | 32,983 | 33,175 |\n| Building improvements | 8,402 | 8,122 | 15,232 |\n| Totals | $85,936 | $83,000 | $109,040 |",
   "ranked_units": [
    {
     "cells": [
      "1-10-1",
      "1-10-2",
      "1-10-3",
      "1-10-4",
      "1-10-5",
      "1-11-1",
      "1-11-2",
      "1-11-3",
      "1-11-4",
      "1-11-5",
      "1-12-1",
      "1-12-2",
      "1-12-3",
      "1-12-4",
      "1-12-5",
      "1-14-1",
      "1-14-2",
      "1-14-3",
      "1-14-4",
      "1-14-5",
      "1-15-1",
      "1-15-2",
      "1-15-3",
      "1-15-4",
      "1-15-5",
      "1-16-1",
      "1-16-2",
      "1-16-3",
      "1-16-4",
      "1-16-5",
      "1-17-1",
      "1-17-3",
      "1-17-5",
      "1-5-1",
      "1-5-2",
      "1-5-3",
      "1-5-4",
      "1-5-5",
      "1-6-1",
      "1-6-2",
      "1-6-3",
      "1-6-4",
      "1-6-5",
      "1-7-1",
      "1-7-2",
      "1-7-3",
      "1-7-4",
      "1-7-5",
      "1-8-1",
      "1-8-2",
      "1-8-3",
      "1-8-4",
      "1-8-5",
      "1-9-1",
      "1-9-2",
      "1-9-3",
      "1-9-4",
      "1-9-5"
     ],
     "text": "File name: \nTable name: \n|  | Year Ended |  |  |  |  |\n|  | September 30, 2007 | % Incr. (Decr.) | September 30, 2006 | % Incr. (Decr.) | September 30, 2005 |\n|  | ($ in 000's) |  |  |  |  |\n| Revenues |  |  |  |  |  |\n| Securities Commissions and |  |  |  |  |  |\n| Investment Banking Fees | $ 41,879 | -4% | $ 43,703 | 46% | $ 29,928 |\n| Investment Advisory Fees | 2,846 | 48% | 1,919 | -34% | 2,890 |\n| Interest Income | 4,042 | 11% | 3,647 | 90% | 1,919 |\n| Trading Profits | 5,254 | 41% | 3,720 | 18% | 3,141 |\n| Other | 5,062 | 123% | 2,274 | 156% | 890 |\n| Total Revenues | 59,083 | 7% | 55,263 | 43% | 38,768 |\n| Interest Expense | 1,075 | -27% | 1,467 | 195% | 497 |\n| Net Revenues | 58,008 | 8% | 53,796 | 41% | 38,271 |\n| Non-Interest Expense |  |  |  |  |  |\n| Compensation Expense | 28,071 | -4% | 29,185 | 48% | 19,758 |\n| Other Expense | 23,302 | 17% | 19,867 | 93% | 10,294 |\n| Total Non-Interest Expense | 51,373 | 5% | 49,052 | 63% | 30,052 |\n| Minority Interest | 2,995 |  | 1,887 |  | 2,292 |",
     "n_tok": 578,
     "rerank_score": 3.0959882736206055
    },
    {
     "cells": [
      "2-4-1",
      "2-4-2",
      "2-4-3",
      "2-4-4",
      "2-4-5",
      "2-5-1",
      "2-5-2",
      "2-5-3",
      "2-5-4",
      "2-5-5",
      "2-6-1",
      "2-6-2",
      "2-6-3",
      "2-6-4",
      "2-6-5",
      "2-7-1",
      "2-7-2",
      "2-7-3",
      "2-7-4",
      "2-7-5",
      "2-8-1",
      "2-8-2",
      "2-8-3",
      "2-8-4",
      "2-8-5"
     ],
     "text": "File name: \nTable name: \n|  | Year Ended |  |  |  |  |\n|  | September 30, 2007 | % Incr. (Decr.) | September 30, 2006 | % Incr. (Decr.) | September 30, 2005 |\n|  | ($ in 000's) |  |  |  |  |\n| Interest Income and Expense |  |  |  |  |  |\n| Interest Income | $ 68,685 | 15% | $ 59,947 | 88% | $ 31,876 |\n| Interest Expense | 59,276 | 25% | 47,593 | 108% | 22,873 |\n| Net Interest Income | 9,409 | -24% | 12,354 | 37% | 9,003 |\n| Expenses | 4,406 | 1% | 4,353 | 43% | 3,041 |\n| Pre-tax Earnings | $ 5,003 | -37% | $ 8,001 | 34% | $ 5,962 |",
     "n_tok": 302,
     "rerank_score": -3.187654495239258
    },
    {
     "cells": [
      "0-1-1",
      "0-1-2",
      "0-1-3",
      "0-2-1",
      "0-2-2",
      "0-2-3",
      "0-3-1",
      "0-3-2",
      "0-3-3",
      "0-4-1",
      "0-4-2",
      "0-4-3"
     ],
     "text": "File name: \nTable name: \n|  | 2007 | 2006 | 2005 |\n| Recurring tenant improvements | $45,296 | $41,895 | $60,633 |\n| Recurring leasing costs | 32,238 | 32,983 | 33,175 |\n| Building improvements | 8,402 | 8,122 | 15,232 |\n| Totals | $85,936 | $83,000 | $109,040 |",
     "n_tok": 144,
     "rerank_score": -3.9584014415740967
    },
    {
     "cells": [
      "1-15-1",
      "1-15-2",
      "1-15-3",
      "1-15-4",
      "1-15-5",
      "1-16-1",
      "1-16-2",
      "1-16-3",
      "1-16-4",
      "1-16-5",
      "1-17-1",
      "1-17-3",
      "1-17-5",
      "1-18-1",
      "1-18-2",
      "1-18-3",
      "1-18-4",
      "1-18-5"
     ],
     "text": "File name: \n| Other Expense | 23,302 | 17% | 19,867 | 93% | 10,294 |\n| Total Non-Interest Expense | 51,373 | 5% | 49,052 | 63% | 30,052 |\n| Minority Interest | 2,995 |  | 1,887 |  | 2,292 |\n| Pre-tax Earnings | $ 3,640 | 27% | $ 2,857 | -52% | $ 5,927 |",
     "n_tok": 151,
     "rerank_score": -6.303568363189697
    }
   ]
  }
 },
 "n_tok": 8191,
 "n_llm_calls": 7,
 "n_rounds": 3,
 "n_sql_errors": 0,
 "seconds": 44.346
}
```
