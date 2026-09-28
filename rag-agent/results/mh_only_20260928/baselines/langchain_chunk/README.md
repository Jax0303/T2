# `langchain_chunk` — LangChain 공식 RAG 튜토리얼 파이프라인 비교군 (MultiHiertt train)

코드 `scripts/baselines/langchain_chunk.py`, 실행 `run.sh`. 기존 스크립트·결과 파일은 건드리지 않았다.

## 무엇을 재현했나

LangChain 공식 튜토리얼(https://docs.langchain.com/oss/python/langchain/rag)의 파이프라인 그대로:

    문서 → RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        → 임베딩 → InMemoryVectorStore → similarity_search(question)  (기본 k=4 청크)
        → rlm/rag-prompt → LLM → 답

- 분할기: `langchain_text_splitters.RecursiveCharacterTextSplitter`, 튜토리얼 값 그대로 (`split_documents`).
- 벡터 저장소: `langchain_core.vectorstores.InMemoryVectorStore` (코사인 유사도, 튜토리얼과 같은 클래스).
  질문마다 그 문서의 청크만 넣고 `similarity_search(question, k=30)` 을 불러 상위 30 을 기록하고,
  **앞 4 개**가 튜토리얼의 `similarity_search(question)`(k=4) 결과이자 LLM 에 넘긴 문맥이다(같은 정렬, 같은 점수).
- 프롬프트: rlm/rag-prompt 원문 + 표시 문장(아래 이탈 2), human 메시지 하나(system 메시지 없음 — 튜토리얼과 같음).
- 문맥 문자열: 튜토리얼처럼 `"\n\n".join(청크 본문)`.

## 같은 조건(본 방법·다른 비교군과 동일)

- 검색 범위: 질문이 속한 MultiHiertt train 문서의 **표만**, 머리글 규칙 v3.3u(`mh_arms.build_tables`), 라벨 없음.
- 표 렌더링: `rag_agent/serialization/chunks.markdown_source` — `fulltable` 조건·`trag_hetero` 와 같은 markdown
  (표 이름이 없으므로 'Table name: ' 줄 제거, 표 사이 빈 줄). 행 문자열은 `chunk` 비교군의 `raw_lines` 와 같다
  (`chunk` 는 여기에 `# ` 제목 줄과 머리글 블록 반복을 얹는데, 이 비교군은 튜토리얼 그대로라 얹지 않는다).
  검증: 882 부분집합 앞 3 문서에서 `full_tables()` 를 빈 줄로 이은 문자열과 바이트 단위 일치.
- 셀 배달 판정: `chunks.table_chunks` 규칙 그대로 — 값이 통째로 청크 안에 있는 데이터 셀만 배달로 친다
  (`add_start_index` 위치로 span 대조, 3 문서 509 span 이 격자 값과 일치 확인). 셀 id 는 `gold_ids` 와 같은
  펼친 격자 좌표 "표-행-열".
- 평가 집합: 검색 = `retrieval/mh_train_s3c_records.jsonl` 의 doc 채점 2,885 문항(코드가 집합 일치를 확인),
  답변 = `results/mh_arms/cap300_20260924/cell_uniq.jsonl` 의 882 문항(query_id 순).
- 채점: `rag_agent/eval/multihiertt_em.mh_exact_match`(공식 EM) + `rag_agent/eval/multihiertt_f1.mh_f1`(공식 F1, 이미
  저장소에 포팅돼 있어 새로 옮기지 않았다). 정답은 `official_answers("train")`(yilunzhao/MultiHiertt 공식 파일).
- 리더: Qwen3-8B 4bit 비생각 모드, greedy, `LocalQwenLLM.complete_batch`(continuous batching, batch 128), seed 42,
  max_new_tokens 384 — 본 방법 답변 실행과 같은 생성 설정.

## 튜토리얼과 다른 점 (튜토리얼 값 ↔ 여기)

| # | 항목 | 튜토리얼 | 여기 | 이유 |
|---|------|----------|------|------|
| 1 | 임베딩 | 클라우드 임베딩(예: OpenAI `text-embedding-3-large`) | `BAAI/bge-base-en-v1.5`, 질의 접두어 "Represent this sentence for searching relevant passages: ", 문단 접두어 없음 (`rag_agent/retrieve/encoders.default_encoder`) | 폐쇄 환경, 본 방법과 같은 인코더. `langchain-huggingface` 가 .venv 에 없어 `langchain_core.embeddings.Embeddings` 를 4 줄로 구현해 같은 인코더를 감쌌다(`HuggingFaceEmbeddings` 와 달리 bge 질의 접두어를 붙인다) |
| 2 | LLM | 클라우드 챗 모델 | Qwen3-8B 4bit, 비생각, greedy, max_new_tokens 384 | 폐쇄 환경, 본 방법과 같은 리더. 튜토리얼은 출력 길이 제한이 없다 |
| 3 | 프롬프트 끝 | "...keep the answer concise.\nQuestion: … \nContext: … \nAnswer:" | 같은 문장 뒤에 " On the last line write 'Final answer: ' followed by only the final value, with no units or explanation." 를 덧붙임 | 채점이 숫자 EM 이라 `answer_accuracy_mh.py` 와 같은 `Final answer:` 추출 규약이 필요. 추출 함수도 같은 것(`answer_accuracy_mh.extract`; 표시가 없으면 출력 전체로 채점) |
| 4 | 문서 | 웹 페이지 본문(`WebBaseLoader`) | 질문 문서의 표 markdown 만 | 같은 조건 제약(표만, 문서 범위) |
| 5 | k | `similarity_search(question)` 기본 4 | 4 (기록용으로 30 까지 순위만 남김) | 없음 |

그 밖(분할기·분할 값·저장소·유사도·문맥 결합·프롬프트 본문·메시지 구조)은 튜토리얼 그대로다.
머리글 반복·행 보존 묶기·"File name:" 접두어는 넣지 않았다(다른 비교군의 것).

## 패키지 버전 (.venv, Python 3.12.3)

langchain-core 1.6.2, langchain-text-splitters 1.1.2 (langchain 본체·langchain-huggingface 없음),
sentence-transformers 5.6.0, transformers 5.12.1, torch 2.12.1+cu130, bitsandbytes 0.50.1, scipy 1.18.0.

## 출력 파일

- `langchain_chunk_records.jsonl` — 문항마다 `query_id, kind, layer, m, question, gold_ids, n_chunks_in_doc,
  ranked_units[≤30]{cells, text, n_tok(Qwen3 토크나이저)}, context(LLM 에 준 문자열 = 상위 4), correct(정답 셀 전부 ⊆ 상위 4 셀),
  any_DIAGNOSTIC, cells_in_context`. 요약 `langchain_chunk.json`(층별 정확도·배달 셀 수·문맥 토큰).
- `langchain_chunk_answer.jsonl` — `query_id, layer, kind, m, question, retrieval_correct, pred, answer, answer_correct(EM),
  f1, raw, marker_found, n_tok(프롬프트 토큰, 채팅 템플릿 포함), n_llm_calls(=1), cells_in_context, context_sha256, seconds`.
  요약 `langchain_chunk_answer.json`(층별 EM·F1·입력 토큰, 프롬프트 원문, 리더 메타).

## 연기 시험 (2026-09-29, `./run.sh --limit 3`, 결과는 `smoke3/`)

882 부분집합 앞 3 문항. 검색 25 s(모델 로드 포함, 임베딩 0.9 s), 답변 24 s(모델 로드 포함, 생성 15.8 s).
검색 정확도 2/3, 청크 4~6 개/문서, 문맥 1,268~1,890 토큰.

| query_id | 유형 | 검색 | 예측 | 정답 | EM | F1 |
|---|---|---|---|---|---|---|
| 007a44f1 | lookup m2 | 맞음 | `-4%` | `no` | 0 | 0.00 |
| 00fa4ad4 | lookup m3 | 틀림 | `Other investments` | `Fixed maturities, Commercial mortgage and other loans` | 0 | 0.22 |
| 01208501 | arith m2 | 맞음 | `793000 65385` | `429192.5` | 0 | 0.00 |

## 전체 실행

    cd rag-agent && setsid nohup results/mh_only_20260928/baselines/langchain_chunk/run.sh > results/mh_only_20260928/baselines/langchain_chunk/full.log 2>&1 &

(`run.sh` 가 두 단계를 각각 `flock .gpu.lock` 으로 감싼다. 출력 파일이 이미 있으면 덮어쓰지 않고 멈춘다.)

예상 시간: 검색 2,885 문서 ≈ 문서당 0.3 s + 로드 → 약 20 분. 답변 882 ≈ 입력 평균 약 1.4k 토큰으로
`fulltable`(1,505 토큰, 5,789 s)·`chunk`(581 토큰, 4,854 s) 사이 → 약 1.5~2 시간. 합계 약 2~2.5 시간.
