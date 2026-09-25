# GATE 0 — 문서 불일치와 출처 확인 (2026-09-25)

추론(모델 생성) 없음. 파일 읽기·git 기록·정확 McNemar 계산만 했다. 이 폴더의 새 파일 외에는 아무것도 고치지 않았다.
경로는 `rag-agent/` 기준. `X^` = 커밋 X 의 부모(삭제 직전 상태).

산출물(이 폴더):
- `gate0_1_scope.py` → `gate0_1_scope.json`
- `gate0_2_candidate_paths.txt`(후보 경로 322개), `gate0_2_scan.py` → `gate0_2_scan.json`(표 전체 계열 arm 이 있는 결과 169개), `gate0_2_table.py` → `gate0_2_table.md`(그중 답변 EM 이 있는 33개)
- `gate0_2_current_pairs.py` → `gate0_2_current_pairs.json`
- `gate0_4_and_indent.py` → `gate0_4_and_indent.json`

## 사양과 저장소 상태의 차이 (사실만)

| 사양의 전제 | 저장소에서 확인한 것 | 근거 |
|---|---|---|
| "현행 문제정의 §1 ①은 '표가 크면 문맥에 안 들어간다'" | 커밋된 CLAUDE.md 는 §1 에 ①을 남겨 두었고(189행), 같은 파일 §0 에 "①은 폐기(2026-09-24 사용자 재확인)"가 있다(55행). 작업 트리의 미커밋 개정판 CLAUDE.md 는 ①을 폐기로 적었다(25행). | `git show HEAD:rag-agent/CLAUDE.md` 55·189행, `CLAUDE.md`(작업 트리) 25행 |
| 사양이 가리키는 §0, §0.4, §5, §8 | 커밋된 판(HEAD)의 절 번호와 일치한다. 작업 트리 판은 절 번호가 바뀌었다(§6 닫힌 노선, §8 소통). | 같은 곳 |
| "512는 BGE 인코더 입력 한도" | 1,561/86% 수치의 512 는 검색 **토큰 예산**(`--budget 512`, bge-small 토크나이저로 셈)이다. 원 사전등록 문구도 "512 예산 초과". | 0-4 참조 |

## 0-1. README .7074 의 실제 검색 범위

| 항목 | 값 | 근거 |
|---|---|---|
| README 문장 | "검색 906/991=.9142, 답변 701/991=.7074" | `../README.md` 11행 |
| 원본 결과 | `results/evaluation_v2/s3c_v2.json` / `s3c_v2_records.jsonl` / `s3c_v2_answer_retrieved.jsonl` | `analysis/comparison_manifest.five.v2.json` (reference=s3c_v2) |
| `arguments.corpus` | `"split"` | `s3c_v2.json` |
| `n_tables` / `n_units` | 538 / 67,664 | `s3c_v2.json` |
| 인코더 | BAAI/bge-base-en-v1.5 rev a5beb1e3…, max_seq_length 512, 문서 최대 144토큰, 초과 0 | `s3c_v2.json` `embedding_input_audit` |
| 리더 | Qwen2.5-7B-Instruct 4bit rev a09a3545…, neutral, seed 42, max_new_tokens 64, context_limit 32768, 초과 0 | `s3c_v2_answer_retrieved.json` |
| 사전등록 | "고정 조건: HiTab test, corpus=split" | `PREREG-2026-09-12-five-representations.md` 7행 |
| 2026-09-21 규칙 | "본 방법의 답변 정확도는 예산 20 · 질문이 속한 표 안에서만 검색하는 조건으로만 보고한다" | `git show HEAD:rag-agent/CLAUDE.md` 34행 |

records 로 센 값(`gate0_1_scope.json`):

| 모집단 | query count | 문맥 20셀 중 질문의 표가 아닌 표의 셀이 1개 이상 있는 질의 | 비율 | 20셀 전부 다른 표 | 문맥에 들어온 서로 다른 표 수 분포(1/2/3/4/5/6/7/8/9/10개) | 검색 정답 | 답변 정답 |
|---|---:|---:|---:|---:|---|---:|---:|
| 단일 셀 조회(주 모집단) | 991 | 506 | .5106 | 40 | 494/261/125/62/27/11/2/2/6/1 | 906 | 701 |
| 채점 전체 | 1,581 | 791 | .5003 | 73 | 818/403/198/90/38/16/5/4/7/2 | 1,392 | 892 |

- 주 모집단 문맥 셀 19,820개 중 4,872개가 질문의 표가 아닌 표의 셀.
- 답변 파일의 `context_sha256` 이 records 와 991/991 일치 → 리더가 받은 문맥 = 위 records 의 문맥.
- 관련 사실: 현재 HiTab 표 전체 비교에 쓴 300건 표본(`results/ksweep_population_300.json`)은 이 `s3c_v2_records.jsonl`(538표 한 색인)의 gold 순위 구간으로 층화 추출했다(파일의 `method` 필드).

## 0-2. "표 전체 입력이 데이터셋에 따라 본 방법을 이기거나 진다" 관찰의 출처

검색 범위: `git log --all`(로컬 브랜치 5개 + 원격 브랜치 20개 + refs/stash) 에서 내용이
`whole_table|whole table|full[ _-]?table|table_md|통째|표 전체` 와 맞는 *.md/*.json/*.py/*.csv/*.jsonl 경로 322개,
파일 이름에 whole/table_md/fulltable/P3/t_table/cleantable 이 든 경로, `goldtable` 검색, 스태시 3개. 삭제 커밋 f4e6865·e67af75·6e8302a 이전 트리 포함.
작업 트리의 미커밋 결과 2건은 별도로 읽었다.

### (A) 답변 EM 이 있는 표 전체 계열 결과

**A1. 정답 표(문서)를 직접 받는 조건**

| 커밋 | 경로 | 데이터셋·split | 모집단 | query count | 인코더(본 방법 쪽) | 리더 | 표 전체 쪽 직렬화 | 절단 | 표 전체 EM | 셀 arm EM | 짝 검정 |
|---|---|---|---|---:|---|---|---|---|---:|---:|---|
| 미커밋(작업 트리) | `results/fulltable_20260924/hitab_rows.jsonl` 대 `results/fair_filter_20260921/rows.jsonl`(arm=ours, `t_sleaf_gold`) | HiTab test | `ksweep_population_300.json`(단일 셀 조회 991에서 층화 300) | 300 | bge-base-en-v1.5, 질문의 표 안에서만 검색, hybrid α.7, 20셀 | Qwen2.5-7B-Instruct 4bit, neutral, 64토큰, greedy | `chunks.markdown_source`: "Table name: 섹션 제목+페이지 제목" + raw texts 한 행당 "\| a \| b \|", 구분선 없음, 헤더 트리 정보 추가 없음(`rag_agent/serialization/chunks.py:10-26`) | 없음(한계 초과 시 오류, 입력 최대 5,289토큰) | .7900 (237/300) | .7500 (225/300) | 표 전체만 43 : 본 방법만 31, p=.2007 (`gate0_2_current_pairs.json`) |
| 미커밋(작업 트리) | `results/mh_arms/cap300_20260924/fulltable.jsonl` 대 `cell_uniq.jsonl`, 집계 `report.json` | MultiHiertt train | 4그룹 상한 300, seed 20260913 | 882 | bge-base-en-v1.5, 문서 안 검색, 머리글 최종 버전, 20셀 | Qwen3-8B 4bit 비생각, cot, 384토큰, greedy | 문서의 표 전부, 표마다 `markdown_source`, 머리글 규칙 v1, 라벨 없음(`scripts/answer_accuracy_mh.py:90-101`) | 없음(한계 40,960, 입력 최대 4,462토큰) | 가중 .2985 [.2611,.3369] | 가중 .3375 [.2981,.3785] | 차 .039 [−.0047,.0825]; 그룹별 McNemar(본 방법만:표 전체만) 조회1 24:29 p=.58, 조회2+ 42:19 p=.0044, 산술1 7:7 p=1.0, 산술2+ 41:29 p=.19 (`report.json` fulltable.vs_cell, 기준=cell_uniq) |
| `54ed06e^` (처음 87cc2e0, 2026-08-09) | `results/baseline_comparison_llm.json` | HiTab dev | 층화 300(조회 150·집계 150) | 300 | bge-small-en-v1.5, 정답 표 안 순위 | openai:gpt-4o | `markdown_table(raw)`: raw texts 헤더 행 + `\|---\|` + 데이터 행(`bcl_87cc2e0` `scripts/baseline_comparison_llm.py:112-124`) | 예산 1,024(bge-small 토큰) 넘으면 **데이터 행을 뒤에서 버림**, 헤더 유지(같은 파일 234-241). 절단 44건 | .5100 | .5167 | 표 전체만 18 : 셀만 20, p=.871 |
| `54ed06e^` (처음 26ed2ed, 2026-08-09) | `results/baseline_comparison_multihiertt.json` | MultiHiertt train | 표 근거만 있는 질문 300 추출, 네 arm 모두 완료 207 | 207 | bge-small-en-v1.5, 문서의 표 안 순위 | openai:claude-sonnet-5 | 문서의 표 전부 markdown | 예산 1,024 넘으면 줄을 뒤에서 버림(`scripts/baseline_comparison_multihiertt.py:213-215` @26ed2ed). 초과 문서 108 | .4058 | .3382 | 25 : 11, p=.0288, Holm .1729 |
| `54ed06e^` (829173c) | `results/baseline_comparison_multihiertt_sonnet5.json` | MultiHiertt train | 같은 모집단 | 100 | bge-small | claude-sonnet-5 | 같음 | 같음 | .38 | .36 | 파일에 기록 |
| `54ed06e^` (829173c) | `results/baseline_comparison_llm_sonnet5_partial.json` | HiTab dev | 부분 실행("run stopped early on purpose") | 63 | bge-small | api.anthropic.com:claude-sonnet-5 | 같음 | 같음 | .6984 | .6667 | — |
| 2ca522e | `results/verbalize_answer_8b_n60.json` | HiTab dev | hitab_dev_random_sample | 60 | bge-small | groq:llama-3.1-8b-instant | S1 fulltable(`oracle` = 정답 표, `scripts/verbalize_answer_eval.py:123-124` @4f41e95) | 문맥 1,200토큰 | oracle .2833 | rag_1t1c .15 / rowchunk .3167 | — |
| 4f41e95 | `results/verbalize_answer_direct_llama-3.1-8b-instant_n100.json` | HiTab dev | 100 | 100 | bge-small | groq:llama-3.1-8b-instant | 같음 | 1,200토큰 | oracle .27 | rag_1t1c .16 / rowchunk .36 | — |
| 2ca522e | `verbalize_answer_70b_n60.json` / `verbalize_answer_gptoss120b_n60.json` | HiTab dev | 60 요청, 4 / 5건만 평가(요율 제한) | 4 / 5 | bge-small | llama-3.3-70b / gpt-oss-120b | 같음 | 1,200토큰 | .25 / .60 | — | — |
| `54ed06e^` (b91c1d4) | `results/realhitbench_tablemd_vs_s2_gptoss.json` | RealHiTBench | 계산+다단계 97 표본, 2건만 평가(429 오류) | 2 | dense | groq:openai/gpt-oss-120b | 재구성 표 markdown 한 덩어리 | — | 0.0 | 0.0 | — |

**A2. 표를 코퍼스에서 검색해 통째로 넣는 조건(`dump`)** — 33개 파일 전체 행은 `gate0_2_table.md`. 공통 조건(`scripts/corpus_dump_vs_cell.py` @`54ed06e^`):
표 색인 문장 = 제목/캡션 + 헤더 라벨(본문 아님, 133-150행) · 문맥 = 검색 순위대로 표 전체 markdown 을 예산 안에 들어가는 동안 넣음, **예산보다 큰 표는 건너뜀**(1089-1092행) · 토큰은 bge-small 토크나이저 · 인코더 bge-small-en-v1.5 · 리더 local Qwen2.5-7B-Instruct(4bit) · 셀 arm 은 코퍼스 전체 셀 색인.

| 데이터셋·split(한 색인 표 수) | 모집단 | query count | 예산 | 검색기 | dump EM | cell EM | 파일 |
|---|---|---:|---:|---|---:|---:|---|
| HiTab dev (424) | hitab_dev_lookup_single | 100 | 256 / 512 / 1024 / 2048 | dense | .09 / .30 / .47 / .47 | .48 / .43 / .42 / .34 | `corpus_dump_vs_cell_h2h_dense_{256,512,1024,2048}.json` |
| HiTab dev (424) | hitab_dev_lookup_single | 100 | 256 / 512 | hybrid | .09 / .27 | .55 / .56 | `corpus_dump_vs_cell_hyb_hitab_{256,512}.json` |
| HiTab dev (424) | hitab_dev_lookup_all | 830 | 512 | hybrid | .2916 | .6542 | `hyb_hitab_s3c_512.json` |
| HiTab test (414, 색인 누락 버그 시기) | hitab_test_lookup_all | 769 | 512 | dense | .329 | .5917 | `test_s3c_512.json` |
| HiTab dev (424) | hitab_dev_corpus_arith | 175 | 256 / 512 / 2048 | dense | .0229 / .0286·.0457 / .0343 | .04 / .0514·.0857 / .0171 | `corpus_dump_vs_cell_reader_dense_*`, `arith_local_4bit_512_{direct,codegen}.json` |
| MultiHiertt dev (1,310) | table-only, seed 42 | 400 | 256 / 512 / 1024 | dense | .0225 / .055 / .0725 | .1325 / .115 / .1125 | `corpus_dump_vs_cell_h2h_mh_dense_*.json` |
| MultiHiertt dev (1,310) | 같음 | 400 | 512 | hybrid | .0825 | .11 | `corpus_dump_vs_cell_hyb_mh_512.json` |
| AIT-QA dev (113) | 답 문자열로 gold 복원 | 451 | 256 / 512 | dense | .0776 / .0798 | .286 / .3038 | `corpus_dump_vs_cell_h2h_aitqa_dense_*.json` |
| AIT-QA dev (113) | 같음 | 451 | 512 | hybrid | .102 | .3215 | `corpus_dump_vs_cell_hyb_aitqa_512.json` |
| RealHiTBench (536) | 답 문자열로 gold 복원 | 243 | 512 / 1024 | dense | .0494 / .0988 | .1975~.2222 / .2099~.2222 | `rhb_dense_{512,1024}_{s2,s3,s3_casefix}.json` |

(모두 `54ed06e^` 에서 읽음. 2026-08-31 커밋 54ed06e 가 삭제.)

### (B) 표 전체 계열이지만 답변 EM 이 없는 것

| 이름 | 커밋·경로 | 내용 |
|---|---|---|
| `P3_whole_table` (4개 데이터셋) | `f4e6865:rag-agent/results/hpc/{hitab,multihiertt,aitqa,realhitbench}_P3_whole_table_summary.json` | 헤더 경로 포함률(HPC)만. HiTab 은 `hitab_dev_lookup_all` 830(dev). EM 없음 |
| `P3_whole_table` 답변(사전등록 2026-09-08) | `PREREG-2026-09-08-policy-answer-em.md` | P1·P2·P3 test 실행 계획. 정정 2 는 P1·P2·P4 집합만 적음. `t_table_hybrid_answer*` 파일은 기록 어디에도 없음 → **찾지 못함** |
| `goldtable` | `ba1e885^:rag-agent/results/hyb_{hitab,multihiertt,aitqa,realhitbench}_s3c_*_a7.json`, `rhb_vm_s3c_512_a7.json` | reader=None(run.json). 검색 지표·토큰만(예: RealHiTBench goldtable 평균 2,200.4토큰). 사전등록 `PREREG-2026-08-27-rhb-nr-large-tables.md` 의 goldtable 답변 결과 파일은 **찾지 못함** |
| `t_table_hybrid` | `f4e6865:rag-agent/results/retrieval_accuracy/t_table_hybrid.json` | 검색만(0-5 참조) |
| `pipeline_lookup_multihiertt.json` 의 `oracle_table` | `54ed06e^` | 정답 표 안의 셀 문장 조건(flat/S2)이며 표 전체 입력이 아님 |

## 0-3. §5 "헤더 깊이가 이득을 설명한다 — 기각(상관 −0.03/−0.07)" 의 원본

| 항목 | 결과 |
|---|---|
| 처음 나온 곳 | 커밋 `47e2a17`(2026-08-20) 에서 `rag-agent/CLAUDE.md` 56-57행으로 추가: "질의별 이득과 깊이의 상관 −0.03 / −0.07. 버킷도 비단조(깊이 4에서 +.28, 깊이 ≥5에서 +.05)" |
| 데이터셋·모집단·arm 쌍·이득 정의·상관 방식 | 문장에 적혀 있지 않음 |
| 계산 코드·결과 파일 | **찾지 못함.** `47e2a17` 트리와 그 이전 모든 트리에 `corrcoef/pearsonr/spearmanr/pointbiserial` 을 쓰는 파일 없음. 2026-06-01~08-21 모든 커밋의 *.md 에서 −0.03 이 깊이·상관과 같이 나오는 줄은 위 CLAUDE.md 한 줄뿐. `47e2a17` 이 추가한 결과·로그 파일에도 없음 |
| 같은 CLAUDE.md 인접 항목 | "질문 유형 … HiTab 두 유형 유의(+.35 p≈0 / +.13 p=.0002), MultiHiertt 무효과" — 두 값이 HiTab / MultiHiertt 를 뜻하는지 문서로 확인 불가 |
| 조사 안 한 곳 | git 밖의 세션 기록(`~/.claude/projects/…`) |

## 0-4. §1 RealHiTBench "중앙 1,561토큰, 86%가 512 초과" 의 원본

| 항목 | 값 | 근거 |
|---|---|---|
| 처음 나온 곳 | 커밋 `97b9bbc`(2026-08-27) — `PREREG-2026-08-27-rhb-nr-large-tables.md` 20행 "gold 표 중앙값이 1,561토큰이고 86%가 512 예산을 넘는다(최대 12,477)", CLAUDE.md 에 "근거 무대" 로 추가 | `git show 97b9bbc` |
| 원자료 | `results/rhb_dense_512_s3_casefix_records.jsonl` 의 `gold_table_tokens` 필드 | `97b9bbc` 트리 |
| 토크나이저 | **BAAI/bge-small-en-v1.5**, `add_special_tokens=False` | `scripts/baseline_comparison_llm.py` `Budget` 84-99행 @41abb44, `corpus_dump_vs_cell.py` `bud = Budget(args.embed_model)` 778행 @41abb44, run.json `embed_model` |
| 직렬화 | 원본 HTML 파싱 격자 → 추정 헤더 경계 → `\| 제목 \|` 줄(있으면) + 헤더 행 + `\|---\|` + 데이터 행, 줄바꿈 연결 | `corpus_dump_vs_cell.py` 432-436행(md), 900-903행(md_tokens) @41abb44 |
| 512 의 뜻 | 검색 문맥 토큰 예산 `--budget 512`(파일 이름 `_512_`) | 같은 파일, 사전등록 문구 |
| 계산 단위 | **질의 243건 단위**(같은 표가 여러 번 셈) | 재계산 `gate0_4_and_indent.json` |
| 재계산 | 질의 단위: 중앙 1,561, 최대 12,477, 512 초과 208/243 = .856 · 서로 다른 표 184개 단위: 중앙 1,557.5, 512 초과 158/184 = .8587 | `gate0_4_and_indent.json` |
| 모집단 | RealHiTBench, 답 문자열로 gold 셀을 복원한 243건(`realhitbench , gold cells recovered by answer match, ambiguous dropped`) | `rhb_dense_512_s3_casefix.json` @54ed06e^ |

## 0-5. 현재 코드의 표 단위 arm

| 항목 | 값 | 근거 |
|---|---|---|
| 존재 | 있음. `--unit table` | `scripts/retrieval_accuracy.py:93`(UNITS), `:701`(`--unit`) |
| 색인 문장 | 표의 비어 있지 않은 셀마다 템플릿 셀 문장을 만들고 `" \| "` 로 이어 붙인 한 덩어리. 제목은 첫 셀에만 | `scripts/retrieval_accuracy.py:420-425` |
| 512 초과 처리 | `--embed-overflow` 기본값 `error` — 초과 문서가 있으면 실행 중단, `truncate` 는 명시해야 하고 건수를 기록 | `scripts/retrieval_accuracy.py:724`, `rag_agent/retrieve/encoders.py:132-157` |
| 현재 트리 결과 | 없음(`"unit": "table"` 인 결과 JSON 0개) | `grep -rl '"unit": "table"' results` |
| 과거 결과 | `t_table_hybrid.json`: HiTab test, corpus=split(538표 한 색인), 단위 538개, bge-base-en-v1.5, hybrid α.7, 예산 20셀. `accuracy_all_mode` .8386(1,245), `gold_table_in_context` .8261(1,581). 당시 인코더 코드에 입력 길이 감사 없음, 결과 파일에 절단 건수 기록 없음 | `f4e6865:rag-agent/results/retrieval_accuracy/t_table_hybrid.json`(생성 575ff88, 갱신 a049817·cf0d120, 삭제 ba1e885), `cf0d120:rag-agent/rag_agent/retrieve/encoders.py`(overflow/truncat 없음) |

## 0-6. `context_limit` 의 출처와 값

| 항목 | 값 | 근거 |
|---|---|---|
| 정의 | `model.config.max_position_embeddings` | `rag_agent/llm/local_qwen.py:96-98` |
| 사용 | 프롬프트 토큰 + 출력 한도 > context_limit 이면 `ValueError`(절단하지 않음) | `scripts/answer_accuracy.py:95-98`; 호출 `answer_accuracy.py:315`, `answer_accuracy_mh.py:320`, `hitab_fulltable_answer.py:52`, `fair_filter_eval.py:281,289` |
| 입력 절단 | 없음(`complete` 에 truncation 없음) | `rag_agent/llm/local_qwen.py:87-94` docstring |
| Qwen2.5-7B-Instruct rev a09a3545… | 32,768 (`model_max_length` 131,072 는 쓰지 않음) | `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a3545…/config.json:12`, `tokenizer_config.json:202` |
| 기록된 값 | 32,768, 초과 0건 | `results/evaluation_v2/s3c_v2_answer_retrieved.json` |
| Qwen3-8B rev b968826… (MultiHiertt 리더) | 40,960, 초과 0건 | `config.json:14`, `results/mh_arms/cap300_20260924/fulltable.run.json`, `fulltable.json` |

## 보조 확인

- HiTab test 538표 raw `texts` 셀 99,736개 중 앞 공백(들여쓰기)으로 시작하는 셀 0개 → 표 전체 직렬화 문자열에 들여쓰기 계층 정보 없음(`gate0_4_and_indent.json`). MultiHiertt 는 확인 안 함.
- 사양 1번 금지("수치 옆에 출처")를 지키기 위해 위 모든 수치는 표의 근거 열 또는 이 폴더의 JSON 에서 왔다.

---

## 추가 요청 (같은 날) — 산출물 `gate0_extra.py` → `gate0_extra.json`, `gate0_extra_table_metrics.py` → `gate0_extra_table_metrics.json`

### 1. HiTab 300건

| arm | 파일 | query count | 리더 입력 토큰 평균 / 중앙 / 최대 | EM |
|---|---|---:|---|---:|
| 표 전체 | `results/fulltable_20260924/hitab_rows.jsonl` (`reader_input_tokens`, `correct_base`) | 300 | 1,005.2 / 771.5 / 5,289 | .7900 (237) |
| 본 방법 `t_sleaf_gold` | `results/fair_filter_20260921/rows.jsonl` arm=ours (`reader_input_tokens` = 무필터 리더 프롬프트, `scripts/fair_filter_eval.py:278-281,302`) | 300 | 1,397.1 / 1,345 / 3,528 | .7500 (225) |

토큰 = Qwen2.5-7B-Instruct 토크나이저, 채팅 템플릿 포함(`rag_agent/llm/local_qwen.py:87-94`).

### 2. MultiHiertt 882건

| arm | 파일 | query count | 리더 입력 토큰 평균 / 중앙 / 최대 | EM(가중) [95% CI] | EM(비가중) | 조회1 / 조회2+ / 산술1 / 산술2+ |
|---|---|---:|---|---|---:|---|
| 표 전체 | `results/mh_arms/cap300_20260924/fulltable.jsonl` (`n_tok`), `report.json` | 882 | 1,504.8 / 1,391.5 / 4,462 | .2985 [.2611, .3369] | .4286 (378) | .5592 / .5633 / .3099 / .2300 |
| 본 방법 `cell_uniq` | `results/mh_arms/cap300_20260924/cell_uniq.jsonl` (`n_tok`), `report.json` | 882 | 694.3 / 677 / 1,255 | .3375 [.2981, .3785] | .4626 (408) | .5355 / .6400 / .3099 / .2700 |

그룹별 정확 McNemar (`report.json` 의 `cell_only` = 기준 arm `cell_uniq` 만 맞힘, `this_only` = `fulltable` 만 맞힘 — `scripts/mh_cap300_report.py:21,70-73`; jsonl 에서 다시 세어 일치 확인):

| 그룹 | query count | 본 방법(cell_uniq)만 맞힘 | 표 전체(fulltable)만 맞힘 | p (정확) | p (Holm, 4개) |
|---|---:|---:|---:|---:|---:|
| 조회 셀1 | 211 | 24 | 29 | .5831 | 1.0 |
| 조회 셀2+ | 300 | **42** | **19** | .0044 | .0178 |
| 산술 셀1 | 71 | 7 | 7 | 1.0 | 1.0 |
| 산술 셀2+ | 300 | 41 | 29 | .1882 | .5646 |

### 3. "표 검색 97" 후보와 t_table_hybrid

저장소 문서에서 "표 검색 97"이라는 문구는 찾지 못했다. 표 단위 지표로 .97대인 값과 .9737 을 모두 적는다.

| 수치 | 파일(필드) | 데이터셋 | split | 검색 범위 | 색인 단위·인코더 | 지표 정의 | 1위/top-k | query count |
|---|---|---|---|---|---|---|---|---:|
| .9737 | `results/strict_fixed_budget/mh_validation_cell_hv3.3_none_doc_fixed_budget20_summary.json` (`groups.overall.derived_table_strict_recall`); 표 `results/STRICT_RECALL_TABLES-2026-09-15.md:51` | MultiHiertt | validation | 질문의 문서 안(문서 929, 표 3,609) | 셀(s3c, 머리글 v3.3, 라벨 없음), bge-base, hybrid α.7 | 20셀 문맥의 셀들이 속한 표 집합이 정답 표를 **전부** 포함한 질의 비율. 파일: "no separate table retrieval stage exists" (`rag_agent/eval/strict_recall.py:34-49,62-68`) | 20셀 문맥 기준(1위 아님) | 911 |
| .9737 | `results/retrieval_accuracy/t_sleaf_gold.json` (`type_accuracy.multi_cell`, 37/38); 논문 `thesis/src/05_results.md:34` | HiTab | test | 질문의 표 안 | 셀(sleaf), bge-base, hybrid α.7 | 20셀 문맥이 정답 **셀**을 전부 담은 비율(표 검색 아님) | 20셀 | 38 |
| .8386 | `f4e6865:rag-agent/results/retrieval_accuracy/t_table_hybrid.json` (`accuracy_all_mode`) | HiTab | test | 538표 한 색인 | 표 하나 = 셀 문장 이어붙임, bge-base, hybrid α.7 | 문맥(예산 20셀, 마지막 단위 통째)이 정답 **셀**을 전부 담은 비율, 데이터 셀 정답 질의 | 문맥에 든 표 수: 1개 1,472 / 2개 107 / 3개 2 (1,581건) | 1,245 |
| .8261 | 같은 파일 (`gold_table_in_context`) | HiTab | test | 538표 한 색인 | 같음 | 문맥에 정답 표가 들어간 비율 | 같음 | 1,581 |
| .8466 | 같은 records, 단일 셀 조회만 (`gate0_extra_table_metrics.json`) | HiTab | test | 538표 한 색인 | 같음 | 정답 셀 포함 = 정답 표 포함(839/991) | 같음 | 991 |

---

## 추가 요청 2 (같은 날) — 리더 생성 없음. 1번만 인코더 임베딩 실행

### 1. 표 단위 검색 재실행

명령(로그 `table_split_truncate.log`, 요약 `table_split_truncate.json`, 레코드 `table_split_truncate_records.jsonl`, 코드 커밋 d8a23f0 + 미커밋 문서 변경(git_dirty=True)):
`PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python scripts/retrieval_accuracy.py --split test --corpus split --template s3c --unit table --embed-model BAAI/bge-base-en-v1.5 --embed-revision a5beb1e3e68b9ab74eb54cfd186867f64f240e1a --embed-overflow truncate --alpha 0.7 --budget 20 --dump-context 1 --tag table_split_truncate --out-dir results/problem_def_audit_20260925`

| 항목 | 값 | 출처 |
|---|---|---|
| 표 단위 문서 수 | 538 | `table_split_truncate.json` `embedding_input_audit.documents.n` |
| BGE 입력 512 초과(절단) | **466 / 538 = .8662** | 같은 필드 `n_overflow`, `overflow_ratio` |
| 최대 입력 토큰 | 11,741 (접두어·특수 토큰 포함, `rag_agent/retrieve/encoders.py:132-150`) | 같은 필드 `max_tokens` |
| 문맥 셀 평균 | 118.55 | `cells_delivered_mean` |
| 채점 1,581건 정답 표 포함 | .8254 | `gold_table_in_context` |
| 데이터셀 1,245건 정답 셀 전부 포함 | .8378 | `accuracy_all_mode` |

단일 셀 조회 991문항, "문맥에 정답 표 포함"(`gold_table_in_context`) 짝 비교 (`table_unit_vs_s3c_v2.py` → `table_unit_vs_s3c_v2.json`):

| arm | 파일 | 색인 단위 | 정답 표 포함 | 정답 셀 포함 | 문맥 속 표 수(1/2/3개) |
|---|---|---|---:|---:|---|
| 표 단위 | `results/problem_def_audit_20260925/table_split_truncate_records.jsonl` | 표 538개(절단 466) | 839 / 991 | 839 | 907 / 82 / 2 |
| s3c_v2 | `results/evaluation_v2/s3c_v2_records.jsonl` | 셀 67,664개 | 951 / 991 | 906 | (0-1 표 참조) |

| 표 단위만 포함 | s3c_v2만 포함 | 정확 McNemar p |
|---:|---:|---:|
| 2 | 114 | 1.6×10⁻³¹ |

옛 `t_table_hybrid`(`f4e6865`)와 대조: 1,581건 중 판정이 다른 질의 9건, 단일 셀 조회 991건 중 8건(옛 결과만 포함 4 · 새 결과만 포함 4) (`table_unit_old_vs_new.txt`).

### 2. HiTab 300건 본 방법 입력 토큰 분해 (`hitab300_token_breakdown.py` → `hitab300_token_breakdown.json`)

- 문장 5,822줄(질의당 평균 19.41줄)을 구성요소로 다시 만들어 기록된 문장과 바이트 일치 확인. 토큰 합계가 질의마다 `reader_input_tokens` 와 일치.
- 토크나이저 Qwen2.5-7B-Instruct a09a3545…, 채팅 템플릿 포함. 토큰마다 글자 다수결로 범주 배정.

| 범주 | 내용 | 질의당 평균 토큰 |
|---|---|---:|
| 표 제목 | 섹션 제목 + ToTTo 페이지 제목 | 502.8 |
| 헤더 경로 | 행 경로 + 열 경로(" > " 포함) | 297.5 |
| 잎 라벨 앞머리 | 문장 앞 "행 잎 / 열 잎:" 의 라벨 | 129.3 |
| 셀 값 | | 73.0 |
| 문장 틀 | In the table ', among, the value of, is, ., ": ", " / " | 296.1 |
| 질문 | "Question: " + 질문 | 24.3 |
| 나머지 | 시스템 프롬프트·채팅 템플릿·Context:·줄바꿈·Answer: | 74.1 |
| 합계 | | 1,397.1 |

### 3. MultiHiertt cap300 조회 셀2+ 그룹 (`mh_lookup_m2_tokens.py` → `mh_lookup_m2_tokens.json`)

토큰 = Qwen3-8B 토크나이저, cot 시스템 프롬프트·채팅 템플릿 포함(`scripts/answer_accuracy_mh.py:314-318`).

| arm | 파일 | query count | 입력 토큰 평균 / 중앙 / 최대 | 정답 수 |
|---|---|---:|---|---:|
| 표 전체 | `results/mh_arms/cap300_20260924/fulltable.jsonl` (layer=lookup_m2+) | 300 | 1,563.1 / 1,478.5 / 4,462 | 169 |
| 본 방법 cell_uniq | `results/mh_arms/cap300_20260924/cell_uniq.jsonl` (layer=lookup_m2+) | 300 | 698.8 / 689.5 / 1,089 | 192 |

---

## 추가 요청 3 (같은 날) — 토크나이저만 (`corpus_token_totals.py` → `corpus_token_totals.json`, 경고 로그 `corpus_token_totals.log`)

조건: Qwen/Qwen2.5-7B-Instruct a09a3545… 토크나이저, `add_special_tokens=False`, 채팅 템플릿·시스템 프롬프트 없음.
직렬화 = `rag_agent/serialization/chunks.py:10-26` `markdown_source`.
HiTab 은 `scripts/hitab_fulltable_answer.py:47-50` 과 같이 제목 = 섹션 제목 + 페이지 제목.
MultiHiertt 는 cap300 과 같은 split=train, 머리글 v1, 라벨 없음, 문서 = 그 문서 표 전부를 "\n" 으로 이음(`scripts/answer_accuracy_mh.py:90-101`). 본문 문단은 포함하지 않음(fulltable 문맥에 없음).
단위 사이 구분자 "\n\n". `" |\n\n"` 이 토큰 1개(`Ġ|ĊĊ`)라 구분자 추가 토큰은 0 — 개별 합과 이어 붙인 값이 같다.

| 대상 | 단위 수 | 총 토큰 | 32,768 의 배수 | 단위당 최소 / 중앙 / 최대 | 32,768 초과 단위 |
|---|---:|---:|---:|---|---:|
| HiTab test 표 | 538 | 427,228 | 13.04 | 89 / 640.5 / 5,201 | 0 |
| MultiHiertt train 문서 | 2,908 | 4,064,348 | 124.03 | 345 / 1,299 / 5,166 | 0 |
| MultiHiertt train 표 | 11,793 | 4,064,348 | 124.03 | 31 / 240 / 3,469 | 0 |
| MultiHiertt cap300 882건의 문서 | 882 | 1,237,945 | 37.78 | 452 / 1,291.5 / 4,362 | 0 |

MultiHiertt train 문서 2,908 = `scripts/mh_arms.py:136-171` `load_population("train")` 의 표 근거만 있는 질의 2,908개의 문서(text 근거가 필요한 4,922건 제외). 채점 모집단 2,871 은 이후 gold 해석 제외 뒤의 수.

작은 단위부터 채웠을 때 32,768 안에 들어가는 최대 단위 수(실제 이어 붙인 문자열로 재확인):

| 대상 | 최대 단위 수 | 전체 대비 | 그때 토큰 |
|---|---:|---:|---:|
| HiTab test 표 | 126 / 538 | .2342 | 32,641 |
| MultiHiertt train 문서 | 59 / 2,908 | .0203 | 32,320 |
| MultiHiertt train 표 | 502 / 11,793 | .0426 | 32,748 |
| MultiHiertt cap300 문서 | 50 / 882 | .0567 | 32,276 |
