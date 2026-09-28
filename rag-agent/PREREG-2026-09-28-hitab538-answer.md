# 사전등록: HiTab 538표 한 색인 조건의 답변 정확도, 300건 (2026-09-28)

실행 전에 커밋한다. 사용자 지시(2026-09-28)로 1회만 실행한다. 결과는 방향과 관계없이 원고(5장 답변 절, 6.5절)에 싣는다.

## 개정 이력 (모두 답변 생성 전, 생성 0건)

- `03ef100` 최초 등록: 본 방법 대 1,000자 청크, 비교 1개.
- `03ef100`→`1811ff0`: `CLAUDE.md` §3.3(HiTab 답변은 표 안 조건만) 예외 문단 한 개 추가. 조건·분석 변경 없음.
- 이번 개정(사용자 지시 2026-09-28): 표 단위 색인 1위 표 전체 입력 팔 추가, 확증 비교 2개(Holm), `CLAUDE.md` §3.3 개정 반영,
  재사용 검색 기록 확인 결과, 사전 노출 상세, 예측 방향 근거. `03ef100` 때 걸어 둔 자동 실행은 GPU 대기 중에 멈췄다(생성 0건).

## 목적

질문의 표를 모르고 표 전체를 리더에 넣을 수 없는 조건에서 답변 정확도를 잰다. HiTab test 538표는 합쳐서
427,228토큰으로 HiTab 리더 입력 한도 32,768의 13.04배다(`results/problem_def_audit_20260925/corpus_token_totals.json`,
Qwen2.5-7B-Instruct 토크나이저, 표 전체 조건과 같은 마크다운 직렬화). 지금까지 HiTab 답변(표 5-4)은 질문의 표 하나
안에서만 검색한 조건이다. 538표 조건에는 표 전체를 입력하는 대조군을 둘 수 없으므로 검색 방법끼리만 비교한다.

**보고 규칙:** `CLAUDE.md` §3.3(2026-09-28 개정): HiTab 답변은 질문의 표 안 검색 조건이 주 조건이다. 538표 한 색인 조건은
이 등록의 조건으로만 보고하고, 두 조건의 수치를 한 표·한 Holm 묶음에 섞지 않는다.

## 표본

`results/ksweep_population_300.json`의 단일 셀 조회 300건(seed 42). 표 5-4와 같은 문항이다.

## 조건 (3개)

셋 다 test 538표 전체를 한 색인에 넣고 검색한 기존 기록을 그대로 쓴다(2026-09-26 재실행, 표 5-3의 원천).

| 조건 | 검색 기록 (`results/rerun_20260926/hitab/`) | records_sha256 | 색인 텍스트 sha256 | 리더 입력 |
|---|---|---|---|---|
| 본 방법(s3c) | `hitab_test_split_s3c_records.jsonl` | `ad276bdb…` | `ff6b0d27…` (표 안 `t_s3c_gold_labelabl`과 같음) | 검색 문맥(예산 20셀) |
| 1,000자 청크 | `hitab_test_split_chunk_records.jsonl` | `aa979ff5…` | `d346c329…` (표 안 `t_chunk_s3c_gold_v2`와 같음) | 검색 문맥(예산 20셀) |
| 표 단위 1위 표 | `hitab_test_split_table_records.jsonl` | `fcb89d53…` | `be0db708…` | 검색 1위 표 전체 |

- 본 방법·청크: 예산 20셀(서로 다른 셀 20개에 이를 때까지 단위를 통째로 넣는 규칙). 표 5-4의 표 안 답변과 같다.
- 표 단위 1위 표: 표 단위 색인의 검색 1위 표 하나만 넣는다. 1위 = 기록의 `context_units[0]`(`budget_select`가 검색 순위
  순서로 쌓는다). 입력은 표 전체 조건(표 5-4 참고 행)과 같은 마크다운(`chunks.markdown_source`, 이름 = 섹션 제목 + 페이지 제목)이다.
  따라서 1위 표가 질문의 표인 문항의 입력은 표 전체 조건과 같다. 검색 성공 = 1위 표가 정답 셀의 표. 기존 기록의 20셀 문맥에는
  1위 표가 20셀 미만이라 2위 표까지 들어간 문항이 있으나(300건 중 25건) 이 팔은 1위 표만 쓴다. 표 단위 색인 임베딩은 입력
  512토큰에서 잘렸다(538표 중 466표, 6.5절).

### 재사용 검색 기록 확인 (2026-09-28)

| 항목 | 세 기록 공통 |
|---|---|
| 생성 커밋 | `provenance.git_commit` = `378db41` (히스토리 다시 쓰기 뒤 `726dc13`, `results/rerun_20260926/README-large-files.md`), `git_dirty` = false |
| 템플릿 | `s3c` (HiTab 최종 템플릿, `CLAUDE.md` §2.1), `label_mix` 없음 |
| 인코더 | `BAAI/bge-base-en-v1.5` (질의 접두어 있음) |
| α | 0.7 (hybrid) |
| 예산 | 20셀 (표 단위 1위 표 팔은 1위 표만 쓰므로 무관) |
| 범위 | `corpus` = `split` (538표 한 색인), `load_evidence` 검증 통과(records_sha256·context_version 2) |

- 새로 검색한 문항: 세 기록 모두 0건 (`results/hitab538_answer_20260928/new_retrieval_query_ids.json`).
- 2026-09-18 잎 라벨 예비 파일과 sleaf 기록은 쓰지 않는다. 표 단위 팔은 질문·정답·정답 셀도 표 단위 기록에서 읽는다.

## 리더·채점 (표 5-4와 같음)

`local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit`, `PROMPTS["neutral"]`, 최대 64토큰, temperature 0, 배치 1(`complete`),
채점 `hitab_exact_match_text`. 본 방법·청크는 `scripts/fair_filter_eval.py` 무필터 경로, 표 단위 1위 표는
`scripts/hitab_fulltable_answer.py`(표 전체 조건과 같은 코드).

**코드 변경(이 개정과 함께 커밋):** `scripts/hitab_fulltable_answer.py`에 `--top1-records`(표 단위 기록의 1위 표를 넣는다,
질문·정답도 그 기록에서)와 `--out` 추가. 옵션 없이 돌리면 기존 표 전체 조건과 같다.

## 명령 (rag-agent/ 에서, 조건마다 1회)

    D=results/hitab538_answer_20260928; R=results/rerun_20260926/hitab
    PYTHONPATH=. .venv/bin/python scripts/fair_filter_eval.py --arms s3c --no-filter \
      --records $R/hitab_test_split_s3c_records.jsonl --out $D/s3c_rows.jsonl > $D/s3c_run.log 2>&1
    PYTHONPATH=. .venv/bin/python scripts/fair_filter_eval.py --arms chunk --no-filter \
      --records $R/hitab_test_split_chunk_records.jsonl --out $D/chunk_rows.jsonl > $D/chunk_run.log 2>&1
    PYTHONPATH=. .venv/bin/python scripts/hitab_fulltable_answer.py \
      --top1-records $R/hitab_test_split_table_records.jsonl --out $D/table_top1_rows.jsonl > $D/table_top1_run.log 2>&1
    PYTHONPATH=. .venv/bin/python $D/analyze.py

(`$D/run.sh`가 다른 세션의 GPU 작업이 끝나기를 기다렸다가 이 순서로 돈다.)

## 분석

- **확증 비교 2개:** 같은 문항끼리 정확 McNemar(양측 이항), α=.05, 두 비교를 한 묶음으로 Holm 보정.
  1. 본 방법 대 1,000자 청크. 예측: 본 방법이 높다.
  2. 본 방법 대 표 단위 1위 표. 방향 예측 없음.
- **예측 방향 근거:** 538표 조건 검색 정확도(991건, 표 5-3) 본 방법 91.4% 대 1,000자 청크 76.6%.
- **기술 통계 (검정 없음):** 세 조건의 검색 성공 수(본 방법·청크 = 정답 셀 전부 포함, records `correct`; 표 단위 = 1위 표가
  정답 셀의 표)와 리더 입력 토큰 평균. 본 방법 538표 조건 정답 수와 표 안 조건 정답 수(237,
  `results/s3c_answer_hitab300_20260926/rows.jsonl`)의 차이와 불일치 수.
- **점검:** 표 단위 1위 표가 정답 표인 문항에서 입력 토큰 수와 출력이 표 전체 조건(`results/fulltable_20260924/hitab_rows.jsonl`)과
  같은지 센다. 다르면 보고한다.
- 집계 코드: `results/hitab538_answer_20260928/analyze.py`(이 개정과 함께 커밋) → `analyze.json`.

## 사전 노출 공개

- 538표 조건 검색 정확도(991건)는 원고 표 5-3에 있다: 본 방법 .9142, 1,000자 청크 .7659, 표 단위 .8466(예산 20셀 문맥 기준,
  1위 표만의 값이 아니다).
- 등록 전에 이 300건의 검색 성공 수를 세었다: 본 방법 274, 1,000자 청크 229. 표 단위 1위 표의 성공 수는 세지 않았다.
  표 단위 기록에서 20셀 문맥에 표가 2개인 문항 수(25)는 셌다(정답 여부와 무관한 개수).
- 2026-09-18 538표 조건 300문제 예비 답변 실험: 조건은 이전 변형 sleaf(잎 라벨), TableRAG(Chen) 셀 검색 재구현 leaf·path,
  TableRAG(Yu) 청크이며, 1,000자 청크·s3c·표 단위는 없다. 문항은 데이터셋 순서 앞 300건(이 표본과 66건 겹침, 머리글이 답인
  문항 포함), 프롬프트 `base`, 리더 Qwen2.5-7B·Qwen3-8B.
  - **파일을 연 적:** 있다. 2026-09-28 오전 이 등록을 쓰기 전에 요약 JSON 5개
    (`results/retrieval_accuracy/t_sleaf_{gold,split}_answer_retrieved_pilot300.json`,
    `t_tablerag_{leaf,path}_v2_split_answer_retrieved_pilot300.json`, `results/evaluation_v2/huawei_char_v2_answer_retrieved_pilot300.json`)의
    정확도 값을 출력해 봤고, Qwen2.5·Qwen3 파일 8개의 jsonl 에서 `query_id`만 읽어 겹침(66건)을 셌다.
  - **본 수치 (Qwen2.5-7B, 답변 / 검색):** sleaf 표 안 .6533 / .95, sleaf 538표 .6167 / .9167, TableRAG(Yu) 538표 .4400 / .70,
    leaf 538표 .0433 / .0933, path 538표 .0500 / .20 (조회 유형별 값 `all_mode`·`any_mode`도 함께 출력됨).
  - **기록(프로젝트 메모)으로 아는 수치 (Qwen3-8B 답변):** sleaf 538표 .5933, 표 안 .6333; leaf 538표 .0367, path .0567;
    TableRAG(Yu) .3833; Qwen2.5 표 안 leaf .1133, path .1400.

## 결과

(실행 뒤 이 아래에 적는다.)
