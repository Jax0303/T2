# 사전등록: HiTab 538표 한 색인 조건의 답변 정확도, 300건 (2026-09-28)

실행 전에 커밋한다. 사용자 지시(2026-09-28)로 1회만 실행한다. 결과는 방향과 관계없이 원고(5장 답변 절, 6.5절)에 싣는다.

## 목적

질문의 표를 모르고 표 전체를 리더에 넣을 수 없는 조건에서 답변 정확도를 잰다. HiTab test 538표는 합쳐서
427,228토큰으로 HiTab 리더 입력 한도 32,768의 13.04배다(`results/problem_def_audit_20260925/corpus_token_totals.json`,
Qwen2.5-7B-Instruct 토크나이저, 표 전체 조건과 같은 마크다운 직렬화). 지금까지 HiTab 답변(표 5-4)은 질문의 표 하나
안에서만 검색한 조건이다.

## 표본

`results/ksweep_population_300.json`의 단일 셀 조회 300건(seed 42). 표 5-4와 같은 문항이다.

## 조건 (2개)

둘 다 test 538표 전체를 한 색인에 넣고 검색한 기존 기록을 그대로 쓴다(2026-09-26 재실행, 표 5-3의 원천).
검색 설정은 표 5-4의 표 안 검색과 같다: hybrid α .7, `BAAI/bge-base-en-v1.5`, 예산 20셀(서로 다른 셀 20개에 이를 때까지
단위를 통째로 넣는 규칙). 색인 텍스트도 표 안 검색과 같다(corpus_text_sha256 일치).

| 조건 | 검색 기록 | records_sha256 | 색인 텍스트 sha256 (표 안 기록과 같음) |
|---|---|---|---|
| 본 방법(s3c) | `results/rerun_20260926/hitab/hitab_test_split_s3c_records.jsonl` | `ad276bdb…` | `ff6b0d27…` (`t_s3c_gold_labelabl`) |
| 1,000자 청크 | `results/rerun_20260926/hitab/hitab_test_split_chunk_records.jsonl` | `aa979ff5…` | `d346c329…` (`t_chunk_s3c_gold_v2`) |

- 300건 전부 두 기록에 있고 문맥(`context`)이 있다. 새로 돌릴 검색은 0건이다.

## 리더·채점 (표 5-4와 같음)

`scripts/fair_filter_eval.py` 무필터 경로: `local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit`, `PROMPTS["neutral"]`,
최대 64토큰, temperature 0, 배치 1(`complete`). 채점 `hitab_exact_match_text`. 코드 변경 없음.

## 명령 (rag-agent/ 에서, 조건마다 1회)

    PYTHONPATH=. .venv/bin/python scripts/fair_filter_eval.py --arms s3c --no-filter \
      --records results/rerun_20260926/hitab/hitab_test_split_s3c_records.jsonl \
      --out results/hitab538_answer_20260928/s3c_rows.jsonl > results/hitab538_answer_20260928/s3c_run.log 2>&1
    PYTHONPATH=. .venv/bin/python scripts/fair_filter_eval.py --arms chunk --no-filter \
      --records results/rerun_20260926/hitab/hitab_test_split_chunk_records.jsonl \
      --out results/hitab538_answer_20260928/chunk_rows.jsonl > results/hitab538_answer_20260928/chunk_run.log 2>&1
    PYTHONPATH=. .venv/bin/python results/hitab538_answer_20260928/analyze.py

## 분석

- **검정 (비교 1개, 보정 없음):** 본 방법 대 1,000자 청크 답변 정답 여부, 같은 문항끼리 정확 McNemar(양측 이항), α=.05.
  예측 방향: 본 방법이 높다.
- **기술 통계 (검정 없음):** 두 조건의 검색 성공 수(정답 셀 전부 포함, records `correct`), 리더 입력 토큰 평균,
  본 방법 538표 조건 정답 수와 표 안 조건 정답 수(237, `results/s3c_answer_hitab300_20260926/rows.jsonl`)의 차이와 불일치 수.
- 집계 코드: `results/hitab538_answer_20260928/analyze.py` (이 등록과 함께 커밋) → `analyze.json`.

## 사전 노출 공개

- 538표 조건 검색 정확도(991건)는 원고 표 5-3에 있다: 본 방법 .9142, 1,000자 청크 .7659.
- 등록 전에 이 300건의 검색 성공 수를 세었다: 본 방법 274, 1,000자 청크 229.
- 2026-09-18에 538표 조건 300문제 예비 답변 실험을 했고 결과 파일이 있다. 조건은 이전 변형 sleaf(잎 라벨), TableRAG(Chen)
  셀 검색 재구현 leaf·path, TableRAG(Yu) 청크이며, 1,000자 청크와 s3c는 없다. 문항은 데이터셋 순서 앞 300건(이 표본과
  66건 겹침, 머리글이 답인 문항 포함), 프롬프트 `base`, 리더 Qwen2.5-7B·Qwen3-8B. 등록자는 그 결과를 안다
  (Qwen2.5-7B: sleaf 538표 .6167 대 표 안 .6533, TableRAG(Yu) .4400, leaf .0433, path .0500). 파일:
  `results/retrieval_accuracy/t_{sleaf,tablerag_leaf_v2,tablerag_path_v2}_split_answer_retrieved{,_qwen3_8b}_pilot300.json`,
  `results/evaluation_v2/huawei_char_v2_answer_retrieved{,_qwen3_8b}_pilot300.json`.

## 결과

(실행 뒤 이 아래에 적는다.)
