# 사후 조건 등록: HiTab s3c 답변 정확도, 300건 (2026-09-26)

실행 전에 커밋한다. **사후 조건**이다 — s3c·sleaf 검색 결과(991건 표 안 s3c 955, sleaf 950,
불일치 12:7)와 sleaf 답변 결과(.7500)를 본 뒤 등록한다. 방향 예측은 하지 않는다.

## 조건

- 질의: `results/ksweep_population_300.json` 300건(seed 42) — sleaf .7500 과 같은 질의.
- 검색 records: `results/retrieval_accuracy/t_s3c_gold_labelabl_records.jsonl`
  (s3c, 질문이 속한 표 안, 예산 20셀, hybrid α .7, bge-base-en-v1.5, 코드 9bd9ab4).
- 리더: sleaf .7500(`results/fair_filter_20260921/rows.jsonl`, arm `ours`, `correct_base`)과 같은
  `scripts/fair_filter_eval.py` 무필터 경로. `local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit`,
  `PROMPTS["neutral"]`, 최대 64토큰, temperature 0, 배치 1(`complete`).
- 생성: 질의당 무필터 리더 1회만. 필터 LLM·필터 후 리더는 돌리지 않는다(`--no-filter`).
- 채점: `hitab_exact_match_text`.

## 분석

- s3c 답변 정확도(300).
- sleaf 대 s3c: 같은 질의끼리 McNemar 정확 이항(`fair_filter_eval.mcnemar`).
- 이 300건의 s3c 검색 정확도(records `correct`)를 함께 적는다.

## 명령

    PYTHONPATH=. .venv/bin/python scripts/fair_filter_eval.py --arms s3c --no-filter \
      --out results/s3c_answer_hitab300_20260926/rows.jsonl

## 코드 변경

`scripts/fair_filter_eval.py`: `EXTRA_ARMS = {"s3c": "t_s3c_gold_labelabl"}`(`--arms` 로만 선택),
`--no-filter`(무필터 리더만, 필터 열 null). 기본 실행과 기존 무필터 경로는 바뀌지 않는다.

## 결과

(실행 후 기록)
