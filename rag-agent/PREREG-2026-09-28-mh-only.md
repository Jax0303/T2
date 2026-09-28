# 사전등록: MultiHiertt 단독 재평가 (지도교수 지시, 2026-09-28 17:3x)

실행 전에 커밋한다. GATE 마다 멈추고 보고한다. test(2,885) 실행은 단계마다 1회, 규칙 선택은 dev(332)에서만. 수치는 `results/` 결과 파일에서만.
산출물 `results/mh_only_20260928/`. HiTab 은 평가·원고에서 뺀다(원고 수정은 GATE 들 뒤 별도 지시).

## 앞선 실행의 처리 (공개)

같은 날 오후 `PREREG-2026-09-28-fair-mh.md`(f693c7f) 아래에서 다음이 이미 돌았다. 이 등록의 규칙과 어긋나는 것은 결과로 쓰지 않는다.
- 재검색 3 arm(본 방법 s3c, 1,000자 청크, TableRAG(Yu) 청킹 이식; `results/fair_mh_20260928/retrieval/`, 캐시 `.cache/rerun_20260926_mh`, 커밋 7821b46):
  기존 기록과 문항별 `correct`·문맥 불일치 0. 순위 순 상위 20 단위의 셀 좌표(`ranked_units`)와 정답 좌표(`gold_ids`)가 추가됐다. → **GATE 1 에서 그대로 쓴다**(재실행 아님).
- LLM 필터를 **test 882 에 먼저** 돌렸다(`results/fair_mh_20260928/filter/s3c_filtered_records.jsonl`, 17:22 완료)와 그 답변 128/882(17:28 중단).
  dev 선택 없이 test 에 돌린 것이라 **이 등록에서는 결과로 쓰지 않는다**(파일은 삭제하지 않고 남긴다).

## §2 본 방법 파이프라인 (코드 대응, GATE 0)

| 단계 | 코드 | 상태 |
|---|---|---|
| 1) 셀 문장(s3c, 행·열 전체 경로, 라벨 없음) | `scripts/mh_arms.py: build_tables(v3.3u)` → `retrieval_accuracy.build_corpus(template="s3c", unit="cell")` | 있음 |
| 2) 임베딩·벡터 색인(bge-base-en-v1.5) | `mh_arms.py` `default_encoder` → numpy 행렬 캐시(정규화 코사인, 전수 내적) | 있음 |
| 3) 유사도·상위 K(하이브리드 α .7) | `mh_arms.py` `sc = .7·minmax(dense) + .3·minmax(BM25)`, 문서 안 상위 2,048 → `budget_select` | 있음 |
| 4) LLM 필터 | **본 방법 경로에 없음.** HiTab 전용 `scripts/fair_filter_eval.py`(부록 A). MultiHiertt 용 `scripts/mh_filter.py`(7821b46·f693c7f, 오늘 작성, 8건 배관 점검 통과)는 §6 에서 dev 로 먼저 검증 | 없음 → §6 |
| 5) 리더 전달 | `scripts/answer_accuracy_mh.py --condition retrieved` (검색 문맥 원문 그대로) | 있음 |
| 6) 리더(Qwen3-8B, do_sample=False, seed 42) | 같은 스크립트, `complete_batch` greedy, `torch.manual_seed(42)`, cot 384 | 있음 |

## §3 검색 지표 (GATE 1)

집합 2,885(최종 규칙), 문서 안. 문항 평균(macro). 코드 `results/mh_only_20260928/metrics.py` 상단에 같은 정의를 적는다.
- 셀 단위 arm: G = 정답 셀 좌표 집합, R_k = 상위 k 셀(좌표 중복 제거). Recall@k=|G∩R_k|/|G|, Precision@k=|G∩R_k|/k, MRR=1/(첫 정답 셀 순위), 20위 안에 없으면 0.
- 청크·행·표 arm: 단위 u 가 g 를 포함하면 g 회수. Recall@k=|{g∈G: ∃u∈R_k, g∈u}|/|G|, Precision@k=|{u∈R_k: u∩G≠∅}|/k, MRR=1/(정답 셀을 포함한 첫 단위 순위).
- 다중 정답 셀은 부분 회수를 그대로 점수로. k=1,5,10,20; Precision 은 5,10,20. 기존 "전부 포함" = Recall@20=1.0 비율(열 하나로만).
- (a) 단위 k 표, (b) 토큰 예산 표: 순위 순으로 단위를 더해 누적 토큰(Qwen3-8B 토크나이저, 단위 텍스트만)이 **1,500 을 넘기 직전**까지 넣은 집합의 Recall·Precision(분모 = 넣은 단위 수)·단위 수·셀 수.
  - 1,500 의 근거(실측): 882 문항 문서의 표 전체 입력 평균 1,504.8토큰(`results/mh_arms/cap300_20260924/fulltable.json`, by_layer.ALL.input_tokens_mean),
    문서 표 전체 토큰 중앙값 1,291.5(`results/problem_def_audit_20260925/corpus_token_totals.json`, multihiertt_cap300_documents). 즉 "문서 표 전부가 평균적으로 들어가는 예산".
- 순위 목록은 상위 20 단위까지만 저장돼 있으므로(k ≤ 20) 토큰 예산 표에서 20 단위 안에 1,500 이 안 차는 문항은 20 단위 전부로 자르고 그 수를 적는다.
- 순위 목록이 없는 arm(표 단위, 행·열, TableRAG(Chen) leaf·path 공식 규칙)은 검색만 1회 재실행(리더 없음, 캐시 재사용)해 목록을 얻는다. 소요 시간을 적는다.

## §4 비교군 점검 (GATE 2)

조건 7개(평가 집합 2,885·문서 안 범위·인코더·머리글 최종 규칙·k·리더·프롬프트) 일치표. 청크: 문서당 청크 수 분포(중앙값·20 이하 비율), 청크 색인에 본문 단락 포함 여부, 표가 청크 경계에서 잘리는지. 행·열: 문서당 행+열 수 분포. TableRAG(Yu)·(Chen): 재구현 차이 표(부록 H 형식). 무작위 행 삭제, MT2Net 제외. 코드는 고치지 않는다.

## §5 비교군 재선정 (GATE 3)

기준: 계층 표 포함 문서에서 셀·근거 검색 후 답변, 공개 코드, MultiHiertt 또는 계층 표 벤치마크 평가. 오늘 오후 조사표(`results/fair_mh_20260928/baselines.md`, 1차 출처 대조)를 기준 형식(저자·연도·제목·arXiv·코드 URL·벤치마크·검색 단위)으로 옮긴다. 새 비교군은 사용자가 고른 뒤 구현.

## §6 LLM 필터 (GATE 4, §3~§5 뒤)

- dev 332(`results/reader_format_20260927/mh_dev_pop332.jsonl`), 상위 K∈{20, 50} 후보(dev 검색은 예산 50 으로 1회 재실행 뒤 앞 20·50 을 자른다), 필터 프롬프트 1개 = `mh_filter.FILTER_SYS`(고정, 위 test 실행과 같은 문장).
  - K=20 근거: 운영점(리더에 주는 셀 20, CLAUDE.md §2.2). K=50 근거: 재정렬 진단의 후보 수(`PREREG-2026-09-27-rerank.md`, 상위 50).
- 측정: 필터 후 Recall@k(k ≤ K)·MRR·남은 셀 수, 빈 선택 수, 범위 밖 번호 수.
- 채택 기준(실행 전 고정): 필터 후 Recall@20 이 필터 전 Recall@20 대비 −1pp 이내 **그리고** 남은 셀 수 평균 < 20.
  - −1pp 근거: dev 332 에서 Recall@20≈.94 의 표준오차 ≈ 1.3pp(√(.94·.06/332)), 1pp = 3.3문항 — 표본 잡음보다 작은 손실만 허용.
- 충족 시(사용자 결정 뒤) test 2,885 검색 1회·답변 882 1회. 미충족 시 필터 없는 파이프라인 유지, dev 결과만 보고.

## 결과

### GATE 0 (17:33) — 위 §2 표. 4) 없음, `mh_filter.py` 재사용 가능(dev 검증 뒤).

### GATE 1 (18:28)

- 재실행: 상위 20 단위 목록·단위 토큰 수가 필요해 7 arm 검색을 1회씩 다시 돌렸다(리더 없음, 임베딩 캐시 재사용, 코드 b433526, `results/mh_only_20260928/run_retrieval.sh`).
  소요: s3c 282s, 청크 82s, TableRAG(Yu) 80s, 표 단위 92s, 행·열 1,358s, Chen leaf 352s, path 348s; dev 332(예산 50, 순위 50) 67s. 17:43–18:28.
  기존 기록 대비 `correct`·문맥 불일치: s3c·청크·TableRAG(Yu) 0(오늘 오후 확인), 나머지는 기존 correct 비율이 표 5-5·부록 결과와 같음(표 단위 .7785→.778, 행·열 .4302→.430, leaf .3470→.347, path .3806→.381).
- 지표: `results/mh_only_20260928/metrics.{md,json,csv}`(CSV 에 출처 파일 열). 표는 metrics.md.
- **한계(사전등록대로 처리):** 순위 목록을 20 단위까지만 저장해, 셀 단위 arm 의 토큰 예산 1,500 표는 20 셀(평균 555토큰)에서 잘렸다(2,885건 전부). dev 곡선에서 50 셀 ≈ 1,390토큰이므로
  1,500 예산 비교에는 순위 60 단위 이상이 필요하다 → 추가 검색 재실행(셀 arm 4개, 약 25분) 여부는 사용자 결정.
- §3.5 dev 332: |G| 최대 16(≤20 확인), 중앙값 2. Recall@k: 1 .2516 / 3 .5514 / 5 .6972 / 10 .8378 / 15 .9055 / 20 .9319 / 30 .9736 / 50 .9892.
  단계 증가: 15→20 +2.64pp, 20→30 +4.17pp(셀당 .42pp), 30→50 +1.56pp(셀당 .08pp). 누적 토큰 평균: 20셀 548, 30셀 830, 50셀 1,390.
  표 전체 입력(882건 프롬프트 포함 1,504.8 / 표 텍스트만 1,403.6; 2,885건 1,502.6 / 1,398.3)에 가까운 것은 k≈50 이다. k=20 은 바꾸지 않는다(test 사전등록).
  답변 대 k(dev, k∈{5,10,20,50}) 실행 중 (`dev_k_answer.sh`).
