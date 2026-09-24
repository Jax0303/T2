# 사전등록 — MultiHiertt 답변 정확도 재실행(그룹당 상한 300, 채택 리더 설정) + "표 전체" 비교군 (2026-09-24)

## 왜

- 2026-09-21 MultiHiertt 답변 실행은 채택 설정(cot/384)이 아닌 neutral/64 로 돌아 무효다(`THESIS-INTERIM-2026-09-23.md` §6.4).
- 설계는 2026-09-23 사용자 결정: 조회/산술 × 셀 1개/2개 이상 4그룹, 그룹당 상한 300, 전체 정확도는 모집단 비율 가중.
- "표 전체를 리더에 넣기" 비교군은 2026-09-23 사용자 결정 8번. 검색 없이 질문의 표(HiTab) / 문서의 표 전부(MultiHiertt)를 준다.

## 표본 — 결과를 보기 전에 고정

- 모집단: MultiHiertt train, 표 근거만 필요한 질문 중 채점 2,871건(검색 레코드 `mh_train_*_hv1_none_doc_records.jsonl`).
- 추출: 그룹마다 질의 id 를 정렬해 `random.Random(20260913).sample(ids, 300)` — `answer_accuracy_mh.py --stratum-cap 300 --sample-seed 20260913`
  그대로. seed 20260913 은 2026-09-13 부터 쓰던 스크립트 기본값이다.
- 결과: 조회 셀 1개 211(전부) · 조회 셀 2개 이상 300/365 · 산술 셀 1개 71(전부) · 산술 셀 2개 이상 300/2,224 = **882건**.
- id 목록 `results/mh_arms/sample_cap300_seed20260913.json`, 정렬 id 의 sha256 `a424e4863a3ff705644b31cd8ac6ed9eb48e59a483c50d32ff0f173855bc182a`.
  7개 방법의 검색 레코드에서 각각 뽑아 같은 882건이 나오는 것을 확인했다.

## 조건

| 항목 | 값 |
|---|---|
| 방법 | 본 방법(cell, 헤더 v1, 라벨 없음), chunk, rowcol, trag_hetero, tablerag_path, tablerag_leaf, randrow — 질문의 문서 안 검색, 셀 예산 20, 기존 검색 레코드 그대로 |
| 표 전체 | 질문 문서의 표 전부를 청킹 arm 과 같은 markdown(`chunks.markdown_source`)으로, `--condition fulltable`. 입력 중앙 1,321·최대 5,190토큰(`results/table_token_lengths.json`) |
| 리더 | `Qwen/Qwen3-8B` 4bit, 비생각, `cot` 프롬프트, greedy, 최대 384토큰(`PREREG-2026-09-13-reader-qwen3.md`, 재확인 `PREREG-2026-09-23-reader-thinking-pilot.md`) |
| 채점 | 주: 공식 EM(`mh_exact_match`, 공식 정답 `yilunzhao/MultiHiertt@f18473d`). 보조 DocMath EM 은 판정에 쓰지 않는다 |
| 생성 방식 | continuous batching(`LocalQwenLLM.complete_batch` = transformers `generate_batch`), 한 번에 128건 넘김(`--batch-size 128`) |

### 생성 방식을 바꾼 이유와 확인

- 배치 1 은 1건당 평균 9.4초(출력 평균 145토큰 × 토큰당 약 0.06초), 8개 조건 × 882건이면 약 18시간.
- 고정 묶음(왼쪽 채움)은 8GB 에서 입력 1,000토큰 × 8건, 1,500 × 4, 3,000 × 2 가 메모리 초과였다(모델 적재 후 여유 0.71GiB).
  continuous batching 은 페이지 단위 KV 캐시라 메모리에 맞춰 동시 처리 수를 정한다.
- **확인(본 실행 전):** 저장된 배치 1 실행(`..._fulln_cot384.jsonl`, 본 방법, 같은 문맥) 앞 120건을 `--batch-size 120` 으로 다시 생성해
  정오를 짝지어 비교한다. **기준 = `CLAUDE.md` §7 짝지음 규칙: 정오 불일치 ≤ 5 이고 정확 McNemar p ≥ .05 이면 채택, 아니면 배치 1 로 돈다.**
  같은 경로의 재현성 기준선으로 배치 1 재실행 20건도 함께 적는다.
- 이 등록 전에 이미 본 것(공개): 같은 120건 중 앞 48건 예비 시험 — 1건당 3.12초, 출력 문자열 일치 27/48, 추출 답 일치 36/48,
  정오 불일치 1:2. 기준 판정에는 120건 결과만 쓴다.
- 출력은 배치 1 과 바이트 단위로 같지 않다. 모든 조건이 같은 방식으로 돌므로 조건 간 비교는 같은 조건에 놓인다.
  2026-09-23 에 배치 1 로 저장된 본 방법 683행은 생성 방식이 달라 **재사용하지 않는다.**

## HiTab 표 전체

- HiTab 7개 방법 표(`THESIS-INTERIM` §6.2)와 같은 300건(`results/ksweep_population_300.json`), 같은 리더(`Qwen2.5-7B-Instruct` 4bit),
  `neutral` 프롬프트, 64토큰, 배치 1. 문맥만 표 전체 markdown(이름 = 색인과 같은 라벨: 섹션 제목 + 페이지 제목).
  `scripts/hitab_fulltable_answer.py`. 입력 중앙 627·최대 5,190토큰.

## 보고

- 그룹별 EM 과 query count, 가중 전체 EM(가중치 211/365/71/2,224 ÷ 2,871)과 그룹 층화 부트스트랩 95% CI(10,000회, seed 0).
- 본 방법 대 각 조건: 그룹별 정확 McNemar, 가중 전체 차이의 층화 부트스트랩 95% CI.
- 검색 성공/실패 부분집합별 EM(스크립트 요약 그대로).
- HiTab 표 전체: 300건 정확도, 본 방법(무필터 `correct_base`) 대비 정확 McNemar.

## 예측 (결과를 보기 전에)

1. MultiHiertt 가중 전체 EM: 본 방법이 검색 7개 방법 중 1위.
2. 본 방법 대 tablerag_path·tablerag_leaf·randrow: 가중 전체 차이의 95% CI 가 0 보다 크다.
3. 본 방법 대 chunk·rowcol·trag_hetero: 가중 전체 차이가 양수(유의성은 예측하지 않는다).
4. 표 전체 대 본 방법: 방향을 예측하지 않는다. HiTab 은 표 전체 입력(중앙 627)이 본 방법 20셀 문맥(중앙 1,264)보다 짧아 표 전체가 이길 수 있다.

## 실행 순서

배치 확인 → (채택 시) 본 방법 → 표 전체 → chunk → rowcol → trag_hetero → tablerag_path → tablerag_leaf → randrow → HiTab 표 전체.
방법마다 새 `--out`(`results/mh_arms/cap300_20260924/`).

## 알려진 한계

- MultiHiertt 정답 근거(`table_evidence`)는 계산 피연산자만 담고, 조건 판정 셀("X 가 가장 큰 해의 Y 증가율"의 X 행)을 빼는 문항이 있다.
  그 문항에서 검색 정확도는 과대다. 답변 정확도는 리더가 조건 셀을 실제로 받았는지까지 반영한다. 규모 추정은 별도 기록.
- 표 전체 조건의 `retrieval_correct` 는 정의상 1 이다.

---

## 배치 확인 결과 (2026-09-24, 본 실행 전)

- 배치 1 재실행 20건: 출력 문자열 20/20 이 저장된 실행과 같다 — 배치 1 경로는 재현된다.
- continuous batching 120건(`results/mh_arms/batchcheck_20260924/cb_120.*`): 출력 문자열 일치 72/120, 추출 답 일치 98/120,
  EM 35/120 대 35/120, **정오 불일치 1:1, 정확 McNemar p=1.0** → 기준(불일치 ≤ 5, p ≥ .05) 통과, **채택**.
- 속도: 1건당 3.08초(배치 1 은 10.1초).

## 생성 설정 수정과 재확인 (2026-09-24 16:0x~16:5x, 결과 보기 전)

- 첫 본 실행(16:02)은 `max_memory_percent=0.9` 에서 첫 128건이 13분 넘게 끝나지 않아 중단했다(저장 0행, 파일 삭제).
  원인: WSL 에서 캐시가 GPU 메모리를 넘겨 시스템 메모리로 넘친 것으로 본다(같은 증상이 메모리 측정 때도 있었다).
- 같은 882건 표본의 앞 48건으로 설정만 바꿔 속도를 쟀다(답은 보지 않음): 0.6·블록 256 3.58초/건, 0.6·블록 64 3.16초/건,
  **0.75·블록 64 2.79초/건 → 채택.** 캐시 6,016토큰으로 가장 긴 입력(표 전체 5,190) + 384 가 들어간다.
  넘침을 막으려고 `torch.cuda.set_per_process_memory_fraction(0.95)` 를 건다(넘치면 OOM 으로 멈춘다).
- 같은 기준으로 재확인(`batchcheck_20260924/cb2_120.*`): 출력 문자열 일치 65/120, 추출 답 일치 100/120,
  EM 35 대 36, **정오 불일치 1:2, p=1.0 → 통과.** 1건당 2.48초.
