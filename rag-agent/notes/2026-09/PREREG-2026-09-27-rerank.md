# 사전등록 — 교차 인코더 재정렬 진단 실험 (2026-09-27)

상태: **실행 전 커밋.** 커밋 전에 실행한 것은 §5 의 기준선 재현 확인(재정렬기 없음)뿐이다.
실행 중 바뀌는 것은 전부 §12 이탈 기록에 적는다.

## 1. 목적

- GATE 1 진단(`results/bottleneck_20260927/diag/`)에서 확인된 실패 문항 중 정답 셀이 21–50위인 문항을,
  표준 교차 인코더 재정렬이 20위 안으로 올리는지 확인한다.
- **진단 실험이다. 본 방법의 기여가 아니다.** 결과와 관계없이 본 방법(s3c 셀 문장 + hybrid α=.7 + 상위 20 셀)의 정의는 바뀌지 않는다.
- `CLAUDE.md` §6 에는 "크로스인코더 재정렬"이 닫힌 노선으로 적혀 있다. 이번 실행은 사용자 지시(2026-09-27)로 하는 진단이다.
  - 2026-09-17 결과(`results/cross_encoder_rerank_20260917/summary.json`)는 템플릿 sleaf, 538표 한 색인, 1위 셀 지표였다.
    문항별 순서가 저장되지 않아 20셀 지표로 다시 계산할 수 없다.

## 2. 재정렬기

- `BAAI/bge-reranker-v2-m3`, 로컬 캐시 snapshot `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`. 학습 없음.
- `sentence_transformers` 5.6.0 `CrossEncoder(max_length=512)`, fp32, cuda, batch_size 50.
- 입력 쌍 = (질문 원문, s3c 셀 문장). 질문에 접두어를 붙이지 않는다.
- 절단 건수 = 재정렬기 토크나이저로 `truncation=False` 토큰 수가 512 를 넘는 쌍의 수. 문항별로 기록한다.
- 후보 1개다. `bge-reranker-base` 는 쓰지 않는다. 선택 절차가 없다.

## 3. 후보 집합과 문맥

- 재정렬 전 순위 = s3c + `BAAI/bge-base-en-v1.5` hybrid, α=.7. 점수 식은 기준선 스크립트 그대로다.
  - HiTab: `scripts/retrieval_accuracy.py`, 질문의 표 안 단위만 min-max.
  - MultiHiertt: `scripts/mh_arms.py`, 코퍼스 전체 min-max 뒤 문서로 마스크, 문서 안 상위 2048.
- 후보 = 재정렬 전 상위 50. 표·문서의 셀이 50 개보다 적으면 그 전부다.
- 후보 50 을 재정렬기 점수 내림차순으로 다시 세운다(동점은 재정렬 전 순서). 그 앞에서 20 셀이 문맥이다
  (`budget_select`, 셀 단위라 20 단위 = 20 셀).

## 4. dev 집합 (2026-09-26 α 선택과 같은 집합·같은 제외 규칙)

| 범위 | 문항 | 기준선 결과(재정렬 전) | 기준선 맞힘 |
|---|---|---|---|
| HiTab dev, 질문의 표 안, 단일 셀 조회 | 1,062 | `results/dev_alpha_20260926/hitab_dev/hitab_dev_gold_prefix`(s3c) | 1,005 |
| MultiHiertt validation, 질문의 문서 안, 머리글 v3.3u, 라벨 없음 | 911 | `results/dev_alpha_20260926/mh_dev/mh_dev_a1.0` | 774 |

- HiTab dev 문항 = `type_accuracy` 의 `single_cell`. 제외 2건(`unreadable_formula`, `no_gold_annotation`)은 단일 셀 밖이다.
- MultiHiertt dev 문항 = `--keep-hybrid`, 표 근거가 있는 질의 929 에서 제외 18(`gold_in_header` 13, `gold_table_unparsed` 5).
  위에서 text-only 115 는 들어오지 않는다.
- MultiHiertt 기준선 `a1.0` 은 라벨 벡터 섞기 α=1.0 으로 돌았다. 이번 실행은 섞기를 하지 않는다.
  재계산 상위 20 이 `a1.0` records 와 911 문항 전부 같음을 §5 에서 확인했다.
- 사용 캐시(재정렬 전 순위 계산용, 재인코딩 없음). 질문 벡터만 실행 중에 인코딩한다:

| 캐시 | sha256 |
|---|---|
| `.cache/dev_alpha_20260926/BAAI_bge-base-en-v1.5_67315_4b07d547cfb5fc069c1aa2c7.npy` (HiTab dev 540표, 코퍼스 `a5569a11…`) | `a9175b353594d20eb068b24dd377212ea0905d4be13c9b9325d80b004f4958de` |
| `.cache/dev_alpha_20260926/cell_135894_81f450bf084fe383_0.npy` (MH validation, 코퍼스 `81f450bf…`) | `2dea49b11e2861e33ecc2e120b411e51f162e6cae5015ec436fb69790e822ee2` |
| `.cache/dev_alpha_20260926/cell_135894_81f450bf084fe383_50000.npy` | `9121e59910cb1df92f378a3463ec6a1bb3b77cb6ff28bc1443582bcba94f99e1` |
| `.cache/dev_alpha_20260926/cell_135894_81f450bf084fe383_100000.npy` | `91c70e6be2d2e0fd3b068a0e5511cf98afe8a0e8a6f9c56c683ac2baf765b133` |

## 5. 커밋 전 확인 (재정렬기 없음) — `results/bottleneck_20260927/dev/check.json`

- 재계산 상위 20 대 기준선 records: HiTab dev 불일치 0/1,062, MultiHiertt dev 불일치 0/911.
- 재정렬 전 맞힘: 1,005/1,062, 774/911. 기준선 파일 값과 같다.
- 쌍 수: HiTab dev 47,270 + MultiHiertt dev 45,450 = **92,720**.
  지시서의 98,650(= 1,973 × 50)은 상한이다. 셀이 50 개보다 적은 표·문서가 있어서 실제 값이 작다.
- 절단 쌍: 0 / 92,720 (max_length 512).

## 6. 지표와 문항 단위 순위

- 지표 = 주 결과표와 같은 정의다. 20 셀 문맥이 정답 셀을 **전부** 담으면 맞힘.
- 문항 순위 `pre_rank`/`post_rank` = 정답 셀들의 순위 중 가장 큰 값. 셀 단위라 맞힘 ⇔ 이 값 ≤ 20.
  - 재정렬 전: 전체 순위. MultiHiertt 는 문서 안 상위 2048, 그 밖이면 없음.
  - 재정렬 후: 후보 50 안의 순위. 정답 셀이 후보 밖에 있으면 없음.
- 이동 건수:
  - 전 ≤20 → 후 >20
  - 전 21–50 → 후 ≤20 (전 21–50 문항 수도 함께 적는다)
  - 전 >50 (없음 포함) — 재정렬로 회복할 수 없다. 건수만 적는다.
- `path_overlap` = |질문 단어 ∩ 정답 셀 경로 단어| / |질문 단어|.
  - 단어 = `encoders._tokenize`, 불용어 = sklearn `ENGLISH_STOP_WORDS`. 경로 = 행 경로 + 열 경로이고, 표 제목과 값은 넣지 않는다.
    HiTab 은 GATE 1(`results/components_20260925/hitab_path_overlap.py`)과 같은 정의다.
  - **MultiHiertt 정의는 GATE 1 에 없던 것이고, 이번 실행 전에 등록한다.** 정답 셀 전부의 경로 단어 합집합을 쓴다.
  - `overlap_zero` = 0 인 문항의 맞힘 수·b:c·이동 건수를 따로 적는다. 회복할 수 없는 하한으로 별도 집계한다.
- MultiHiertt 는 조회 셀1 / 조회 셀2+ / 산술 셀1 / 산술 셀2+ 별로도 적는다(records 의 `layer`).

## 7. 통계와 채택 기준 (결과 전 확정, 바꾸지 않음)

- 재정렬 전 대 후, 같은 문항끼리 정확 McNemar(이항, 양측). b = 전만 맞힘, c = 후만 맞힘.
- **채택 = 두 조건을 모두 만족.**
  - (i) 두 dev 모두 후 맞힘 ≥ 전 맞힘.
  - (ii) 적어도 한 dev 에서 c > b 이고 p < .05.
- 미달이면 test 에 적용하지 않고 dev 결과만 보고한다. 종료한다.
- 충족 여부만 보고하고 대기한다. test 적용은 별도 지시 후에 한다.

## 8. 처리량과 시간 상한

- 모델 적재 뒤 HiTab dev 첫 100쌍을 한 번 `predict` 해서 초당 쌍 수를 잰다(예열 없음).
- 예상 시간을 두 값으로 보고한다: 98,650쌍 기준, 실제 92,720쌍 기준.
- 예상이 3시간을 넘으면 멈추고 보고한다. 3시간 이하면 dev 전체를 실행한다.

## 9. test 적용 (채택 + 별도 지시 후에만)

- 범위: HiTab test 질문의 표 안 991, HiTab test 538표 한 색인 991, MultiHiertt train 질문의 문서 안 2,885.
- 재정렬 전 s3c 대비 McNemar. **Holm 보정 묶음 = 이 세 범위(3 검정).**
- 결과표에 s3c 외 비교군을 함께 놓으면 같은 재정렬을 그 비교군에도 적용한다. 적용하지 않은 표는 "s3c 단독"으로 표기한다.
- GATE 1 버킷(§10)별로 회복된 문항 수(틀림→맞힘)와 반대(맞힘→틀림) 건수를 적는다.
- test 재정렬 전 순위 캐시. sha256 이 재실행 결과 코퍼스와 일치하는 파일이다. 지시서 경로 `recheck_20260926/cell_425870_*` 는 쓰지 않는다:
  - HiTab `.cache/rerun_20260926/BAAI_bge-base-en-v1.5_67664_ea0c705bc7d8831aa9f7db1a.npy` `eaa8d45f5c6e797cf4c433c98bb0ad3136b79cc6b29b7e691dad15ecb88e1f90`
  - MH `.cache/rerun_20260926_mh/cell_423473_ccef1b9526bb259c_<시작>.npy`:
    `0` 699b410e0f141eae056afbaaa52bdcaa02949bf17b4a70ed2d38194f627e6bfe,
    `50000` 227efcd00af58e15217d5966cd0f98a2784883d985256e57ff46a064507160df,
    `100000` 38263611cb5681340529b779ea7151d9792fc4e32231d5a467f795e5bfd60507,
    `150000` a0f17ae2ccaa50b1c8f35d6797713df8e7ede7fcd7c491cb0953dbe09f21f732,
    `200000` 8bb322c9560f29544a49e0de447ed9dcb79224fc07899d5eab3334549af9aacc,
    `250000` 92429c7bfa81a151b13138c94b4cbc10940e89a70131c019a889905db2030538,
    `300000` c6d13f8dbd5169799bc165e6fade3ba435cd39cbc53411d8a3eecbcb98ba81b4,
    `350000` b1418664d7bfbacd494cebf2281839103af2716f02a2443fcdd156839504e347,
    `400000` 593062b84adeb40c854ef06165b009d3aaafc5933902838210ebd3a0e9fd6df6

## 10. GATE 1 버킷 정의 (`results/bottleneck_20260927/diag/summary.json`, test, s3c)

HiTab:
- A = 538표 한 색인에서 정답 표의 셀이 상위 20 에 하나도 없음.
- B = 정답 셀 순위 21–50.
- C = 51 이상.
- D = 20위 이내인데 실패(0건).

사용자 지정 대상 버킷:
- HiTab 표 안: B (20건).
- HiTab 538표: B (25) + A 중 정답 셀 ≤50 (16).
- MultiHiertt 문서 안: F·G·H 중 누락 셀 전부 ≤50 (F 220, G 31, H 45).

MultiHiertt `bucket_definitions` (그대로 옮김):
- E: 상위 20 셀 중 정답 표(들)의 셀이 0개
- H: 정답 표 셀은 있으나 정답 셀은 0개 찾음
- F: 정답 셀 일부 찾음, 누락 셀의 표가 모두 문맥에 다른 셀로 들어와 있음
- G: 정답 셀 일부 찾음, 누락 셀 중 표가 문맥에 없는 것이 있음
- missing rank: 문서 안 hybrid 순위(상위 2048 까지), 그 밖은 >50

판정 원값(HiTab 538표 A 40건, `judge40_rater_A.csv`): (가) 예 20 / 아니오 20, (다) 예 1 / 아니오 29 / 판단불가 10.
이 값으로만 기록한다. 17/3/20 은 파일 근거가 없다.

## 11. 저장

- 스크립트: `results/bottleneck_20260927/rerank_dev.py` (`--check`, `--throughput 100`, 모드 없음 = dev 전체).
- 문항별 `results/bottleneck_20260927/dev/{hitab_dev,mh_dev}.jsonl`.
  - 한 줄 = 재정렬 전 상위 50 `[cell_id, hybrid 점수]`, 재정렬 후 상위 50 `[cell_id, 재정렬기 점수]` 전체 순서.
  - 그 밖에 정답 cell_id, `pre_rank`, `post_rank`, 전후 맞힘, `path_overlap`, 쌍 수, 절단 수.
  - 집계값만 저장하지 않는다.
- 집계 `dev/summary.json`, 로그 `dev/run.log`, 처리량 `dev/throughput.json`, 재현 확인 `dev/check.json`.
- 100MB 넘는 파일은 커밋하지 않는다. 대신 sha256 과 재생성 명령을 README 에 적는다.

## 11a. 보고 범위 (2026-09-27 사용자 결정, test 실행 전)

- 재정렬은 본 방법(s3c)에 포함하지 않는다. 결과는 병목 진단 절에 s3c 단독 전/후 비교로만 보고한다. 비교군에는 적용하지 않는다.
  (§9 의 "비교군에도 적용" 규칙은 이 결정으로 쓰지 않는다.)
- test 는 1회 적용한다. 설정은 dev 와 같다. 재정렬 전 맞힘이 955 / 906 / 2,491 과 다르면 멈춘다.
- test 실행 로그에는 재정렬 전 기준선 확인과 처리 문항 수만 출력한다. 중간 정확도는 출력하지 않는다.
- 보고: 범위별 전/후 맞힘, b:c, McNemar 양측 p, 3범위 Holm p; 이동 건수; GATE 1 버킷별 회복·새로 틀림; overlap_zero 별도;
  MultiHiertt 4그룹별(탐색적 결과로 표기); 절단 건수; 실행 시간.
- 스크립트 `results/bottleneck_20260927/rerank_test.py`, 출력 `results/bottleneck_20260927/test/`.

## 12. 이탈 기록

- 2026-09-27 03:26 dev 전체 실행 1회차가 HiTab dev 600/1,062 문항 이후 `torch.AcceleratorError: CUDA error: unknown error`
  (재정렬기 predict 중)로 멈춤. 결과 파일(jsonl·summary)은 쓰이지 않았다. 로그는 `dev/run_crash1.{out,log}` 로 보존.
  `CLAUDE.md` §7(GPU 를 Windows 쪽과 공유, 같은 설정으로 이어 돌린다)대로 같은 스크립트·같은 설정으로 처음부터 다시 실행.
  1회차 로그(`run_crash1.{out,log}`)에 출력된 값은 재정렬 **전** 기준선 확인 두 줄
  (`[hitab dev] 1062 items, pre correct 1005, top20 check ok`, `[mh dev] 911 items, pre correct 774, top20 check ok`)과
  처리 문항 수(`hitab_dev 200/1062`, `400/1062`, `600/1062`)뿐이다. 재정렬 **후** 정확도·맞힘 수는 출력되지 않았다.
- 2026-09-27 03:27 2회차 실행이 인코더 적재 직후(진행 0문항) Claude Code 세션 종료와 함께 멈춤. 결과 파일 없음.
  로그 `dev/run_crash2.{out,log}` 보존. 같은 스크립트·같은 설정으로 3회차 실행.
- 2026-09-27 03:39 3회차 실행 완료(636.6초). dev 결과 = `dev/summary.json`, 문항별 `dev/{hitab_dev,mh_dev}.jsonl`. 채택 기준 충족(§7).
- 2026-09-27 03:44–04:12 test 1회 적용 완료(1,667.8초, 중단 없음). 재정렬 전 955 / 906 / 2,491 일치, 재계산 상위 20 불일치 0.
  결과 = `test/summary.json`, 문항별 `test/{hitab_intable,hitab_538,mh_indoc}.jsonl`. 실행 출력 `test/run.out`
  (실행 중에는 `results/bottleneck_20260927/test_run.out` 에 쓰고 끝난 뒤 옮김).
- 2026-09-27 푸시 전 커밋 다시 쓰기(사용자 지시). `bd5add5` 에 든 114.06MB 파일
  (`results/mh_arms/mh_train_tablerag_leaf_hv3.3_none_doc_records.jsonl`)이 GitHub 100MB 한도를 넘어 기록에서 뺐다
  (`git filter-branch --index-filter`, 작성·커밋 시각과 메시지 유지). sha256·백업 경로·재생성 명령은 `results/mh_arms/README-large-files.md`.
  옛 해시 → 새 해시: `410c3dc` → `410c3dc`(바뀌지 않음), `bd5add5` → `6998304`, `378db41` → `726dc13`,
  `5f7ac77` → `ff75bd4`, `4a46e20` → `4db5ec1`, `7be1579` → `3b8b31a`.
  이 문서 앞부분·대화 보고에 나온 옛 해시는 이 대응으로 읽는다.
