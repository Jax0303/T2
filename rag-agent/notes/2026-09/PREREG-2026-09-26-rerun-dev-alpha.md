# 사전등록: 주 결과표 재실행 + dev 에서 라벨 섞기 비율 선택 (2026-09-26)

실행 전에 커밋한다. 리더 생성은 없다. 실행 중에는 코드를 고치지 않는다.

## 사용자 결정 (2026-09-26)

- **본 방법 = s3c 로 확정** (HiTab). 이전 HiTab 본 방법 sleaf 는 비교군으로 싣는다.
- HiTab dev·MultiHiertt dev 를 라벨 섞기 비율 선택에 쓴다. `CLAUDE.md` §3.3(HiTab dev 는 쓰지 않는다)의 예외다.

## 항목 1 — 주 결과표 재실행

- 새 임베딩 캐시(`.cache/rerun_20260926`, `.cache/rerun_20260926_mh`)로 전부 다시 인코딩한다.
- **HiTab test**: bge-base-en-v1.5, hybrid 가중 .7, 예산 20셀. 범위는 두 가지다: 질문이 속한 표 안(`--corpus gold`), 표 538개 한 색인(`--corpus split`).
  - 채점 대상: 단일 셀 조회 991(mode all, m=1, aggregation none).
  - 비교군 10개: s3c, sleaf, 표 단위, 행 단위(값만), 고정 청크 1000자, trag_hetero, TableRAG-leaf, TableRAG-path, RowCol(값만), RandRow(값만).
  - RandRow 는 538표 범위를 코드가 거부한다(질문의 표 안에서만 정의).
- **MultiHiertt train**: 머리글 최종 규칙(v3.3u), 문서 안 검색. 같은 비교군 10개, 가중·예산은 HiTab 과 같다. 채점 대상은 채점 문항 전부다.
- 인자는 기존 결과 JSON 의 `arguments` 와 같다. 다른 것은 태그·출력·캐시 경로뿐이다.
  - 입력이 512 토큰을 넘을 수 있는 단위(표·행·청크·trag_hetero, MH rowcol)는 `--embed-overflow truncate`. 넘친 수는 결과 JSON `embedding_input_audit` 에 남는다.
- 집계(`results/rerun_20260926/compare.py`):
  - 정확도.
  - 기존 records 가 있는 칸은 불일치 문항 수.
  - s3c 대비 McNemar(정확 이항, b = s3c만 맞힘, c = 비교군만 맞힘) + Holm. Holm 의 가족은 한 데이터셋·한 범위 안의 비교군 전부다.

## 항목 2 — 라벨 섞기 비율 α 를 dev 에서 선택

α 는 hybrid 가중(.7 고정)이 아니다. 셀 벡터 = normalize(α·셀 문장 벡터 + (1−α)·표 라벨 벡터)의 α 다(`--label-mix`, 2650cd8).

- 후보 11개, 이 순서: α = 1.0, 0.9, …, 0.1, 그리고 **접두어**(라벨을 문장에 넣는 방식).
  - HiTab: α 후보는 s2 문장(`--template s2 --label-mix α`), 접두어는 s3c.
  - MH: α 후보는 라벨 없는 셀 문장(`--label-rule none --label-mix α`, 라벨 벡터 = L1), 접두어는 `--label-rule L1`.
- dev 모집단:
  - HiTab dev 1,671 문항 — 선택 지표는 test 주 모집단과 같은 정의인 **단일 셀 조회** 정확도. 범위는 질문의 표 안 / dev 표 540개 한 색인, 범위마다 따로 고른다.
  - MultiHiertt dev(validation) — 표 근거가 있는 질의 전부(`--keep-hybrid`, 929 중 채점 가능 문항). 머리글 v3.3u, 문서 안.
  - MH dev 임베딩은 새로 인코딩해 `.cache/dev_alpha_20260926` 에 저장한다.
- 선택(`results/dev_alpha_20260926/select_alpha.py`, dev 결과만 읽는다):
  - 데이터셋별 최고 = 맞힌 수 최대.
  - 공통 최고 = (HiTab dev 표 안 정확도 + MH dev 문서 안 정확도) / 2 최대.
  - 동률이면 위 후보 순서에서 앞선 것.
- 적용: 고른 후보를 HiTab test(해당 범위; 공통 최고는 두 범위 모두)와 MH train(문서 안)에 **한 번만** 돌린다(`results/dev_alpha_20260926/apply/`).
- 방향 예측은 하지 않는다.

## 명령

    setsid nohup bash results/rerun_20260926/run.sh > results/rerun_20260926/run.log 2>&1 &
    .venv/bin/python results/rerun_20260926/compare.py

## 코드 변경 (이 커밋)

- `scripts/mh_arms.py`: `--unit table`(HiTab 과 같은 `build_corpus` 표 단위), `--keep-hybrid`(accuracy 모드에서 text_evidence 가 함께 있는 질의 유지). 기본 동작은 바뀌지 않는다.

## 결과

(실행 후 기록)
