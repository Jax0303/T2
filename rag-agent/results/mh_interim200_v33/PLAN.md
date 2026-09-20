# interim200 헤더 규칙 v3.3 재평가 계획 (2026-09-14, 생성 전) — 사후 분석·개발 표본

**정정 4 채택 기준 통과 뒤 사용자 지시(2026-09-14)로 실행한다.** 이 계획은 v3.2 로 써 두었다가 실행 전에 규칙 이름·단위 수만 v3.3 으로 바꿨다(이전 경로 `results/mh_interim200_v32/`, git 이력). 결과를 본 뒤 규칙을 고치지 않는다. 전체 1,047 문항 실행은 보류다.
기존 결과 파일은 덮어쓰지 않고, 새 출력은 이 폴더에만 쓴다.

## 표본의 지위

- `results/mh_interim200/ids_200.json`: MultiHiertt **train** 분할 200 문항, id 불변(`ids_sha256` `52696e6b…`).
  분할 이름은 train 그대로다.
- 이 200 문항은 **사후 분석·개발에 사용된 표본**이다. 헤더 규칙 v3 의 결함 사례를 이 표본의 본 방법 대 chunk 답변
  불일치 28 건에서 찾았고, v3.1·v3.2·v3.3 은 그 규칙을 좁힌 것이다(`DATA-USE-2026-09-14.md` §5). 여기서 나온 v2 대비 차이는
  규칙이 이 표본에 맞춰진 몫을 포함하므로 **독립적인 미관측 test 평가로 쓰지 않는다.** 별도 dev split 이 아니다.

## 조건 — 바뀌는 것은 헤더 규칙 하나

| 항목 | v2 (기존, 재실행 없음) | v3.3 (새로) |
|---|---|---|
| 헤더 규칙 | v2 | v3.3 (`PREREG-2026-09-14-header-v3.md` 정정 4) |
| 검색 인자 | `results/mh_arms/mh_{cell,chunk}_hv2.json` 의 `arguments` | 같다. `header_rule`·`tag`·`out_dir` 만 다르다 |
| 라벨 | none | none |
| 리더 | Qwen3-8B 4bit (revision `b968826d…`), `cot`, 384 토큰, greedy, batch 1, seed 42 | 같다 |
| 질의 | 200 id | 같은 200 id (`--same-queries-as ref_contexts_{ours,chunk}_200.jsonl`) |
| 문맥 | 기준 해시 고정 | 규칙 때문에 달라진다 — 해시를 고정하지 않고 바뀐 질의 수를 센다 |

- 짝지음 근거: v3.3 에서 평가 문항 1,047·200 모두 제외 0, gold 셀 집합 변화 0
  (`results/mh_header_v3_3/impact/impact.json` `evaluation_sets`). interim200 중 gold 셀 문장이 바뀐 질의 35, 문서에 바뀐
  문장이 있는 질의 121.
- arm: **본 방법**(cell, v2 에 없던 단위 64,822·2,454 표)과 **chunk**(v2 에 없던 단위 182·84 표). 둘 다 같은 표 파서로
  만들어져 v3.3 에서 색인 문장이 바뀌므로 함께 다시 잰다.
- 넣지 않는 arm: MT2Net(데이터셋 문장, 헤더 규칙과 무관), Huawei(v3.3 에서 바뀐 단위 0). RowCol(3,231 단위)·
  TableRAG allobj_path(37,527 단위)는 바뀌지만 이번 요청 범위 밖이다 — 사용자가 정한다.

## 실행 순서 (승인 뒤)

0. 실제 모델 스모크(출력 `smoke/`, 판정에 쓰지 않음):
   ```
   PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python scripts/mh_arms.py --unit cell --template s3c --header-rule v3.3 \
       --label-rule none --max-docs 20 --tag smoke_cell --out-dir results/mh_interim200_v33/smoke
   PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python scripts/answer_accuracy_mh.py --records results/mh_interim200_v33/smoke/smoke_cell_records.jsonl \
       --scope doc --condition retrieved --header-rule v3.3 --reader "local:Qwen/Qwen3-8B?quantization=4bit" \
       --prompt cot --max-tokens 384 --limit 2 --out results/mh_interim200_v33/smoke/smoke_cell_answer.jsonl
   ```
1. `bash results/mh_interim200_v33/run_v33_200.sh >> results/mh_interim200_v33/run_v33_200.log 2>&1` —
   검색 본 방법 → 검색 chunk → 답변 본 방법 → 답변 chunk → 분석. 검색은 train 모집단 2,908 질의 전체를 돈다.
   중단되면 같은 명령을 다시 돌린다(완료 레그는 건너뛰고 답변은 `--resume`).

## 분석 — `analyze_v33.py` (생성 전 고정)

- 멈춤 검사: 네 답변 파일의 200 id, `records_sha256`·`retrieval_records_sha256` 일치, 생성 설정·모델 revision 동일,
  새 검색 요약의 헤더 규칙 v3.3·라벨 none, 검색 인자 동일(규칙·tag·out_dir 제외), 질의마다 layer·m 동일.
- 묶음 A: 본 방법·chunk × 전체/다중 조회/다중 산술, v3.3 대 v2. 정확 McNemar, Holm(6), 짝지음 부트스트랩 CI
  (20,000 회, 층 안 복원추출, Bonferroni 동시 수준).
- 묶음 B: v3.3 에서 본 방법 대 chunk, 칸 셋, Holm(3). v2 의 같은 칸(전체 9:19)을 보정 없이 병기.
- 기술: 문맥이 바뀐 질의 수, 200 문항 검색 성공 변화, v2 불일치 28 건(본 방법만 9 / chunk 만 19)의 v3.3 결과,
  결함 사례 문서 9 건의 결과.
- 배관 점검(생성 전): v2 파일을 새 규칙 자리에 넣어 돌렸다 — 묶음 A 차이 전부 0:0, v2 본 방법 대 chunk 9:19·
  다중 조회 3:7·다중 산술 4:5 재현. 출력은 저장하지 않았다.
- 해석 한계: query count=200 은 5pt 차이를 가를 검정력이 낮다(`REPORT-2026-09-13.md` §0). 방향과 불확실성만 적는다. 이 표본은
  규칙 개발에 쓰였으므로 개선이 나와도 일반화 근거가 아니다. 검색 요약에 남는 train 모집단 전체 검색 정확도도 규칙을
  train 표 전체로 측정·수정했으므로 같은 제한을 붙인다. 이 결과로 v3.3 채택 여부를 자동으로 정하지 않는다.

## 예상 시간 (v2 실행 기록 기준, RTX 3060 Ti 한 장에서 순서대로)

| 단계 | 근거 | 예상 |
|---|---|---:|
| 검색 본 방법 | `mh_arms/run_hv2_cell.log` 15:29:53 → 15:40:56 (dense 425,870 단위 363 s). 문장이 바뀌면 코퍼스 해시가 바뀌어 임베딩 캐시를 쓰지 못한다 | 약 11 분 |
| 검색 chunk | `mh_arms/run_v2base.log` 16:38:23 → 16:41:37 (dense 108 s) | 약 3 분 |
| 답변 본 방법 200 | `mh_interim200/run_interim.log` 23:29:14 → 00:00:18 (생성 1,837 s) | 약 31 분 |
| 답변 chunk 200 | `mh_interim200/run_arms200.log` 00:48:56 → 01:14:10 (생성 1,496 s) | 약 25 분 |
| 스모크·분석 | 모델 적재 2 회 포함 | 약 5 분 |
| **합계** | | **약 75 분** |

문맥 길이가 v2 와 달라지므로 답변 시간은 달라질 수 있다.
