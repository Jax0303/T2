# 사전등록: MultiHiertt table_description 셀 문장 대 본 방법 셀 문장, 검색 정확도 (2026-09-28)

실행 전에 커밋한다. 1회 실행. **결과는 방향과 관계없이 원고에 싣는다**(원고 반영 시점은 사용자 지시).

배경: MultiHiertt 는 셀마다 머리글을 붙인 문장(`table_description`, 예: "Table 3 shows Policy loans of
December 31, 2015-1 Fair Value Level 3 (in millions) is 11657 .")을 함께 배포한다
(인용 대조 B9/B10, `reports/citation_check_20260928.md`). 본 방법 셀 문장이 이 문장보다 검색에 나은지 잰다.

등록 전에 본 것(검색 결과는 보지 않았다) — `results/table_description_20260928/inspect_desc.{py,json,log}`:
- 본 방법 색인 셀 423,473개 중 같은 좌표에 table_description 이 있는 셀 389,725개(92.03%).
- 채점 질의 2,885건의 정답 셀 7,651개는 **전부** table_description 이 있다(없는 셀 0).
- 없는 셀 33,748개의 값: 대시 21,993 · 숫자 9,326 · 글 2,429.
- 문서 안에서 값을 뺀 문장이 다른 셀과 같은 셀: table_description 27,343/389,725(7.02%), 본 방법 0(최종 규칙이 0을 강제).

## 조건 (본 방법과 다른 것은 셀 문장 하나)

| | 본 방법 | table_description |
|---|---|---|
| 셀 문장 | s3c, 머리글 최종 규칙(v3.3u) | 데이터셋 `table_description[“{표}-{행}-{열}”]` 원문 그대로 |
| 색인 셀 집합 | 423,473 | 같음(`covers` 불변) |
| 인코더 | `BAAI/bge-base-en-v1.5` | 같음 |
| 검색기 | hybrid α=0.7 (dense + BM25) | 같음 — BM25 도 바뀐 문장으로 만든다 |
| 범위·예산 | 질문의 문서 안, 셀 20개 | 같음 |
| 채점 | train 표 근거 2,885건, 정답 셀 전부 포함 | 같음(gold 해석은 문장과 무관) |

- 짝짓기 키: `table_description` 키 = 펼친 격자 좌표 = `mh_arms.cell_id` 의 `::` 뒤(`table_evidence` 와 같은 규약).
- **table_description 이 없는 셀의 처리: 본 방법 s3c 문장을 그대로 둔다.** 이유: 색인 셀 집합(후보)을 본 방법과
  똑같이 두기 위해서다. 빼면 후보가 줄어 table_description 쪽이 쉬워지고, 값만 남기면 그 셀들이 거의 검색되지 않는
  방해 셀이 되어 역시 쉬워진다. 이 셀들은 정답 셀이 아니므로(위 0개) 정답 셀의 문장은 전부 table_description 이다.
  결과에 대체된 셀 수(`n_units_table_description`)를 적는다.
- 문장 앞의 "Table k shows" 도 데이터셋 문장이므로 지우지 않는다.

## 예측

전체 검색 정확도는 본 방법이 table_description 보다 높다(방향만, 크기는 예측하지 않는다).
근거: 문서 안 같은 문장 비율이 본 방법 0% 대 7.02%이고, 머리글 고친 버전에 고유화만 더했을 때 검색에서
고유화 쪽만 맞힌 질의가 더 많았다(29:8, `PREREG-2026-09-24-mh-answer-cap300.md` 결과 절). 그룹별 방향은 예측하지 않는다.

## 분석

- 두 방법의 검색 정확도(전체, 조회1/조회2+/산술1/산술2+).
- 같은 질의끼리 정확 McNemar(b = 본 방법만 맞힘, c = table_description 만 맞힘), 전체와 그룹별 b:c·p.
  그룹별 검정은 보고용이고 다중비교 보정은 하지 않는다 — 판정은 전체 하나.
- 본 방법 수치는 재실행하지 않고 주 결과 records(`results/rerun_20260926/mh/mh_train_s3c_records.jsonl`, 코드 378db41,
  .8634, 이전 실행과 불일치 0)를 쓴다. `compare.py` 가 두 파일의 채점 질의 집합·그룹이 같은지 확인하고 멈춘다.

## 명령

    PYTHONPATH=. setsid nohup .venv/bin/python scripts/mh_arms.py --split train --unit cell --template s3c \
      --header-rule v3.3u --cell-text table_description --cache-dir .cache/tdesc_20260928 \
      --out-dir results/table_description_20260928 --tag mh_train_s3c_tdesc \
      > results/table_description_20260928/mh_train_s3c_tdesc.log 2>&1
    PYTHONPATH=. .venv/bin/python results/table_description_20260928/compare.py

## 코드 변경

`scripts/mh_arms.py --cell-text {ours,table_description}`(기본 `ours` = 기존 경로 그대로). `table_description` 은
`build_corpus` 가 만든 셀 문장 중 같은 좌표에 데이터셋 문장이 있는 것만 바꾼다. `--unit cell` 전용, `--label-mix` 와 같이 못 쓴다.
배관 점검: 문서 30개(`--max-docs 30`, 임시 폴더)로 한 번 돌려 대체 4,319/4,592 확인 — 보고 수치 아님.

## 결과

실행 코드 b4187a0(미커밋 변경 없음), 2026-09-28 05:14 종료. `results/table_description_20260928/`
(`mh_train_s3c_tdesc.{json,log}`, `_records.jsonl`, 비교 `compare.{json,log}`).
대체된 셀 389,725/423,473, 채점 2,885건(제외 23: 머리글 칸 17·표 파싱 실패 4·빈 칸 2 — 본 방법과 같음),
채점 질의 id 해시 본 방법과 같음, 인코더 입력 잘림 0.

| 그룹 | 질의 수 | 본 방법 | table_description | 본 방법만 : table_description 만 | p (정확 McNemar) |
|---|---:|---:|---:|---|---:|
| 전체 | 2,885 | .8634 (2,491) | .8458 (2,440) | 133:82 | .00062 |
| 조회 1 | 212 | .9575 (203) | .9528 (202) | 5:4 | 1.0 |
| 조회 2+ | 367 | .8774 (322) | .8747 (321) | 16:15 | 1.0 |
| 산술 1 | 71 | .9577 (68) | .9577 (68) | 1:1 | 1.0 |
| 산술 2+ | 2,235 | .8492 (1,898) | .8273 (1,849) | 111:62 | .00024 |

**예측(전체에서 본 방법이 높다): 지지됨.** 차이는 산술 2+ 에서 나오고 다른 세 그룹은 차이가 거의 없다.
그룹별 p 는 보정하지 않은 보고용이다. 원인(같은 문장 비율 등)은 검정하지 않았다.
