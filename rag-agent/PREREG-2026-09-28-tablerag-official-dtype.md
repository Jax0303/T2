# 사전등록: TableRAG 셀 검색 재구현을 원본 숫자 열 판정 규칙으로 다시 검색 (2026-09-28)

실행 전에 커밋한다. 검색만 6회, 1회 실행. 답변은 돌리지 않는다. 원고 반영은 결과 보고 뒤 사용자 결정.

## 배경

인용 대조 C25(`reports/citation_check_20260928.md`): 원고의 TableRAG(Chen et al., 2024) 셀 검색 재구현은 "쉼표를 지우고
비어 있지 않은 칸이 모두 숫자면 숫자 열"(`trag._is_numeric_column`, 이하 원고 규칙)로 숫자 열을 정한다. 숫자 열은 최솟값·최댓값
요약 문서 하나로 접혀 개별 값이 색인에 없다. 원본 코드는 pandas 로 정한다 — `utils/utils.py: infer_dtype`(49–68행)가
`pd.to_numeric(errors='ignore')` 뒤 object 로 남은 열에 `pd.to_datetime(errors='raise')` 를 시도하고, 예외는 모두 삼킨다.
`agent/retriever.py: build_cell_corpus`(114–126행)는 object 가 아닌 열을 요약 문서 하나로 접는다
(google-research 커밋 08a8d6736475776f42ffac23b2c13111a28e5795, 2026-09-28 다시 내려받아 확인).

## 등록 전에 본 것 (검색 결과는 보지 않았다)

`results/tablerag_official_20260928/bound.{py,json,log}` — 정답 셀 전부가 어떤 TableRAG 문서의 셀 매핑에 들어 있는
질의 비율(전달 가능 상한, 예산·검색과 무관):

| | 원고 규칙 leaf | 원고 규칙 path | 원본 규칙 leaf | 원본 규칙 path | 모두 범주형 |
|---|---:|---:|---:|---:|---:|
| HiTab test 단일 셀 조회 991 | .3300 | .3300 | .6761 | .3330 | 1.0 |
| MultiHiertt train 2,885 | .9251 | .9251 | .9858 | .9844 | 1.0 |

- 인용 대조 때 보고한 원본 규칙 상한(leaf .6751, path .3290)과 조금 다르다. 그때 계산 코드는 날짜 변환 단계를 넣었는지
  기록이 없다(`analysis/upstream_parity.py` 의 옮긴 `infer_dtype` 에는 날짜 단계가 없다). 이번 구현은 원본 함수를 그대로 옮겼다.
- 원본 규칙에서 날짜 dtype 으로 바뀌는 열: HiTab 3(표·모드 기준, 예: "6 february 1952", "2:16.64"), MultiHiertt 18.
- 코드 변경 뒤에도 원고 규칙 색인 글은 이전 실행과 같다(corpus_text_sha256 세 조건 일치, `bound.json`).

## 조건 (원고 규칙 실행과 다른 것은 숫자 열 판정 하나)

- `--tablerag-dtype official`: 원고 재구현이 펼친 프레임(열 이름 = `_col_name` 의 leaf 또는 path, 칸 = `fmt_value` 문자열)에
  원본 `infer_dtype` 를 그대로 적용한다(`rag_agent/serialization/tablerag_unit.py: official_frame`, pandas 2.3.3).
  이름이 같은 열은 원본처럼 변환 예외로 object 로 남는다.
  - 숫자 dtype 열: 원고 규칙의 숫자 열과 같은 요약 문서·셀 매핑(최솟값·최댓값 행). 요약 문서의 dtype 글자는 원고 재구현처럼
    "float64" 로 고정이다(원본은 실제 dtype, 예: int64 — 기존 차이, 부록 H 대상).
  - 날짜 dtype 열: 원본처럼 요약 문서 하나(`{"column_name", "dtype", "min", "max"}`, pandas 값 그대로), 셀 매핑은 최솟값·최댓값 행.
  - 나머지 열: 원고 규칙과 같은 범주형 문서(칸마다, (열, 값) 중복 제거).
- 나머지 인자는 `results/rerun_20260926/run.sh` 와 같다: bge-base-en-v1.5, hybrid α=0.7, 예산 20셀, template s3c.
  HiTab test 질문의 표 안·538표 한 색인 × leaf·path, MultiHiertt train 문서 안(머리글 최종 규칙) × leaf·path.
- 비교 기준: 원고 규칙 records(`results/rerun_20260926/{hitab,mh}/*tablerag_{leaf,path}*`), 본 방법 s3c records(같은 폴더).

## 예측

1. HiTab leaf: 원본 규칙이 원고 규칙보다 단일 셀 조회 검색 정확도가 높다(두 범위 모두). 전달 가능 상한이 327 → 670건이다.
2. HiTab path: 상한 변화가 3건이라 방향을 예측하지 않는다.
3. MultiHiertt leaf·path: 원본 규칙이 원고 규칙보다 전체 검색 정확도가 높다.
4. 본 방법 s3c 는 여섯 조건 모두의 주 모집단(HiTab 단일 셀 조회, MultiHiertt 전체)에서 원본 규칙 재구현보다 높다.

## 분석 (`results/tablerag_official_20260928/compare.py`)

- 조건마다 검색 정확도: HiTab 질문의 표 안은 세 유형(단일 셀·다중 셀·산술), 538표는 단일 셀 조회, MultiHiertt 는 네 그룹과 전체.
- 원본 규칙 대 원고 규칙, 본 방법 대 원본 규칙: 같은 질의끼리 정확 McNemar(b:c, p). 보고용이며 판정은 위 예측의 방향과 p<.05.
- 결과를 보고 원고 표를 바꿀지(원고 규칙 결과를 대체·병기·부록)는 사용자가 정한다.

## 명령

    cd rag-agent && setsid nohup bash results/tablerag_official_20260928/run.sh > results/tablerag_official_20260928/run.log 2>&1 &

## 코드 변경

- `rag_agent/serialization/tablerag_unit.py: official_frame` 추가.
- `scripts/retrieval_accuracy.py: tablerag_units` 에 `official` 분기(날짜 열 요약 문서 포함), `--tablerag-dtype` 선택지 추가.
  `infer`·`all_object` 경로는 바뀌지 않는다(위 해시 확인).
- `scripts/mh_arms.py --tablerag-dtype` 선택지 추가.
- 점검: HiTab test 538표·MultiHiertt train 11,793표 전부에서 원본 규칙 색인이 예외 없이 만들어진다(HiTab leaf 38,052·path 20,804단위,
  MultiHiertt leaf 524,810·path 530,119단위).

## 결과

(실행 후 기록)
