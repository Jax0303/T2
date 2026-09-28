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

2026-09-28 05:35~06:07, `run.sh`(시작 커밋 bf5f11a). HiTab 4회의 `git_dirty=True` 는 다른 세션의 분석 파일
(`results/lookup_vs_arith_20260928/posthoc/*.py`) 수정 때문이다. MultiHiertt 2회 결과에 찍힌 커밋 f87c7c2 는 실행 중 다른 세션이
커밋한 HEAD 이며, bf5f11a..f87c7c2 사이 `scripts/`·`rag_agent/` 변경은 0줄이다. 제외 사유별 개수는 원고 규칙 실행과 같다
(HiTab 3건, MultiHiertt 23건). 색인 단위: HiTab leaf 38,052·path 20,804, MultiHiertt leaf 524,810·path 530,119.
집계 `results/tablerag_official_20260928/compare.{json,log}`.

| 조건 | 모집단 | 질의 수 | 원본 규칙 | 원고 규칙 | 원본만 : 원고만 | p | 본 방법만 : 원본만 | p |
|---|---|---:|---:|---:|---|---:|---|---:|
| HiTab 표 안 leaf | 단일 셀 | 991 | .4067 | .2775 | 177:49 | 3.4×10⁻¹⁸ | 560:8 | 5.4×10⁻¹⁵⁴ |
| | 다중 셀 | 38 | .3947 | .1316 | 11:1 | .0063 | 20:0 | 1.9×10⁻⁶ |
| | 산술 | 216 | .2407 | .2037 | 28:20 | .31 | 138:4 | 6.0×10⁻³⁶ |
| HiTab 표 안 path | 단일 셀 | 991 | .2896 | .2916 | 0:2 | .50 | 675:7 | 1.3×10⁻¹⁸⁹ |
| | 다중 셀 | 38 | .1842 | .1842 | 0:0 | 1 | 28:0 | 7.5×10⁻⁹ |
| | 산술 | 216 | .1944 | .1944 | 0:0 | 1 | 154:10 | 2.7×10⁻³⁴ |
| HiTab 538표 leaf | 단일 셀 | 991 | .0616 | .0676 | 10:16 | .33 | 846:1 | 1.8×10⁻²⁵² |
| HiTab 538표 path | 단일 셀 | 991 | .1413 | .1423 | 0:1 | 1 | 769:3 | 6.2×10⁻²²⁵ |
| MultiHiertt leaf | 전체 | 2,885 | .3470 | .3328 | 59:18 | 3.1×10⁻⁶ | 1549:59 | <10⁻³⁰⁰ |
| MultiHiertt path | 전체 | 2,885 | .3806 | .3601 | 74:15 | 1.5×10⁻¹⁰ | 1465:72 | <10⁻³⁰⁰ |

MultiHiertt 그룹별(원본 규칙 대 원고 규칙): leaf 조회1 .5047 대 .4953(4:2), 조회2+ .2698 대 .2534(7:1, p=.070),
산술1 .6761 대 .6761(1:1), 산술2+ .3342 대 .3195(47:14, p=2.7×10⁻⁵); path 조회1 .6274 대 .6132(4:1),
조회2+ .3760 대 .3406(14:1, p=.00098), 산술1 .6479 대 .6479(0:0), 산술2+ .3494 대 .3302(56:13, p=1.7×10⁻⁷).

예측 판정:
1. HiTab leaf 원본 규칙이 더 높다(두 범위): **질문의 표 안은 지지**(.2775 → .4067), **538표는 지지 안 됨**(.0676 → .0616, 10:16, p=.33).
2. HiTab path: 예측 없음. 두 범위 모두 거의 같다(0:2, 0:1).
3. MultiHiertt 원본 규칙이 더 높다: **leaf·path 모두 지지**(.3328 → .3470, .3601 → .3806).
4. 본 방법이 여섯 조건 모두의 주 모집단에서 원본 규칙 재구현보다 높다: **지지**.

원고 반영(원고 규칙 결과를 대체·병기·부록)은 사용자 결정 대기. 부록 H 숫자 열 판정 행은 아직 "확인 중"이다.

## 사후 추가 (2026-09-28, 결과 확인 뒤 사용자 지시): 원고 대체와 답변 재생성

위 "답변은 돌리지 않는다"와 달리, 사용자가 원고 규칙 결과를 공식 규칙 결과로 **대체**하기로 정하고 답변 표본의 문맥이 바뀐
조건의 답변을 다시 생성하라고 지시했다. 결과(검색 정확도)를 본 뒤 정한 사후 추가이며, 예측은 두지 않는다.

- 문맥이 바뀐 문항 수(`results/tablerag_official_20260928/answers/context_changed.{py,json}`, 리더 입력 문자열 목록 비교):
  HiTab 300(표 5-4 표본) leaf 207, path 38 (답변에 쓴 `t_tablerag_*_v2_gold` 문맥은 원고 규칙 재실행 문맥과 300건 모두 같음).
  MultiHiertt 882(부록 F 표본): 처음 규칙 답변 문맥 대비 leaf 726·path 668, 머리글 고친 규칙(v3.3) 답변 문맥 대비 leaf 604·path 437,
  최종 규칙 원고 규칙 문맥 대비 leaf 359·path 305.
- HiTab 300: `scripts/fair_filter_eval.py --no-filter --arms tablerag_{leaf,path} --records <공식 규칙 records>` — 표 5-4 의 기존 행과
  본 방법(s3c) 행을 만든 것과 같은 리더(Qwen2.5-7B-Instruct 4bit, neutral, 64토큰, temperature 0)·채점. 출력
  `results/tablerag_official_20260928/answers/hitab_tablerag_{leaf,path}_official_rows.jsonl`. 표 5-4 의 본 방법 대비 McNemar 와
  8개 비교 Holm 을 다시 계산한다.
- MultiHiertt 882 (사용자 결정 2026-09-28): 이미 돌린 공식 규칙 검색(최종 머리글 규칙 v3.3u)의 문맥으로 답변 2회(leaf, path).
  `scripts/answer_accuracy_mh.py --scope doc --batch-size 128 --header-rule v3.3u --same-queries-as results/mh_arms/cap300_20260924/cell.jsonl
  --condition retrieved --records <공식 규칙 records>` (리더 Qwen3-8B 4bit 비생각, cot, 384토큰 — 표 5-8 의 조건과 같음). 출력
  `results/tablerag_official_20260928/answers/mh_tablerag_{leaf,path}_official.jsonl`. 짝은 최종 규칙 본 방법(`cell_uniq`, 408/882).
  부록 F 의 옛 TableRAG 4행(처음 규칙·v3.3 문맥, 원고 숫자 열 규칙)은 빼고, 새 2행을 최종 규칙 본 방법과 짝지어 싣는다.
- 부록 A (사용자 결정 2026-09-28): 같은 300건에서 필터 실험도 공식 규칙 문맥으로 다시 돈다.
  `scripts/fair_filter_eval.py --arms tablerag_{leaf,path} --records <공식 규칙 records>` (필터 포함; 이를 위해 `--records` 를 필터와
  함께 쓸 수 있게 조건 한 줄을 풀었다, rowexp 는 그대로). 출력 `results/tablerag_official_20260928/answers/hitab_tablerag_{leaf,path}_official_filter_rows.jsonl`.
  이 실행의 무필터 답은 위 무필터 실행과 같아야 한다(같은 리더·temperature 0) — 다르면 보고한다.
