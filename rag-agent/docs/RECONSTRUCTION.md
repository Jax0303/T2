# RECONSTRUCTION.md — HiTab → 검색 코퍼스 재구성 프로토콜

생성: 2026-06-05 · 빌더: `scripts/build_corpus.py` · seed 무관(결정적, 정렬·기계적)

## 1. 목적과 근거

HiTab(Cheng et al., ACL 2022, arXiv:2108.06712)은 "질문 + 정답 표"가 주어진 QA/NLG
데이터셋이라 그대로는 검색(retrieval) 실험이 불가하다. TARGET(Ji et al., 2025,
arXiv:2505.11545)이 Spider/BIRD/FeTaQA를 검색 태스크로 재구성한 것과 **동일한 표준 절차**를
따른다. 답을 바꾸지 않으며, 표를 코퍼스로 모으고 원본 question→table 정렬을 검색 정답으로 쓰는
재배열일 뿐이다.

## 2. 입력 (원본 필드)

- 표 파일: `data/hitab/data/tables/{hmt,raw}/{table_id}.json`
  - 사용 필드: `title`, `top_root`(열 헤더 트리), `left_root`(행 헤더 트리), `data`(2D 셀 배열)
  - 로드 우선순위: `hmt` → `raw` (T2 `src/data/loader.load_table`와 동일)
- 질문 파일: `data/hitab/data/{train,dev,test}_samples.jsonl`
  - 사용 필드: `id`, `question`, `table_id`(=정답 표), `answer`, `aggregation`

## 3. 코퍼스 구성 규칙

1. **표 id 수집은 기계적**: `tables/{hmt,raw}/*.json` 파일명(stem)의 합집합 = 고유 table_id.
   임의 선택·제외 없음. → 3597개.
2. 각 표 레코드(`corpus/tables.jsonl`):
   `{table_id, title, n_rows, n_cols, top_header_depth, left_header_depth, raw_cells}`
   - `raw_cells` = `{top_root, left_root, data}` 원본 계층 구조 **무손실 보존**.
   - 헤더 깊이 = 헤더 트리 경로 최대 길이(`HeaderTree.get_top_paths/get_left_paths`).
3. **세 가지 직렬화 동시 생성**(추후 통제변수, 효과는 측정 대상이지 가설 아님):
   - `plain_markdown` — 행/열 leaf 헤더 + 마크다운 표 (표당 1 레코드)
   - `json_kv` — `{title, rows:[{__row__, col:val,...}]}` JSON (표당 1 레코드)
   - `header_path` — 헤더 경로별 `"title [a > b > c] cells..."` (표당 N 레코드, leaf-path 단위 청크)
   - 산출: `corpus/serialized/{fmt}.records.jsonl`
   - **재발명 금지**: 직렬화·헤더트리는 기존 T2 구현(`src/serializers/*`, `src/data/header_tree.py`)을
     그대로 재사용 → prebuilt chroma 임베딩과 직렬화 텍스트가 동일하게 보장됨.

## 4. 정답 라벨링 규칙

- `queries.jsonl` 각 레코드:
  `{query_id, question, gold_table_id, answer, aggregation_label, split}`
- `gold_table_id`는 **원본 샘플의 `table_id`를 그대로** 사용(재정렬·추정 없음).
- `aggregation_label`은 원본 `aggregation` 주석 그대로(난이도 층화에 사용).
- 모든 split(train/dev/test)을 포함하되 `split` 필드로 구분. 1차 평가 split = **dev**.

## 5. 제외 건수·사유

- `logs/exclusions.jsonl`에 모든 제외를 사유와 함께 기록.
- 이번 빌드 제외: **표 0건, orphan query 0건** (`logs/exclusions.jsonl` 비어 있음).

## 6. 빌드 결과 (측정값, `corpus/build_summary.json`)

| 항목 | 값 |
|---|---|
| 고유 표 파일 | 3597 |
| 코퍼스에 유지된 표 | 3597 |
| `len(corpus) == 고유 table_id 수` | **True** |
| 직렬화 청크 | plain_markdown 3597 · json_kv 3597 · header_path 86390 |
| 질문 (train/dev/test) | 7417 / 1671 / 1584 (합 10672) |
| orphan question (정답 표 코퍼스 부재) | **0** |

## 7. 재현 절차

```bash
python scripts/build_corpus.py --data-dir data/hitab --out-dir .
```
출력: `corpus/tables.jsonl`, `queries.jsonl`, `corpus/serialized/*.records.jsonl`,
`logs/exclusions.jsonl`, `corpus/build_summary.json`.

## 8. 기존 검색 eval과의 관계 (중요)

기존 `results/retrieval_eval_full.json`은 **dev pool=540**(dev 질문이 가리키는 고유 정답표만)을
검색 풀로 썼다. 본 재구성은 검색 풀을 **전체 3597표**로 확장한다(더 큰 distractor 풀 = 더 현실적·
더 어려움). 따라서 Phase 2 이후 수치는 기존 540-pool 수치와 직접 비교 불가하며, 두 설정을 명시 분리한다.
