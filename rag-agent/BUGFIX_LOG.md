# BUGFIX_LOG

## 2026-09-08 — 검색 파이프라인이 표와 질의를 조용히 버리고 있었다 (test 4건)

> 아래에서 결함 위치로 지목한 `scripts/corpus_dump_vs_cell.py`, `analysis/cell_rank_dump.py`
> 는 같은 날 커밋에서 **삭제됐다**. 이 항목은 무엇이 왜 틀렸는지의 기록이고, 파일은
> git 히스토리에 있다. 고쳐서 남긴 것은 `rag_agent/` 쪽 두 건(2)(3)이다.

연구실에서 "표 일부가 아예 색인되지 않는다"고 잡은 것과 같은 계열의 결함이 이쪽에도
있었고, 실제로는 그보다 넓었다. 네 건 전부 **코드를 직접 읽고 test 분할 전수로
계측해서** 확인했으며, 근거 수치는 아래 각 항의 괄호 안에 있다.

### (1) 표 색인 누락 — test 538표 중 414표만 색인 (23% 누락)

**대상**: `scripts/corpus_dump_vs_cell.py: build_corpus()`

**증상**: 코퍼스가 그 분할이 참조하는 표의 77%만 담고 있었다. 색인되지 않은 표를
묻는 질의 314건은 모집단에서 그냥 사라졌다.

**원인**: 코퍼스를 `point3_reconstruction_cost.build_table_paths()` 가 `None` 을
돌려주지 않는 표로 한정하고 있었다. 그 함수는 **헤더 복원 실험**의 것이고, 내부에서
`tree_reconstruct_hitab_raw.align()` 이 raw 격자와 데이터 행렬을 **셀 값 일치율**로
맞춰 보고 0.90 미만이면 표를 버린다. test 에서 124표가 여기서 탈락했고 탈락 사유는
전부 값 일치율 미달이었다(축 정합 실패 0건, 그중 상당수는 .89대).

**핵심**: 그 정렬 결과는 색인에 **쓰이지 않는다.** `hitab_corpus()` 가 `pt` 에서 읽는
것은 `gold_rp`/`gold_cp`/`n_r`/`n_c` 뿐이고, 이들은 각각 `bt.row_path(i)`,
`bt.col_path(j)`, `bt.n_rows`, `bt.n_cols` 를 그대로 담고 있다 — 파싱된 헤더 트리라
raw 격자와 맞출 일이 없다. **아무것도 사지 않고 표 23%를 지불한 관문이었다.**

**수정**: `build_corpus` 가 경로를 파싱된 표에서 직접 만든다. 관문 삭제.

### (2) gold 셀 해석 누락 — test 1,584질의 중 937건만 채점 (41% 누락)

**대상**: `rag_agent/bench/hitab.py: _coords_of()` / `_coord_offset()`

**증상**: 질의 351건이 gold 셀 0개로 해석돼 모집단에서 빠졌다. (1)의 314건과 합쳐
채점된 질의는 937/1,584 = 59.2% 였다.

**원인 두 가지**:
- `_coords_of` 는 `quantity_link` 만 읽고, 그중에서도 값이 **숫자로 파싱되는 것만**
  남긴다. 답이 라벨인 질의("어느 지역이 더 높았나" — argmax/argmin/비교)는
  `quantity_link` 가 비어 있어(test 300건) 또는 값이 텍스트여서(51건) 전멸한다.
- 좌표를 데이터 행렬로 옮길 때 (헤더행수, 헤더열수) 를 6×6 격자에서 **값이 맞아떨어지는
  조합을 찾는 방식**으로 추측한다.

**핵심**: 추측할 필요가 없다. HiTab 은 `answer_formulas`(`=B3`, `=SUM(B8:B10)`)와
`reference_cells_map`(`{"B3": "(0, 1)"}`)로 답 셀을 직접 준다. 전수 확인:
`linked_cells` 좌표는 **raw `texts` 격자**를 가리키고(quantity_link 1,725/1,740,
entity_link 4,189/4,189 값 일치), Excel 참조는 링크 좌표에 행 +3·열 +0 이며
(1,882/1,882), raw 트리와 hmt 트리를 **함께 걸어가면** raw 행/열 → 데이터 행/열이
정확히 나온다(행 538/538표, 열 505/538표; 나머지는 병합 헤더라 값 대조로 맞춘다).

**수정**: 새 모듈 `rag_agent/bench/hitab_grid.py` 가 이 매핑과 gold 해석을 담당하고,
`hitab.load_queries()` 가 그리로 간다. 추측 코드는 삭제했다.

**추가 (같은 계열)**: 헤더 셀이 두 열에 걸쳐 병합돼 있으면 raw 트리에는 한 번,
파싱된 트리에는 두 번 나타나 트리 동행이 그 열을 못 채운다. 그 열의 gold 셀은
`gold_unmappable` 로 떨어졌다(32건). 두 축 다 단조이므로 **데이터 밴드의 raw 줄 수와
파싱된 표의 줄 수가 같을 때만, 그리고 트리 동행 결과와 어긋나지 않을 때만** 순서로
짝지어 빈칸을 메운다(`HitabTable._fill_gaps`).

**결과**: 1,584 중 **1,245** 가 데이터 셀 gold(주지표), 336 이 헤더 답, **3건만 제외**.
검증: `tests/test_hitab_grid.py` — 단일 참조 gold 셀의 값이 공표된 answer 와
**995/996** 일치.

### (3) BGE 쿼리 지시문이 붙지 않았다

**대상**: `rag_agent/retrieve/encoders.py` / `retrieve/hybrid_index.py`

**증상**: `default_prefixes()` 가 BGE 의 쿼리 지시문
("Represent this sentence for searching relevant passages: ")을 정의해 두고 있고
테스트까지 있는데, **주 검색 경로 어디에서도 호출되지 않았다.** `HybridIndex`
`cell_rank_dump.py` `corpus_dump_vs_cell.py` 전부 맨 질문을 인코딩했다. BGE v1.5 는
질의-문서 비대칭을 이 접두어로 학습한 모델이다.

**수정**: `SentenceTransformerEncoder.encode_query()` 를 만들고 접두어를 모델 이름에서
자동 결정한다. `HybridIndex._dense_scores` 가 그것을 부른다. 비용은
`--no-query-prefix` 로 잰다.

### (4) 채점기가 질의를 조용히 떨어뜨렸다

**대상**: `analysis/cell_rank_dump.py`

**증상**: 코퍼스에 없는 gold 셀은 `ranks` 에서 **말없이 빠지고**, gold 가 하나도
안 남으면 `continue` 로 레코드 자체가 사라진다. 남은 gold 부분집합으로 계산한
all-covered 는 정의상 부풀려진다.

**수정**: 대체 계측기 `scripts/retrieval_accuracy.py` 는 분할의 모든 질의를
레코드로 남기고, 못 푸는 질의는 사유와 함께 `excluded_by_reason` 에 센다.

## 2026-08-31 — `gold_parts()` 콤마 오파싱 (Phase 4f/4g)

**대상**: `analysis/phase4_summary.py: gold_parts()`

**증상**: AIT-QA의 천단위 콤마 정답(`'1,179'`, `'13,200'` 등)에서 모델이
콤마 없는 동일 수치(`'1179'`, `'13200'`)를 출력했는데 오답으로 채점됐다.

**원인**: `ast.literal_eval('1,179')`는 괄호 없는 콤마 표현식을 튜플
리터럴로 파싱해 `(1, 179)`를 돌려준다. `gold_parts`가 이를 다중 gold
`['1', '179']`로 취급 → `em()`의 다중 gold 분기 진입 → 예측을 콤마로 split한
`['1179']`와 길이가 달라 즉시 0 반환. `norm_em`(= 통화기호 제거, 천단위 콤마
제거)은 호출조차 되지 않았다.

**성격**: 사전등록된 정규화 규칙(천단위 콤마 제거)이 실행되지 않은 구현
버그다. 채점 규칙 변경이 아니며 PREREGISTER 개정이 아니다.

**수정**: 문자열이 `[` 또는 `(` 로 시작할 때만 `ast.literal_eval` 컨테이너
파싱을 시도한다. 그 외는 단일값으로 취급한다. `em()`, `norm_em()`, `_same()`,
R1 규칙은 변경하지 않았다.

**영향**: 전 1156행 재채점 결과 0→1 7행, 1→0 0행. 전부 `aitqa`
(P1 1, P4 3, gold_cell 3). 다른 4개 pool 변동 없음.
query_id: q-401(P1), q-146/q-196/q-118(P4), q-396/q-118/q-267(gold_cell).

**재집계 산출물**: `results/phase4/FINAL.md` (`analysis/phase4g.py`).
수정 전 수치는 `results/phase4/final_summary.md`, `phase4b.md`, `phase4e.md`에
그대로 남겨 두었다.
