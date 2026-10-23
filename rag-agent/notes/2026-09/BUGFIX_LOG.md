# BUGFIX_LOG

## 2026-09-10 — 채점한 문맥과 리더가 받은 문맥이 서로 달랐다 (P1 2건 + 2건)

외부 감사(`T2_비교군_문헌_코드감사_2026-09-10.md`)가 지목한 결함을 코드에서 재현하고
고쳤다. **네 건이 전부 같은 뿌리다: 검색이 점수를 매긴 셀 집합과 리더에게 실제로 전달된
셀 집합이 같은 집합이 아니었다.** 표 1 과 표 2 가 "같은 운영점"에 있다는 것이
(`CLAUDE.md` §0.2) 이 리포의 전제이므로, 이건 지표 정의가 깨진 것이다.

재현: `tests/test_baseline_units.py::test_rowcol_delivers_the_subtable_it_scored`,
`tests/test_arm_fairness.py::test_budget_never_scores_a_cell_it_did_not_deliver`.

### (1) P1 — `rowcol_select()`: 교집합으로 채점하고 합집합을 배달했다

**대상**: `scripts/retrieval_accuracy.py: rowcol_select()`

**증상**: `RowColRetrieval` (TableRAG, Chen et al. NeurIPS 2024 §4.2) 은 상위 K 행과
상위 K 열이 만나는 **sub-table** 을 리더에게 준다 — 공식 구현은
`df.iloc[row_ids, col_ids]`. 우리 코드는 `got = R & C` 로 교집합을 채점해 놓고
`ctx = (rows[:k] + cols[:k])[:dump]` 로 **선택된 행 문장과 열 문장을 통째로 이어붙여**
전달했다. 리더는 교집합 밖의 값을 읽는다.

**최소 반례**: 첫 행 `{a,b}`, 첫 열 `{a,c}` → 채점 `{a}` 1칸, 배달 `a,b,c`.

**크기**: 주 모집단 991건에서 배달된 값 칸이 평균 **166.3 → 36.5** 로 줄었다(채점한
셀은 22.95). 감사는 저장된 hit 와 전달 문맥 기준 hit 가 1,581건 중 최소 526건에서
어긋난다고 보고했다.

**수정**: `subtable_lines()` 를 만들어 **채점한 교집합을 그대로 sub-table 로 잘라**
배달한다. 렌더링은 코퍼스 자신의 `line_text()` 를 재사용하므로 문맥과 색인 단위의
표기가 갈라지지 않는다. `build_corpus()` 가 `rowcol` 일 때만 `{table_id: (표, 제목)}`
를 함께 돌려준다 — sub-table 은 질의 시점에 잘리므로 표가 살아 있어야 한다.

**점수에 미친 영향**: **검색 정확도는 바뀌지 않는다.** 채점 집합이 원래 교집합이었기
때문이다 — 고친 코드로 전수 재실행해 판정 변동 **0건**을 확인했다
(`results/audit_fix_20260910/t_rowcol_values_fix.json`, 0.4811 / 0.5179 / 0.4889 그대로).
바뀌는 것은 **리더 쪽**이다. **표 2·2b 의 RowCol 답변 EM 은 새 문맥으로 리더를 다시
돌리기 전까지 보류한다.**

### (2) P1 — `budget_select()`: 배달하지 않은 단위의 셀을 채점했다

**대상**: `scripts/retrieval_accuracy.py: budget_select()`

**증상**: `dump` 는 **쓰기만** 멈추고 훑기는 멈추지 않았다. 셀을 배달하지 않는 문서
(TableRAG 의 schema·stub 문서)가 gold 앞에 오면, `dump` 를 넘어선 단위까지 `got` 에
들어가 **gold 가 없는 문맥이 HIT 로 기록된다.**

**최소 반례**: 셀 0개 문서 둘 + gold 문서 하나, `dump=2` → 검색은 성공, 리더 문맥에
gold 없음.

**수정**: **배달을 채점에 맞춘다** — `dump` 가 켜져 있으면 예산이 고른 단위를 **전부**
쓴다. `dump` 는 스위치이지 두 번째 예산이 아니다. `dump=0`(검색 전용)에서는 동작 불변.

⚠️ **반대 방향으로 고치면 안 된다** (이 세션이 처음에 그렇게 고쳤다가 되돌렸다).
훑기를 `dump` 에서 끊어도 채점=배달은 회복되지만, 그러면 운영점이 **"20셀"에서
"20단위"로** 옮겨 간다. 그 차이는 **셀 0개 단위를 갖는 arm 에서만** 생기므로
(`tablerag` 평균 배달 셀 20.87 → 9.90) 우리가 한 baseline 에만 handicap 을 얹는 것이
된다. `CLAUDE.md` §0.1 은 운영점을 **리더에게 주는 셀 수**로 못 박고 있고,
`tests/test_arm_fairness.py` 가 막으려는 것이 정확히 이 종류다. 단위를 제한하고
싶으면 그 손잡이는 따로 있다 — `--max-units` (TableRAG 네이티브 top_k=5 용).

**점수에 미친 영향**: **검색 정확도는 바뀌지 않는다** — 채점 쪽 훑기가 그대로다.
바뀌는 것은 **리더 문맥**이고, 셀 0개 단위를 갖는 `tablerag` arm 에서만 실제로 달라진다
(다른 arm 은 단위마다 셀이 1개 이상이라 20셀 예산이 20단위 안에서 끝난다).
따라서 **`tablerag` 두 행의 답변 EM 도 RowCol 과 같이 재실행 전까지 보류한다.**

### (3) gold 좌표를 64개로 잘라 저장했다

**대상**: `scripts/retrieval_accuracy.py` — `r["gold_cells"] = sorted(list(gold))[:64]`

**증상**: 검색 판정은 **전체 gold** 로 하는데, 기록에는 64개만 남았다. 이후
`answer_accuracy.py: gold_context()` 가 그 잘린 집합으로 gold/oracle 리더 문맥을
만든다 — 헤더답 29건에서 실제 65~408개 좌표 중 64개만 들어갔다.
**주 모집단 991건은 gold 가 1개라 영향이 없다.**

**수정**: 자르지 않는다.

### (4) 표 생성기가 옛 리더 레그를 읽고 있었다

**대상**: `analysis/accuracy_tables.py`

**증상**: 2026-09-09 제목 버그 수정 후 `t_s3c_hybrid_answer_gold_v2` 를 재실행해 뒀는데,
생성기는 조건 이름을 파일 이름으로 그대로 써서 수정 **전** 레그(`..._answer_gold`,
`..._answer_oracle`)를 계속 읽었다. 그래서 문서에 옛 천장이 남았다:
주 모집단 `gold` **0.9576 → 0.9566** (949→948/991), `oracle` **0.8819 → 0.8809** (874→873).

**수정**: `LEG` 딕셔너리 한 곳이 조건→파일 이름을 정하고, 모든 자리가 `leg_file()` 을
지난다. `analysis/compose_oracle.py` 는 어느 gold 레그로 합성할지를 인자로 받는다
(기본 `gold_v2` → `oracle_v2`). 한쪽 레그에만 있는 메타 키는 같다고 우기지 않고
`meta_unverified` 에 적는다.

### 함께 넣은 계측 (같은 사고의 재발 방지)

- `context_sha` — 검색 기록과 답변 기록 양쪽에. 채점한 문맥과 답한 문맥이 같은지가
  파일만 보고 확인된다. 이 감사가 손으로 다시 해야 했던 일이다.
- `code_revision` · `chunk_overlap` — 결과 요약에 재현 정보를 적는다.

### 재실행

`results/audit_fix_20260910/`. 검색 전용이므로 GPU 가 필요 없다.

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

## 2026-09-09 — 리더 `gold`/`oracle` 조건이 색인에 없는 제목을 썼다

**대상**: `scripts/answer_accuracy.py: gold_context()`

**증상**: `gold_context` 가 만든 gold 셀 문장이 같은 질의의 검색 문맥 어디에도
문자열로 존재하지 않는다. 주지표(all·`aggregation=none`·m=1) 검색 성공 906건 중
**188건(20.8%)**.

    gold_context : In the table 'career statistics', among cska sofia > total, ...
    색인/검색 문맥 : In the table 'Hristo Yanev: career statistics', among cska sofia > total, ...

**원인**: 색인은 `scripts/retrieval_accuracy.py: build_corpus()` 에서 ToTTo 페이지
제목을 앞에 붙인다(`results/tableconf/totto_page_titles.json`, 1,851표). `gold_context`
는 `tab.title`(HiTab 이 가진 ToTTo **섹션** 제목)을 그대로 썼다. 같은 규칙이 리포에
세 벌 있었고 — `caption.effective_titles(mode="page")`, `build_corpus` 안의 인라인
3줄, 그리고 `gold_context` 의 **누락** — 셋이 어긋났다.

**성격**: 표기 문제가 아니라 측정 결함이다. **`retrieved` 조건과 `gold`/`oracle`
조건이 서로 다른 렌더링 위에서 비교되고 있었다.** 제목 내용이 EM 을 움직인다는 것은
이미 측정돼 있다 — CLAUDE.md §5 생성 제목 기각(.3082 → .2239, p=.0002). 방향은
`gold` 조건이 **불리한** 쪽이므로 리더 천장은 과소평가일 수 있다.

**수정**: 규칙을 `rag_agent/serialization/caption.py: with_page_title()` 한 곳으로
모으고 세 호출자가 전부 그것을 부른다. 고친 뒤 188 → **0**.

**검사**: `tests/test_gold_context_matches_index.py` — 검색 성공·m=1 질의는
`gold_context` 문장이 리더가 받은 문맥 줄에 그대로 있어야 한다. 이 결함이 있으면
실패한다. 전체 82 passed.

**영향 — 아직 재실행하지 않았다**: `gold`/`oracle` 조건의 산출물이 전부 이 결함
위에서 나왔다. `results/retrieval_accuracy/t_s3c_hybrid_answer_gold*.jsonl`,
`VERDICT_PROMPT.md`(gold .8648 → .8639), `TABLES.md` 의 gold 행, 그리고 그 값을
쓰는 CLAUDE.md §5 예산 축소 기각(“distractor 0 인 gold 조건조차 0.8474”)이
해당한다. **재실행 전까지 리더 천장 수치는 하한으로 읽는다.** `retrieved` 조건은
색인 문장을 그대로 쓰므로 영향 없음 — 표 1 과 표 2b 의 검색 arm 수치는 무관하다.
