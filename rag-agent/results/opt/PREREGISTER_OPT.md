# PREREGISTER_OPT — 무학습 검색 최적화 사전등록

- 작성: 2026-09-15 (KST). 분할 생성 2026-09-14T21:57:42+09:00.
- 상태: **GATE 0 승인 2026-09-15** (§12). 이 파일·`split.json`·`TEST_FROZEN.md`·`scripts/opt_split.py` 를
  한 커밋으로 동결한다. 동결 후 이 문서는 변경하지 않는다.
  어긋남이 생기면 이 파일을 고치지 않고 `results/opt/DEVIATIONS.md` 에 날짜와 함께 추가한다.
- 모델 학습·파인튜닝 없음. 공개 체크포인트 추론만.
- 이 문서에는 결과 수치가 없다. 아래 수치는 분할 모집단 크기와 기존 기록의 인용뿐이며 출처를 붙인다.

---

## 0. 데이터·분할

| 항목 | 값 | 출처 |
|---|---|---|
| 원본 | HiTab `test_samples.jsonl` sha256 `cfb8296a…c1f339` | `results/opt/split.json` |
| 분할 단위 | 표. `sorted(set(table_id))` 를 `random.Random(42).shuffle`, 앞 `round(0.4·n)` = dev | `scripts/opt_split.py` |
| 표 | 전체 538 / dev 215 / test 323, 교집합 0 | `split.json` |
| 질의 | dev 592 / test 992 | `split.json` |

유형 정의 (데이터셋 필드만 사용, `scripts/opt_split.py: qtype`):
`excluded` gold 해석 불가 · `any` 답이 헤더 · all 모드 중 `single` aggregation=none·gold 1셀 ·
`multi` aggregation=none·gold ≥2셀 · `arith` aggregation≠none.

| 유형 | dev | test | 합 |
|---|---:|---:|---:|
| single | 363 | 628 | 991 |
| multi | 19 | 19 | 38 |
| arith | 73 | 143 | 216 |
| any | 136 | 200 | 336 |
| excluded | 1 | 2 | 3 |
| 계 | 592 | 992 | 1,584 |

유형×aggregation 전수: `split.json: type_x_aggregation`.

관측 사항 (해석 없음):
- test 표 323 중 131 이 dev 표와 **제목(`tab.title`)이 글자 그대로 같다** (`split.json`).
- 분할 전에 이미 전 질의(dev·test 모두)의 기존 검색 결과를 관측했다:
  `results/retrieval_accuracy/t_s3c_hybrid_records.jsonl` (1,584행, all 모드 .89 n=1,245),
  `t_s2_hybrid.json` (all 모드 .7711 n=1,245) 외 같은 디렉터리의 arm 전부.
- 연구실 커밋 `e67af75`(2026-09-14 17:30 KST, 분할보다 앞섬)가 같은 설정의 유형별 결과를 추가했다:
  `t_s3c_hybrid_qtype.json` (dev·test 합산 single .9142 n=991 · multi .8421 n=38 · arith .7870 n=216).
  대조(2026-09-15): 유형 판정 `retrieval_accuracy.query_type` 은 `opt_split.qtype` 과 1,584 질의 전부 일치.
  질의별 `correct` 는 `t_s3c_hybrid_records.jsonl` 과 1,584건 전부 같다. `gold_cells` 는 any 모드 29건에서
  새 파일 쪽이 옛 파일의 상위집합이다.
- 2026-09-15 사용자 지시로, 기존 v2 기록(분할 전 실행, dev·test 합산)에서 셀 예산 없는 유형별 결과를 새로 읽었다:
  `results/evaluation_v2/type_topm_exact_v2.json` (`analysis/type_accuracy_v2.py`). 규칙: 상위 m 셀 문장 = gold 집합이면 1
  (m=|gold|, 안정 정렬 순서). 본 방법·MT2Net 의 single/multi/arith 값과, 본 방법 성공 질의에 정답 셀 문장만 준 답변 EM
  (`s3c_v2_answer_gold_modeall`)의 유형별 값. 새 검색·생성 실행은 없다.
- 2026-09-15 GATE 0 승인 전, dev 질의만으로 baseline 실패를 분류했다: `results/opt/diag/topm_failures_dev.json`
  (`analysis/topm_failure_diag.py`, v2 기록 순위 사용). dev baseline top-m 안정 정렬 값(single 203/363 · multi 5/19 ·
  arith 12/73)과 single 실패 160건의 순위·위치·어휘 분류를 관측했다. test 질의는 분류하지 않았다.
- 계획서의 진단 수치(형제 셀 74%, 점수 차 중앙값 .032, .01 미만 22%, 다른 표 26%,
  산술 41%·다중 53%, 산술 122/168)의 출처 파일을 리포에서 찾지 못했다(grep). 계획서의 산술 분모 168 은
  이 모집단의 arith 216 과 다르다.

## 1. 공통 조건

- 검색 코퍼스: **538표 전체의 셀** (67,664 단위, 기존 `t_s3c_hybrid` 와 같은 haystack). test 표는
  색인에 방해물로 남고 test 질의는 채점하지 않는다. [§11-1 승인]
- 인코더 `BAAI/bge-base-en-v1.5` rev `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`, 쿼리 접두어 현행
  (`encoders.py`), `--embed-overflow error`.
- 하이브리드 α=0.7 고정: `s = 0.7·minmax(dense) + 0.3·minmax(BM25)` (`hybrid_index._minmax`,
  `sparse_bm25.SparseBM25`, `_tokenize`). 정렬 `np.argsort(-s, kind="stable")`.
- **셀 예산 없음.** 판정은 전 코퍼스 순위로 한다(§2). 기록에 남기는 상위 20 단위는 재현 점검 깊이이며 지표가 아니다.
- 리더/생성기 `Qwen/Qwen2.5-7B-Instruct` rev `a09a35458c702b33eeacc393d103063234e8bc28`,
  `LocalQwenLLM` 4bit NF4, `complete(temperature=0.0)` (= `do_sample=False`), `torch.manual_seed(42)`.
- dev 계측기는 dev 질의 ID 만 받는다. test 질의는 STAGE 6 스크립트 한 번만 채점한다.
  산출물이 이미 있으면 실행을 거부한다(덮어쓰기 금지). 경로 `results/opt/stage{N}/`.

## 2. 지표 정의

gold, 모드(all/any)는 `load_queries` → `hg.gold_target` 그대로. `G` = gold 셀 집합, `m = |G|`.
`s(u)` = 그 단계의 최종 순위 점수(STAGE 1·5 하이브리드, STAGE 2 RRF, STAGE 4 상위 20 안 CE 점수).
순위 `rank(u)` = 안정 정렬 위치(1부터).

| 지표 | 정의 | 대상 |
|---|---|---|
| **top-m 엄격** (주) | all: `min_{g∈G} s(g) > max_{u∉G} s(u)`. any: `max_{g∈G} s(g) > max_{u∉G} s(u)`. 동점이면 실패. m=1 이면 top-1 엄격 | 전 유형 |
| top-m 안정 정렬 | all: `set(order[:m]) == G`. any: `order[0] ∈ G`. 동점은 단위 인덱스 순 | 전 유형 |
| MRR | 질의 평균 `1/rank(가장 앞선 gold)`, 상한 없음(전 코퍼스 정렬) | 전 유형 |
| 실패 분류 | single 의 top-m 엄격 실패를 `동점`(안정 정렬 1위가 gold) / `유형1`(1위 ≠ gold, 같은 표) / `유형5`(다른 표)로 전수 분류 | single |
| 점수 차 | single 실패 질의에서 `max_{u≠gold} s(u) − s(gold)` (≥0). 중앙값, `<.01` 비율. (a) 실패 전체 (b) 유형1 각각 | single |

점수 차는 조건마다 질의별 min-max 후 값이라 조건 간 척도가 같지 않다(기록만).

## 3. 통계·채택 규칙

- 부트스트랩: 대응(같은 질의 재표집). `rng = np.random.default_rng(42)`,
  `idx = rng.integers(0, n, (10000, n))`, `Δ_b = mean(x[idx]) − mean(y[idx])`, 백분위 2.5/97.5.
- 이진 지표 p: McNemar 정확검정 양측 `min(1, 2·Binom.cdf(min(b,c); b+c, .5))`, `b+c=0` → 1.
- Holm: 표준 step-down, 단조 보정, 1 상한.
- **기준 충족** (단계별 판정 지표에서): `Δ > 0` 이고 95% CI 하한 > 0. 비교가 여럿인 단계는 추가로
  Holm 보정 p < .05.
- 가족: STAGE 1 {T1, T2*, T3} = 3 · STAGE 5 {T=1,3,5} = 3. STAGE 2, 4 는 단일 비교(보정 없음). [§11-6 승인]
- 기준 충족은 채택의 **필요조건**이다. 채택은 각 GATE 에서 사람이 한다. 미충족 조건은 채택할 수 없다.
- 여럿이 충족하면 제시 순서: Δ 점추정 큰 순 → Holm p 작은 순 → 나열 순.
- 각 단계의 기준 = 직전 단계까지 채택된 파이프라인(없으면 baseline).
- 연속 지표(MRR, 점수 차)는 CI 만 보고하고 판정에 쓰지 않는다.
- 모든 조건·모든 지표·모든 유형을 표에 싣는다. 판정 지표가 아닌 것도 싣는다.

| 단계 | 판정 지표 | 판정 모집단 (dev n) |
|---|---|---|
| 1 | top-m 엄격 | single 363 |
| 2 | top-m 엄격 | multi+arith 92 [§11-3 승인] |
| 3 | 삭제 (§6) | — |
| 4 | top-m 엄격 | single 363 |
| 5 | top-m 엄격 | single 363 |

---

## 4. STAGE 1 — 셀 문장 템플릿 (겨냥: 유형1)

`join_path` = 현행 `" > "` 결합. `title` = `with_page_title(tab.title, page_title)`.
`value` = `fmt_value`. `row_leaf = row_path[-1]` (없으면 빈 문자열), `col_leaf` 동일.

- **baseline**: `caption_sentence(title, row_path, col_path, value, template=structural_compact)`
  — 기존 `--template s3c`. 예: `In the table '…', among percent > food service, the value of eastern ontario > french-language workers is 52.1.`
  재현 점검: dev 질의마다 baseline 안정 정렬 상위 20 단위의 셀 순서가 `results/evaluation_v2/s3c_v2_records.jsonl` 의
  `context_units` 셀 순서와 같아야 한다. 1건이라도 다르면 멈추고 보고한다. 임베딩은 그 실행의 인코더 메타데이터
  (장치 포함)와 같게, 질의는 1건씩 인코딩한다(그 실행과 같은 호출 형태).
- **T1_leaf_first**: `head = " ".join(비어있지 않은 row_leaf, col_leaf)`,
  `ps = " / ".join(비어있지 않은 join_path(row_path), join_path(col_path))`,
  `tail = [ps(있으면), f"table '{title}'"(title 있으면)]`,
  텍스트 = `f"{head} — {value}."` (head 없으면 `f"{value}."`) + (tail 있으면 `f" ({', '.join(tail)})"`).
  dense·BM25 두 다리 모두 이 텍스트.
- **T2_field_split**: 필드 `leaf = f"{head} — {value}"`, `path = ps`, `caption = title`.
  각 필드를 같은 인코더로 따로 임베딩(같은 문자열은 한 번), 빈 필드는 기여 0.
  `dense = w_leaf·cos(q,leaf) + w_path·cos(q,path) + w_cap·cos(q,caption)` (원 코사인 가중합 후 min-max).
  BM25 다리는 **baseline 문장** 그대로.
  격자 6점, 음수 없음: (w_leaf, w_path, w_cap) = (.4,.2,.4) (.4,.3,.3) (.5,.2,.3) (.5,.3,.2) (.6,.2,.2) (.6,.3,.1).
  6점 전부 모든 지표를 보고한다. T2* = dev single top-m 엄격 최고점(동점 → MRR → 격자 순).
  T2* 는 같은 dev 에서 골라 같은 dev 로 판정한다(절차 사실로 기록).
- **T3_no_caption**: baseline 렌더러에 `title=""`. 이때 `render` 는 S2 경로 문자열 `"{path}: {value}"` 로
  떨어진다(`templates.py:124`). 기존 `--template s2` 텍스트와 같은지 실행 시 텍스트 해시로 확인해 보고한다.
  같다면 이 조건의 dev+test 합산 결과(.7711, `t_s2_hybrid.json`)는 이미 관측된 것이다. 표 ID 는 메타데이터로만 보관.

측정(dev, 4조건 + T2 격자 6점): top-m 엄격, top-m 안정 정렬, MRR (전 유형),
실패 분류(동점/유형1/유형5) 건수, 점수 차 중앙값·`<.01` 비율, 동점 건수.

## 5. STAGE 2 — 질의 분해 (겨냥: 유형6 다중·산술)

- 적용: **dev 전 질의에 균일 적용** (aggregation·gold m 으로 고르지 않음). [§11-4 승인]
- 생성 `max_tokens=256`, §1 조건. 프롬프트 전문:

system:
```
You decompose questions about statistical tables into search queries for retrieving table cells.
```
user:
```
Question: {question}

Each value needed to answer the question is one table cell. Write one short search query per cell, naming what the cell's row and column describe. If one cell is enough, write exactly one query. Output only a JSON array of strings, e.g. ["query 1", "query 2"].
```

- 파싱: 출력의 첫 `[` 부터 마지막 `]` 까지 `json.loads`. 비어 있지 않은(strip 후) 문자열 ≥1 개의 리스트여야 한다.
  완전 중복 문자열은 순서 유지로 제거(제거 건수 보고). 실패 → 원 질문 하나로 대체, **실패 건수 보고**.
- 검색: 하위 질의마다 STAGE 1 채택 텍스트로 §1 하이브리드 전 코퍼스 정렬.
  원 질문 목록은 넣지 않는다(계획서대로 하위 질의만).
  RRF `score(u) = Σ_i 1/(60 + rank_i(u))`, 내림차순, 동점은 단위 인덱스 안정 정렬. 판정 점수 = RRF 점수.
- 측정(dev, 전 유형): top-m 엄격, top-m 안정 정렬, MRR, 하위 질의 수 분포(1,2,3,…개 질의 수),
  파싱 실패 수, 중복 제거 수, 기준 대비 Δ·CI·McNemar.

## 6. STAGE 3 — 삭제 (2026-09-15 GATE 0)

초안은 상위 k 셀의 같은 행·열 셀을 문맥에 더하고 4096 토큰으로 채우는 단계였다. 판정 지표가 셀 예산 없는
순위 지표(top-m 엄격)로 바뀌었고, 문맥에 셀을 더하는 조작은 순위를 바꾸지 않으므로 판정할 지표가 없다. 실행하지 않는다.

## 7. STAGE 4 — 크로스인코더 재정렬 (겨냥: 유형1·3)

- `cross-encoder/ms-marco-MiniLM-L-6-v2` rev `c5ee24cb16019beea0893ab7796b1df96625c6b8`, 무학습.
  `max_length=512`(초과 쌍 수 보고), GPU fp32, 배치 20.
- 기준 순위의 상위 20 단위(재정렬 후보 수, 셀 예산 아님)만 `(question, 채택 템플릿 셀 문장)` 점수로 내림차순 재정렬
  (동점은 원 순위), 21위 이하 불변.
- top-m 엄격: gold 전부가 상위 20 후보 안에 있고, all 은 gold CE 점수 최솟값 > 나머지 후보 CE 점수 최댓값,
  any 는 gold CE 점수 최댓값 > gold 아닌 후보 CE 점수 최댓값. gold 가 하나라도 20 밖이면(any 는 전부 밖이면) 실패.
- 측정(dev): top-m 엄격, top-m 안정 정렬, MRR (전 유형), 재정렬 전후 gold 순위 변화 분포
  (single: 1→1 / 1→강등 / 승격→1 / 개선(1 아님) / 불변 / 악화 / 20 밖), 1위 승격:강등 McNemar,
  유형3 하위집단 수치, 질의당 추가 지연.
- 지연: CE 토크나이즈+순전파+정렬, `torch.cuda.synchronize()` 포함 `time.perf_counter`, 모델 로드 제외,
  앞 5 질의 워밍업은 지연 통계에서 제외(채점은 포함). 평균·중앙값·p95 초.
- 유형3 하위집단 (질문 소문자, 사전 고정 정규식):
  - NEG `\b(not|no|never|none|neither|nor|without|except|excluding|other than|non)\b|n't\b`
  - TIME `\b(1[89]\d{2}|20\d{2})\b|\b(january|february|march|april|june|july|august|september|october|november|december|years?|months?|quarters?|decades?|century|before|after|since|until|during|earlier|later|previous|recent|recently|season)\b`
  - 유형3 = NEG 또는 TIME. NEG, TIME, 유형3, 나머지 각각 n 과 모든 지표 보고.

이전 재정렬 기각 기록과 이번 조건 (기록 인용, 해석 없음):

| | R① 2026-09-03 | E3 2026-09-07 | 탐색 2026-09-08 | 이번 STAGE 4 |
|---|---|---|---|---|
| 출처 | `RESULTS.md` §12, `results/rerank/VERDICT.md` | `RESULTS.md` §18 | `HANDOFF-2026-09-08-2.md` §3.3 | 이 문서 |
| 재정렬기 | `BAAI/bge-reranker-large` 기성품 | `models/ce-cell-p1` (bge-reranker-base 도메인 학습) | `bge-reranker-base` 기성품 | `ms-marco-MiniLM-L-6-v2` 기성품 |
| 1단계 인코더 | `bge-base-cell-ft-p0` **파인튜닝**, α=0.8, title page | p1 **파인튜닝** hybrid | 기성품 arm (문서 표기) | `bge-base-en-v1.5` 기성품 α=0.7 |
| 모집단 | 구 HiTab dev 830 (m=1) | 단일 dev clean 809 | 문서에 명시 없음 | 이번 dev single 363 (HiTab test 에서 분할) |
| 재정렬 폭 | top-10 | top-10 | 문맥 20셀 | top-20 |
| 판정 지표 | hit@1 및 답변 EM | hit@1, hit@3 | gold 1등 비율 | top-m 엄격(동점 실패) |
| 기록된 결과 | hit@1 .7193→.6181, 67:151 p=1.26e-08; EM 주검정 p=.0058 | hit@1 .6959→.6378 | .6164→.5655, 114:163 p=.0039 | — |
| 사전등록 | 있음(`afb36e7`) | 있음(기각) | 없음(탐색) | 이 문서 |

계획서 서술 "당시 조건(파인튜닝 인코더 + EM 기준)" 과 기록의 차이: R① 판정에는 hit@1 도 있었고,
기성품 인코더 위 재정렬 탐색 기록(2026-09-08)이 따로 있다. `RESULTS.md` §11 은 MultiHiertt
within-doc 에서 기성품 bi-encoder 위 `cross` 열 +.037 을 기록한다.

## 8. STAGE 5 — 2단계 검색 (겨냥: 유형5)

### 5-1 표 캡션 생성 (색인 시점 1회, 538표 전부, 질의·gold 미사용)

입력: `title`; `COLS` = 열 경로(`join_path`) 열 순서 중복 제거 앞 20개 `"; "` 결합;
`ROWS` = 행 잎 라벨 행 순서 중복 제거 앞 20개 `"; "` 결합; `VALS` = 행 우선 순서 비어 있지 않은 데이터 셀 값 앞 5개 `"; "` 결합.
`max_tokens=96`, §1 조건, 출력 strip 후 첫 줄. 프롬프트 전문:

system:
```
You write short descriptive captions for statistical tables.
```
user:
```
Title: {title}
Column headers: {COLS}
Row headers: {ROWS}
Example values: {VALS}

Write one sentence (at most 40 words) describing what this table reports: the subject, the measures, the breakdowns, and the time period if shown. Use only the information above. Output only the sentence.
```

표 단위 텍스트: `f"{title}. {caption} Columns: {전체 열 경로 '; '}. Rows: {전체 행 잎 '; '}."`.
bge 토큰이 인코더 `max_seq_length` 를 넘으면 행 라벨을 뒤에서부터, 그다음 열 경로를 뒤에서부터 빼며 맞춘다. 잘린 표 수 보고.
생성 캡션 538개 원문을 `results/opt/stage5/captions.jsonl` 에 저장.

### 5-2 표 먼저, 그 표의 셀만

- 표 검색: 538 표 텍스트에 §1 하이브리드(α=0.7, BM25 는 표 텍스트) → 상위 T 표. **T ∈ {1, 3, 5}**.
- 셀 검색: 기준 셀 점수 계산 후 상위 T 표의 셀만 남기고 그 안에서 다시 min-max 합산
  (기존 `--corpus gold` 마스킹과 같은 방식) → 안정 정렬. STAGE 4 채택 시 이 순위의 상위 20 을 재정렬.
- gold 표가 상위 T 밖이면 top-m 엄격 실패.
- 측정(dev, T별 전부): top-m 엄격, top-m 안정 정렬, MRR (전 유형), 실패 분류(유형5 건수 변화 포함),
  gold 표가 상위 T 표 안에 든 질의 비율(진단, 전 유형), 기준 대비 Δ·CI·McNemar·Holm(3).

## 9. STAGE 6 — test 단 1회

- STAGE 1·2·4·5 채택 조합을 고정. 조합 순서: 템플릿(S1) → 분해·RRF(S2) → 표 제한(S5) → 상위 20 재정렬(S4).
  채택 안 된 단계는 건너뛴다.
- 실행 전 `TEST_FROZEN.md` 의 test ID 해시 `9480fe37…1ed2e3f` 를 대조, 불일치면 중단. `FINAL_TEST.md` 가 있으면 거부.
- test 992 질의(single 628 / multi 19 / arith 143 / any 200 / excluded 2) 1회. 결과 후 조정 없음.
- `results/opt/FINAL_TEST.md`:
  - baseline vs 최종: §2 전 지표 × 전 유형.
  - McNemar 정확검정, 부트스트랩 95% CI(§3), Holm 가족 = 이진 판정 4개
    {single·multi·arith·any top-m 엄격}.
  - dev Δ 와 test Δ 를 지표×유형별로 나란히.
  - 누적 표: baseline → +채택 S1 → +S2 → … 각 행을 같은 test 실행에서 측정, 직전 행·baseline 대비 Δ·CI·McNemar(보정 없음 표기).
  - 해석 문장 없음.

## 10. 기존 지시·기각 기록과 겹치는 지점 (사실 목록)

이 노선은 2026-09-14 사용자 계획서와 2026-09-15 GATE 0 승인(§12)에 따른다. 아래는 기존 문서와 겹치는 지점의 기록이다.

| 계획 요소 | 기존 기록 | 출처 |
|---|---|---|
| dev/test 분할 | "dev 없음 — test 전용" | `CLAUDE.md` §0.4 |
| top-m 엄격, 셀 예산 없음 | 운영점 하나(리더에게 주는 20셀)에서 질의 단위 정확도; 2026-09-15 사용자 지시 "budget·검색 20셀 고려하지 않음" 이 이 노선에서 그 운영점을 대체 | `CLAUDE.md` §0.1, 이 문서 §12 |
| MRR | recall@k 류 지표 미사용 | `CLAUDE.md` §0.1 |
| T1 어순 변경 | 문장 다듬기·어순 변경 기각 (n=830, EM, S2 계열) | `CLAUDE.md` §5 |
| T2 필드 가중 | 열 경로 BM25 가산 기각 (dev 830) | `CLAUDE.md` §5 |
| T3 캡션 제외 | `s2` arm 이 같은 텍스트일 수 있음(§4) | `t_s2_hybrid.json` |
| STAGE 4 | 재정렬 기각 3건(§7 표), "다시 제안하지 말 것" | `HANDOFF-2026-09-08-2.md` §3.3 |
| STAGE 5-1 | 제목 없는 코퍼스에 제목 생성 기각 (2026-08-27) | `CLAUDE.md` §5 |
| STAGE 5-2 | `cascade` 기각 (S2 에선 맞고 S3 에선 뒤집힘) | `CLAUDE.md` §5 |

## 11. GATE 0 확인 항목 — 2026-09-15 답

1. 검색 코퍼스 538표 전체 — 승인.
2. STAGE 3 의 k — STAGE 3 삭제(§6)로 해당 없음.
3. STAGE 2 판정 지표 = dev multi+arith(n=92) top-m 엄격. (초안의 쿼리 단위 Recall 은 셀 예산 지표라 삭제.)
4. STAGE 2 분해를 전 질의에 균일 적용 — 승인.
5. STAGE 3 비교 기준 — STAGE 3 삭제로 해당 없음.
6. STAGE 5 에도 Holm(3) 적용 — 승인.
7. 엄격 = 동점 실패 — 승인, top-m 엄격으로 일반화(§2).
8. 동결 = 승인 후 커밋 — 승인.

## 12. GATE 0 승인 기록 (2026-09-15)

- 사용자 지시 순서: "budget 은 고려하지 않고 … 1 or 0" → "검색 20셀 이런 것도 생각하지 마" → 실패 원인 설명 후
  "판정 지표를 top-m 규칙으로 바꾸고 GATE 0 을 승인하면 STAGE 1 부터 dev 에서 돌린다" 는 제안에 "진행해".
- 초안(커밋되지 않음) 대비 변경:
  1. 주지표 `top-1 엄격`(single) → `top-m 엄격`(전 유형, m=1 에서 같음). `top-m 안정 정렬` 보조 지표 추가.
  2. `Recall@20`(셀 단위)·`쿼리 단위 Recall`(20셀 문맥) 삭제. 셀 예산 20 삭제.
  3. STAGE 3 삭제(§6).
  4. STAGE 2 판정 지표 → top-m 엄격.
  5. STAGE 1 재현 점검: 20셀 판정 대조 → 상위 20 단위 순위 대조(v2 기록).
  6. STAGE 4 top-1 엄격 → top-m 엄격(상위 20 후보 안).
  7. STAGE 6 Holm 가족 5개 → 4개.
- 승인 전 관측(§0): `type_topm_exact_v2.json`(dev·test 합산, 안정 정렬), `topm_failures_dev.json`(dev 만).
