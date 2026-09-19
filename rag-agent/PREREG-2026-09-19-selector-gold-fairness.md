# 사전등록: LLM 셀렉터(top-K→1) 를 자기 표 안 검색(gold) 조건에서, 다른 셀 단위 arm에도 붙여 공정성을 검증한다 (2026-09-19)

## 왜 다시 재는가

`results/selector_top20_20260916/SELECTOR_TOP20.md`(top-20 후보 중 LLM이 1개 선택 후
그 문장만으로 답변)는 **ours(structural_leaf)에만** 적용됐고, **코퍼스 전체 검색**
(`--corpus split`) 조건에서 쟀다(.7316, n=991). 그런데 `results/SUMMARY_TABLES-2026-09-18.md`
§1/§3의 arm 비교표는 전부 **자기 표 안에서만 검색**(`--corpus gold`, TableRAG 원 논문의
표별 설정과 동일)이다. 사용자가 교수님께 "필터링으로 0.82까지 올랐다"고 말했는데
그 수치가 리포 어디에도 없고, `SELECTOR_TOP20.md` 자신의 문장("0.8 목표까지 남은 거리
.0684")은 0.8을 **아직 못 채운 목표**로 적고 있다 — 실측이 아닐 수 있다.

두 가지를 이 실험으로 고친다.

    1. 코퍼스 전체 검색으로 잰 셀렉터 수치를 gold(자기 표 안) 조건으로 다시 잰다
       → 0.82가 실측이면 이 조건에서 나와야 한다.
    2. 셀렉터를 ours 외 다른 "원자적 셀 후보" arm(mt2net, TableRAG-leaf/path)에도
       똑같이 붙여, 이득이 ours만의 것인지 원자적 셀 단위라면 다 받는 일반 효과인지 가른다.

chunk/trag_hetero/rowcol는 budget=20 조건에서 후보가 **통짜 유닛 1개**로 나온다
(오늘 세션에서 실측 확인: `cells_in_context=16, n_units=1`) — "20개 중 1개 선택"이라는
설계 자체가 성립하지 않는다. 이 arm들은 이 실험의 대상이 아니고, 결과 문서에 그 이유를
명시한다(조용히 빼지 않는다).

## 개입

`scripts/selector_top20_eval.py`에 오늘 추가:
- `--corpus gold`: `retrieval_accuracy.py` main()의 마스킹 공식과 동일하게, 쿼리마다
  자기 table_id 셀만 후보 풀로 남긴다(`sel = flatnonzero(table_ids == q.table_id)`,
  그 부분집합 안에서 minmax 재정규화 후 정렬 — retrieval_accuracy.py:879 그대로 재현).
- `--template mt2net`: `TEMPLATE_BY_NAME`에 MT2NET 추가(`sentence_disambiguation_eval.py`
  1줄), 기존 `build_leaf_corpus(template_name=...)` 파라미터를 그대로 활용.
- `--arm tablerag-leaf|tablerag-path`: 코퍼스 전체를 임베딩하지 않고, gold 조건이므로
  쿼리마다 자기 표 하나에 대해서만 `retrieval_accuracy.tablerag_units()`로 후보를 만들고
  `retrieval_accuracy.budget_select()`로 budget=20까지 채운다(§1/§3의 TableRAG 행과 동일
  선택기). 후보 개수는 arm이 자연히 만드는 만큼(8~19개 정도, 20 고정 아님) 그대로 쓴다.
- `--selftest`: `--corpus gold` 마스킹이 실제로 다른 표의 후보를 섞지 않는지, gold 셀이
  자기 표 마스킹 안에 있는지 GPU/LLM 없이 assert로 확인(통과 확인 완료, 아래 §검증).

## 모집단

정렬된 query_id 기준 **첫 150개** (n=150 파일럿) — `SELECTOR_TOP20.md` Addendum 1/2가
쓰던 것과 같은 파일럿 정의라 그 문서의 다른 파일럿 수치들과 직접 비교 가능.
리더/셀렉터 모두 `local:Qwen/Qwen3-8B?quantization=4bit`(SELECTOR_TOP20.md와 동일 모델 —
§1/§3 표는 `Qwen2.5-7B-Instruct`를 쓰므로, 표1의 .7567/.7333과 비교하는 것은 리더 모델도
다르다는 뜻이고 "sanity range check" 이상의 의미는 없다는 것을 미리 못박는다).

## arm (8개, 전부 `--corpus gold`)

    A1  ours(structural_leaf)  method=stuff  — 선택 없음, top-k 그대로 스터핑
    A2  ours(structural_leaf)  method=llm    — 20개 중 1개 선택
    B1  mt2net                 method=stuff
    B2  mt2net                 method=llm
    C1  tablerag-leaf          method=stuff
    C2  tablerag-leaf          method=llm
    D1  tablerag-path          method=stuff
    D2  tablerag-path          method=llm

## 예측 (실행 전 고정)

- **A2(ours+셀렉터, gold)가 0.82 근처(±.05)로 나올 것으로 예측한다** — split 조건 기준선
  .5923(k=1)→.7316(top20+셀렉터) 대비, gold 조건 기준선 자체가 이미 훨씬 쉬움(§1의 ours
  스터핑-20 .7567 vs split 스터핑 없는 k=1 .5923) 이므로 같은 상대적 개입 효과를 얹으면
  0.8대에 닿을 개연성이 높다. **이 예측이 맞으면 "0.82"는 이 조건(gold+셀렉터)의 실측치였을
  가능성이 높다는 뜻으로 적는다 — 그렇다고 사용자가 실제로 이 조건을 돌렸다고 단정하지는
  않는다(파일이 없다는 사실은 안 바뀐다).**
- **A2가 A1보다 유의하게 높을 것으로 예측한다** (McNemar p<.05) — split 조건에서 이미
  검증된 패턴(구조가 다른데도 상대적 순서는 유지될 것).
- **B2(mt2net+셀렉터)도 B1보다 오를 것으로 예측하되, 상승폭은 A2-A1보다 작거나 비슷할
  것으로 예측한다** — mt2net은 ours와 정확히 같은 원자적 셀 단위·같은 검색기이므로
  구조적 이점이 같다. 이득이 mt2net에서도 비슷하게 나타나면 "셀렉터가 ours만의 이점"이라는
  주장은 **기각**해야 한다(원자적 후보 단위라면 다 받는 일반 효과).
- **C2/D2(TableRAG-leaf/path+셀렉터)는 상승폭이 A2-A1보다 뚜렷이 작거나 없을 것으로
  예측한다** — TableRAG 후보(숫자열 min/max 요약, 범주형 dedup 등)는 §2가 이미 정리한
  것처럼 개별 숫자 셀을 직접 담지 못하는 경우가 많아(계층표 연산의 피연산자 대부분이 숫자
  셀), 후보 풀 자체의 품질 상한이 낮다 — 셀렉터가 완벽해도 못 채우는 천장이 있다.
- **이 패턴이 그대로 나오면**: "셀렉터가 대체로 원자적 후보 단위 전반에 이득을 주지만
  (mt2net도 오른다), 후보 자체가 정답을 못 담는 arm(TableRAG)에는 안 먹힌다" —
  이러면 "ours만의 이점"이 아니라 **"원자적 셀 인덱싱 + 셀렉터"의 결합 이점**이라고
  적어야 한다. 이게 0.82를 baseline과 나란히 놓을 때 방어 가능한 유일한 문장이다.
- 어느 예측이든 벗어나면 그대로 적고 표를 그대로 싣는다 — 결과를 보고 문장을 고르지 않는다.

## 고정

- 커밋 기준: 이 세션 시작 시점 HEAD(mt2net 행 추가만 반영, 미커밋).
- 인코더 `BAAI/bge-base-en-v1.5`, 하이브리드 α=0.7. 셀렉터/리더 `Qwen3-8B` 4bit,
  temperature 0(stuff/llm 기준 — self_consistency/order_ensemble 등 다른 method는 이
  실험 범위 밖). 채점 `hitab_exact_match_text`.
- 회귀 검사: 코드 변경 후 기존 `--corpus split`(기본값) 경로가 그대로인지 `--selftest`와
  별개로, 기존 `results/selector_top20_20260916/selector_top20_qwen3_8b.jsonl`의 로직과
  이번 스크립트의 split 경로가 수학적으로 동일함을 코드 대조로 확인(변경 diff가
  `--corpus gold`일 때만 분기하도록 작성됨 — 실행 재확인은 시간상 생략, diff 리뷰로 대체).

## 검증 (실행 전 완료)

`--selftest`: n=5 쿼리에서 (a) `--corpus gold` 마스킹이 다른 표의 후보를 절대 섞지 않음,
(b) gold 셀이 자기 표 마스킹 풀 안에 반드시 존재함, (c) tablerag 유닛이 잘 형성됨 — 전부
통과. GPU 스모크 테스트(n=3, 6개 arm×method 조합 전부)도 크래시 없이 통과, 필드값 sanity
확인 완료(gold_rank/gold_in_topk/k_actual 등).
