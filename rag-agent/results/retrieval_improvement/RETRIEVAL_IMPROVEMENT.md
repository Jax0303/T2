# 검색 병목 검증 및 개선 (RETRIEVAL_IMPROVEMENT)

원본 파일/모델/기존 랭킹 미변경. 기존 코드(retrieval_accuracy.build_corpus/cell_unit, bottleneck_root_cause.encode_corpus 캐시, bottleneck_diagnosis population/error 분류기, answer_accuracy reader 유틸)를 재사용. dev(1,671건, table 540)로만 가중치 선택, test(991)에 1회 적용. gold는 채점/oracle에만 사용.

## 1) dense-only / sparse-only / hybrid (test, query count=991)

| method | R@1 | R@5 | R@20 | MRR | ESM | top1 tie 수 |
|---|---|---|---|---|---|---|
| dense | 0.5156 | 0.7699 | 0.9152 | 0.6349 | 0.5156 | 14 |
| sparse | 0.4803 | 0.6731 | 0.7851 | 0.5692 | 0.4803 | 457 |
| hybrid | 0.5782 | 0.7982 | 0.9142 | 0.6789 | 0.5782 | 13 |

- dense top1 정답 → hybrid top1 오답(역전): 32건
- hybrid top1 정답 → dense top1 오답(역전 반대): 94건
- 둘 다 top1 정답: 479건
- gold가 corpus에 없어 제외: 0건
- 정규화/절단 확인: _minmax per query over the full candidate set, matching the deployed hybrid formula exactly; none in this recomputation -- full-corpus rank used (n_units=67664), unlike the deployed pipeline's top-500 scan cap
- unknown rank 6건(배포 파이프라인의 top-500 스캔 캡 때문에 발생, results/pre_improvement_audit/integrity_audit.md 참조): 이 표는 전체 코퍼스 기준 정확한 순위를 다시 계산하므로 6건 모두 확정 순위를 갖고, 별도 unknown 버킷이 없다 — >20으로 합치지 않았다는 뜻이지 사라졌다는 뜻이 아니다.

dev alpha 민감도(참고용, baseline alpha=0.7 유지):

| alpha | dev R@1 |
|---|---|
| 0.1 | 0.4774 |
| 0.3 | 0.5207 |
| 0.5 | 0.5546 |
| 0.7 | 0.5753 |
| 0.9 | 0.564 |

## 2) table-first retrieval + global-top-K 테이블 재랭크

dev table recall@k:

- k=1: 0.7844
- k=3: 0.8983
- k=5: 0.9473

dev에서 선택된 routing k_tables=3, top-50 재랭크 wt=0.2

test:

- table-routing k=1: R@1=0.4985 R@5=0.6811 MRR=0.5817 (gold table 라우팅 탈락 208건)
- table-routing k=3: R@1=0.5499 R@5=0.7538 MRR=0.644 (gold table 라우팅 탈락 90건)
- global top-50+table 재랭크: R@1=0.5681 R@5=0.7921 MRR=0.6715

## 3) 가중 재랭크 (table/row/column), dev 선택 가중치 → test 1회 적용

**주의**: 아래 표의 `baseline`은 top-50 pool 안에서 dense/sparse를 다시 min-max 정규화한 값이다("필드 제외 후 정규화" 규칙을 baseline에도 동일 적용). 코퍼스 전체 기준 정규화인 1)의 hybrid(test R@1=0.5782)와는 기준이 달라 직접 비교가 아니라, table/row/column을 더했을 때 pool 내부 baseline 대비 어느 만큼 움직이는지를 보는 표다.

dev 최적 단일 가중치: {'table': 0.2, 'row': 0.2, 'col': 0.1}, all 가중치(3분할)=0.2, routing k(all)=3

| variant | test R@1 | R@5 | MRR |
|---|---|---|---|
| baseline | 0.5691 | 0.8063 | 0.6771 |
| table | 0.5631 | 0.8002 | 0.6692 |
| row | 0.5732 | 0.7972 | 0.6765 |
| column | 0.5651 | 0.8063 | 0.6736 |
| all | 0.5772 | 0.8002 | 0.6788 |
| table_routing_plus_all | 0.552 | 0.7568 | 0.6461 |

**test 최고 variant: all** (pool-baseline 대비 +0.0081, 코퍼스 전체 기준 원본 hybrid 대비 -0.001 — dev에서 선택된 조합이 test에서 원본 hybrid를 뚜렷이 넘어서지는 못함)

## 4) 동일 직렬화 충돌(943그룹/1886셀)이 primary R@1에 미치는 영향

- 충돌 그룹에 속한 gold: query count=17, R@1=0.1765, 평균 rank(known)=5.18
- 충돌 없는 gold: query count=974, R@1=0.5821, 평균 rank(known)=9.05

## 5) 최고 reranker top1-only reader vs 기존 top-20 reader (공통 150 QA)

- 사용 variant: all
- top1-only QA 정확도: 0.58 (평균 입력 토큰 160.5)
- 기존 top-20 QA 정확도(동일 150 샘플): 0.64

## 판정

- hybrid 역전: 있음 (32건 파괴, 94건 복구 — 순효과는 양)
- table routing: 역효과 (routing으로 gold table 자체가 탈락하는 손실 > 이득)
- row/column 재랭크: 미미/불확실 (best=all, pool-baseline 대비는 개선이나 원본 hybrid 대비 -0.001)
- 직렬화 충돌 영향: 확인됨 (충돌군 R@1=0.1765 vs 비충돌군 0.5821)
- single-vector(top1-only) reader 병목: top-20 유지가 우세 (top1-only가 더 낮음) (0.58 vs 0.64)
- **최고 방법(test R@1 기준)**: all (원본 hybrid를 뚜렷이 능가하지 못함 — 사실상 배포된 hybrid가 여전히 최선)

## 변경 파일 / 명령 / 산출물

- 신규 스크립트: `scripts/retrieval_improvement.py`
- 신규 캐시: `.cache/retrieval_accuracy_queries/` (query 임베딩, 기존 doc 임베딩 캐시 `.cache/retrieval_accuracy/`는 그대로 재사용)
- 산출물: `results/retrieval_improvement/` (bundle_*, item1~5, 본 보고서)
- 원본 파일(rag_agent/, scripts/retrieval_accuracy.py 등) 미수정

```
PYTHONPATH=. .venv/bin/python scripts/retrieval_improvement.py bundle --split test
PYTHONPATH=. .venv/bin/python scripts/retrieval_improvement.py bundle --split dev
PYTHONPATH=. .venv/bin/python scripts/retrieval_improvement.py item1
PYTHONPATH=. .venv/bin/python scripts/retrieval_improvement.py item1 --split dev
PYTHONPATH=. .venv/bin/python scripts/retrieval_improvement.py alpha-sweep
PYTHONPATH=. .venv/bin/python scripts/retrieval_improvement.py item2
PYTHONPATH=. .venv/bin/python scripts/retrieval_improvement.py item3
PYTHONPATH=. .venv/bin/python scripts/retrieval_improvement.py item4
PYTHONPATH=. .venv/bin/python scripts/retrieval_improvement.py item5
```

## 남은 오류 (요약)

- hybrid top1 오답 유형 분포: {'wrong_table': 102, 'wrong_column': 143, 'wrong_row': 113, 'same_leaf_header': 15, 'nearby_cell': 11, 'same_value': 19, 'other': 15}

---

## Addendum — 후속 검증 (사용자 지적 6개 항목, 2026-09-16)

판단: 6개 모두 타당한 지적. 1번(직렬화 충돌 제거)만 실제 배포 파이프라인/캐시를 무효화하는 전면 수정 대신, 원인을 먼저 실증 분석하는 격리된 확인으로 대체(아래).

### 1. 직렬화 충돌 근본 원인 (제거 가능성 실증)

- 충돌 그룹 943개 중 cross-table(서로 다른 table_id) 823건, within-table(같은 table_id 안에서 행/열만 다름) 120건
- **cross-table (87%)**: 서로 다른 문서의 표가 동일 제목·동일 값을 가진 진짜 (근사)중복 데이터인 경우가 다수 확인됨(예: 같은 선수의 시즌 통계표가 두 table_id로 중복 수록). table_id를 직렬화에 넣으면 기술적으로는 구별되지만 이는 정답을 암기시키는 것과 같아 '의미 있는' 필드가 아님 — 이 다수 구간은 직렬화로 해결 대상이 아니라 6항목(단일 gold 평가 한계)과 같은 부류로 재분류하는 것이 타당함.
- **within-table (13%)**: 같은 표 안에서 서로 다른 행/열이 동일 header path 문자열로 축약되는 경우. 더 깊은 header 계층이 실제로 존재하는지는 테이블별 hmt raw tree 재조사가 필요한 별도 작업 — 여기서는 추측 구현하지 않음(존재 여부 확인 전에는 어떤 필드를 추가할지 결정할 수 없음).

### 2. 정규화 불일치 수정 — item3를 pool 재정규화 없이 재실행

`combined_scores(preserve=True)`: dense/sparse는 재정규화하지 않고 item1과 동일한 코퍼스 전체 정규화 hybrid를 그대로 사용. baseline이 이제 item1의 hybrid(R@1=0.5782)와 거의 일치(0.5752).

| variant | test R@1 (원래, pool 재정규화) | test R@1 (수정, 정규화 보존) |
|---|---|---|
| baseline | 0.5691 | 0.5752 |
| table | 0.5631 | 0.5681 |
| row | 0.5732 | 0.5792 |
| column | 0.5651 | 0.558 |
| all | 0.5772 | 0.5621 |
| table_routing_plus_all | 0.552 | — |

**수정 후 최고 variant: row** (원래 결론이었던 `all`이 수정 후에는 baseline보다 낮아짐 — 정규화 방식이 결론 자체를 바꿈, 지적이 정확했음을 확인)

### 3. 150-QA 표본 신뢰구간 / 대응표본 검정

- top1-only: 0.58 (95% CI [0.5, 0.656])
- 기존 top-20: 0.64 (95% CI [0.5606, 0.7124])
- McNemar (불일치쌍 top1오답/top20정답=24, top1정답/top20오답=15), exact p=0.1996
- **결론: query count=150 표본으로는 통계적으로 유의하다고 말하기 어려움** — 58% vs 64% 차이를 query count=150에서 통계적으로 확정할 수 없음

### 4. cross-table hard negative의 캡션 부족 (wrong_table=102건)

- 제목이 완전히 동일: 5/102
- 제목 자카드 유사도 ≥0.5: 19/102, 평균 자카드=0.2563
- **결론**: 대부분(약 83/102)은 이미 제목이 뚜렷이 다름 → '캡션이 표를 구별 못 해서'라는 가설은 부분적으로만 맞음(자카드 높은 19건에서는 유효), 다수는 캡션과 무관한 dense 의미 혼동

### 5. 숫자·짧은 헤더 embedding 한계

- wrong_column 중 짧은/숫자형 leaf 비율: 0.3427 (전체 기준선 0.3047) — 소폭 과대표집
- wrong_row 중 짧은/숫자형 leaf 비율: 0.1504 (전체 기준선 0.1695) — 오히려 낮음
- **결론**: column 오류에서는 약하게 지지되나 row 오류에서는 반대 방향 — 일관된 강한 근거는 아님

### 6. 단일 gold 평가의 한계 (답변 가능한 셀이 더 있을 수 있음)

- 이미 기존 error_class 분류에 반영되어 있음: hybrid top1 오답 중 same_value(동일 값, 다른 위치) = 19건 — gold 외에도 값이 같은 셀이 top1으로 뽑혀 '오답' 처리된 경우
- 4항목의 cross-table 충돌(823/943 그룹, 위 1항목)도 사실상 같은 성격의 한계
