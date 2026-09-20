# 완전 계층구조 복원 평가 (FULL_STRUCTURE_EVAL)

gold nhr/nhc는 채점에만 사용, 복원 입력(경계 추정/경로 재구성)에는 전혀 사용하지 않음. 경계 추정은 이미 구현·검증된 `rag_agent.reconstruct.guess_n_header_rows/guess_n_header_cols`를 재사용(새 휴리스틱 없음). 원본 파일(rag_agent/, scripts/{retrieval_accuracy,bottleneck_diagnosis,bottleneck_root_cause}.py) 미수정 — 신규 스크립트 `scripts/full_structure_eval.py`만 추가.

## 경계 부트스트랩 절차 (gold 미사용)

```
nhc0=guess_cols(nhr=1); nhr=guess_rows(nhc=nhc0); nhc=guess_cols(nhr=nhr); gold never used as input
```

## 1) header boundary (nhr/nhc), fully-predicted-path만 해당

- 대상 table 수: 538 / 538 (제외 0: {})
- nhr exact accuracy: 0.7007, MAE=0.3662
- nhc exact accuracy: 0.9257, MAE=0.1152
- gold-path/known-boundary 조건은 nhr/nhc가 gold이므로 이 항목이 정의상 100%/0 — 비교 대상 아님(fully-predicted-path만 해당하는 항목)

## 2) 비교표 — path EM / F1 / pair F1 / joint EM

| condition | row EM | col EM | row F1 | col F1 | row pair F1 | col pair F1 | joint EM(gold cell, query count=991) |
|---|---|---|---|---|---|---|---|
| gold-path | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 (991/991, trivial: paths ARE gold by construction) |
| known-boundary predicted-path | 0.6905 | 0.9442 | 0.9084 | 0.9857 | 0.8593 | 0.9784 | 0.5237 (519/991) |
| fully-predicted-path | 0.7669 | 0.888 | 0.927 | 0.9628 | 0.8753 | 0.9487 | 0.6852 (679/991) |

## 3) Hybrid retrieval (R@1/5/20, MRR, ESM), test query count=991

| condition | R@1 | R@5 | R@20 | MRR | ESM |
|---|---|---|---|---|---|
| gold-path | 0.5752 | 0.7982 | 0.9142 | 0.6773 | 0.5752 |
| known-boundary predicted-path | 0.553 | 0.7891 | 0.9082 | 0.6644 | 0.553 |
| fully-predicted-path | 0.5631 | 0.778 | 0.9082 | 0.6641 | 0.5631 |

fully-predicted-path fallback-to-value-only 셀 수: 936 / 67664

## 한계

- 경계 부트스트랩은 1라운드 상호 보정만 수행(2회 이상 반복 시 소폭 달라질 수 있음, 미검증).
- `guess_n_header_cols`는 header_grid.py 자체 문서 기준 test 91.8% exact(다른 실험/데이터 경로, 2026-08-26 측정) — 이 표의 nhc 정확도는 본 실험(split corpus 538 tables, retrieval용 정의)에서 다시 측정한 것으로 그 수치와 다를 수 있음.
- tree_reconstruct_hitab_raw.py 모듈 docstring은 "n_header_cols용 guesser 없음"이라고 적혀 있으나 실제로는 guess_n_header_cols가 이미 구현되어 있음(문서가 stale) — 원본 미수정 원칙에 따라 docstring은 고치지 않음, 여기 기록만 남김.
- row_map/col_map(어느 raw 줄이 데이터 셀인지)은 gold hmt 트리에서 유도된 매핑을 그대로 사용 (gold-path/known-boundary 조건과 동일 전제) — 이는 세 조건 모두가 이미 공유하는 전제이며, nhr/nhc 자체를 gold에서 가져오는 것과는 다르다.
