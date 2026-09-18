# 병목 원인 진단 (H1/H2/H3)

BOTTLENECK_DIAGNOSIS.md의 후속. ESM 정의는 그대로(단일 gold+고정 k 구조상 k=1에서만 ESM>0). alpha=0.7 등 기존 하이퍼파라미터는 재튜닝하지 않았고, gold는 채점/oracle 조건 외에는 랭킹에 쓰지 않았다.

**gold_rank 미확인 6건** (retrieval_accuracy.py 상위-500 스캔 기준, 별도 처리 — 아래 표의 버킷/평균에 섞지 않음): 4c25190668c859cfffef1f14bd69ffe0, 7c2d4c7010f3d02a4cb5b1da1c4313a4, 89bab78aee8ef8065330506d7128f3f6, 8db22461be5d09fe27d62d3b758fad97, b8b332eb07c52c9c44666293f5ea8e2b, d7848bd4a92822affd40df34e647de7d

## H1 — predicted 구조 복원 오류

- 재구성 대상 표 538개 중 0개 제외 ({})
- row path EM 0.6905 / col path EM 0.9442
- row path-node F1 0.9084 / col path-node F1 0.9857
- row pair P/R/F1 {'precision': 0.862, 'recall': 0.8565, 'f1': 0.8593, 'n_gold_edges': 11797, 'n_rec_edges': 11721}
- col pair P/R/F1 {'precision': 0.9774, 'recall': 0.9793, 'f1': 0.9784, 'n_gold_edges': 6000, 'n_rec_edges': 6012}

structure correct/wrong × predicted_path 검색 성능:

| subset | n | R@1 | R@5 | R@10 | R@20 | MRR | median rank |
|---|---|---|---|---|---|---|---|
| structure correct | 519 | 0.553 | 0.7765 | 0.8728 | 0.9171 | 0.6593 | 1 |
| structure wrong | 472 | 0.553 | 0.803 | 0.8602 | 0.8983 | 0.6699 | 1 |

structure-correct 부분집합에서 gold_path vs predicted_path (재구성이 맞아도 남는 차이가 있는지):

| repr | n | R@1 | R@5 | R@10 | R@20 | MRR | median rank |
|---|---|---|---|---|---|---|---|
| gold_path | 519 | 0.5453 | 0.7803 | 0.869 | 0.9094 | 0.6547 | 1 |
| predicted_path | 519 | 0.553 | 0.7765 | 0.8728 | 0.9171 | 0.6593 | 1 |

## H2 — embedding이 구조를 무시하는가

| representation | R@1 | R@5 | R@10 | R@20 | MRR | median rank | n_units | cache_hit |
|---|---|---|---|---|---|---|---|---|
| value_only | 0.005 | 0.0081 | 0.0091 | 0.0111 | 0.0065 | 31835 | 67664 | False |
| gold_path | 0.5752 | 0.7982 | 0.8698 | 0.9142 | 0.6773 | 1 | 67664 | True |
| predicted_path | 0.553 | 0.7891 | 0.8668 | 0.9082 | 0.6644 | 1 | 67664 | False |

## H3 — 구조적 hard-negative가 리더를 방해하는가

gold-only 기준선 답변 정확도: 0.9933 (n_sample=150)

| condition | n | 답변 정확도 |
|---|---|---|
| gold_1_random_first | 150 | 0.94 |
| gold_1_random_last | 150 | 0.94 |
| gold_1_hard_negative_first | 150 | 0.84 |
| gold_1_hard_negative_last | 150 | 0.76 |

### gold_1_hard_negative_first — distractor 구조 클래스별

| class | n | 답변 정확도 |
|---|---|---|
| nearby_cell | 1 | 1.0 |
| other | 3 | 0.6667 |
| same_leaf_header | 4 | 0.5 |
| same_value | 9 | 0.7778 |
| wrong_column | 64 | 0.9375 |
| wrong_row | 44 | 0.8182 |
| wrong_table | 25 | 0.72 |

### gold_1_hard_negative_first — score margin 구간별 (margin은 distractor가 top-1일 때만 계산됨)

| margin bucket | n | 답변 정확도 |
|---|---|---|
| 0.00-0.05 | 42 | 0.7381 |
| 0.05-0.15 | 18 | 0.7222 |
| 0.15-0.30 | 4 | 0.5 |
| unknown | 86 | 0.9302 |

### gold_1_hard_negative_last — distractor 구조 클래스별

| class | n | 답변 정확도 |
|---|---|---|
| nearby_cell | 1 | 1.0 |
| other | 3 | 0.6667 |
| same_leaf_header | 4 | 0.5 |
| same_value | 9 | 0.8889 |
| wrong_column | 64 | 0.8594 |
| wrong_row | 44 | 0.6818 |
| wrong_table | 25 | 0.64 |

### gold_1_hard_negative_last — score margin 구간별 (margin은 distractor가 top-1일 때만 계산됨)

| margin bucket | n | 답변 정확도 |
|---|---|---|
| 0.00-0.05 | 42 | 0.6667 |
| 0.05-0.15 | 18 | 0.5556 |
| 0.15-0.30 | 4 | 0.5 |
| unknown | 86 | 0.8605 |

## 판정

**H1 (구조 복원 오류): Partial** — path EM 0.6905 — 일부 재구성 오류가 존재하나 성공/실패군 recall 차이는 0.0188

**H2 (임베딩이 구조 무시): Rejected** — gold_path recall@20 - value_only recall@20 = 0.9031 — 임베딩이 구조를 명확히 반영함

**H3 (구조적 hard-negative 방해): Partial** — random distractor 대비 hard-negative 정확도 0.1 낮음; 구조적 클래스(wrong_row/column 등) 평균 0.8139 vs 그 외 0.7215

## 한계

- predicted structure는 헤더 블록 크기(nhr/nhc)는 gold를 쓰고 경로 내용만 재구성함 (tree_reconstruct_hitab_raw.py의 known-boundary 모드와 동일 조건).
- H3 margin은 hard-negative distractor가 해당 질의의 top-1 검색 결과와 정확히 같을 때만 계산됨 — 그 외는 'unknown' 버킷.
- controlled QA는 n=150 표본(2026-09-16 seed=42 고정, 재실행 없음)에 대한 재분석이며 새 리더 호출은 없음.

## 산출물

- `results/bottleneck_root_cause/structure_metrics.json`
- `results/bottleneck_root_cause/structure_decomposition.json`
- `results/bottleneck_root_cause/retrieval_{value_only,gold_path,predicted_path}.json(l)`
- `results/bottleneck_root_cause/top1_error_detail.csv`
- `results/bottleneck_root_cause/hardneg_analysis.json`
