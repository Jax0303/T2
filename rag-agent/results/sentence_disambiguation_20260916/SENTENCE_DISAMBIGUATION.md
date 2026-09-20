# 문장 축-구분 개입 (structural_leaf) — 2026-09-16

모집단: hitab test primary (mode=all, m=1, aggregation=none), query count=991. encoder/hybrid(alpha=0.7)/corpus(split)는 baseline과 동일 — 바뀐 변수는 cell 문장 템플릿 하나(s3c -> structural_leaf, row/col leaf 접두)뿐.

## R@1/5/10/20/MRR — baseline vs structural_leaf

| repr | R@1 | R@5 | R@10 | R@20 | MRR | median rank |
|---|---|---|---|---|---|---|
| gold_path (s3c, baseline) | 0.5752 | 0.7982 | 0.8698 | 0.9142 | 0.6773 | 1 |
| predicted_path (s3c, baseline) | 0.553 | 0.7891 | 0.8668 | 0.9082 | 0.6644 | 1 |
| structural_leaf (gold path content, new) | 0.6176 | 0.8032 | 0.8789 | 0.9213 | 0.7049 | 1 |

## top-1 오류 클래스 — baseline(gold_path, s3c) vs structural_leaf

| class | baseline n | baseline % | structural_leaf n | structural_leaf % |
|---|---|---|---|---|
| same_value | 22 | 0.0523 | 20 | 0.0528 |
| wrong_row | 113 | 0.2684 | 91 | 0.2401 |
| wrong_column | 143 | 0.3397 | 126 | 0.3325 |
| same_leaf_header | 15 | 0.0356 | 18 | 0.0475 |
| nearby_cell | 11 | 0.0261 | 6 | 0.0158 |
| wrong_table | 102 | 0.2423 | 98 | 0.2586 |
| other | 15 | 0.0356 | 20 | 0.0528 |
| **total errors (non-exact-match)** | 421 | 1.0 | 379 | 1.0 |

집계 전/후 비교일 뿐, 문항 단위 매칭(McNemar류)은 하지 않았다 — wrong_row/column 건수 감소가 같은
문항이 exact_match로 넘어갔다는 보장이지, 클래스별 페어링을 검증한 것은 아니다.

## 개입 2 (title 강화) — 실행 전에 기각

`results/bottleneck_root_cause/top1_error_detail.csv`의 wrong_table 102건 중, gold 표 제목과
top1 표 제목이 실제로 같은 문자열인 경우는 **5건(4.9%)**뿐 (91건은 제목이 다름, 6건은 제목 조회
실패). 이 저장소에 이미 이 축의 선례가 있다: `results/tableconf/VERDICT.md` (2026-09-02, dev,
title-collision 제목 겹침 문제)는 `with_page_title`(ToTTo 페이지 제목 접두)로 고쳤고, 그 수정이
지금 배포 코퍼스(build_repr_corpus)에 이미 적용돼 있다. 현재 wrong_table의 대다수(89.2%)는
제목이 이미 서로 다른데도 틀리므로, 문장에 제목 신호를 더 넣어도 고칠 정보가 없다 — no-op.
구현하지 않음.

## 답변 정확도 (Qwen3-8B 4bit, k=1, structural_leaf corpus)

population/reader/prompt/scorer 는 K_LADDER_TABLES-2026-09-16.md 의 k=1 레그와 동일
(`--reader local:Qwen/Qwen3-8B?quantization=4bit --prompt neutral --max-tokens 64 --primary-only`),
문맥만 structural_leaf 코퍼스의 top-1으로 교체.

| 조건 | 답변 정확도 | 검색 성공 시 | 검색 실패 시 |
|---|---|---|---|
| k=1 baseline (s3c) | .5923 | .9877 (query count=570) | .0570 (query count=421) |
| k=1 structural_leaf | **.6307** | .9886 (query count=612) | .0528 (query count=379) |

+.0384 (baseline 대비). 0.8 목표까지 남은 거리: **.1693**. 검색 성공/실패 시 조건부 정확도는
거의 그대로(.99/.05대) — 개입은 R@1(=성공 비율)만 올렸고, 리더의 조건부 성능에는 영향이 없다.
산술적으로 이게 최선이다: 성공시 정확도가 이미 .99 근처 천장이라, 목표 도달에 필요한 건 R@1
자체를 .8 근처까지 올리는 것뿐인데 이번 개입의 R@1 기여는 +.0424(.5752→.6176)뿐이었다.

## 결론

두 개입 중 하나만 실행(축-구분 접두), 하나는 실행 전 진단으로 기각(제목 강화, 정보 없음 확인).
R@1 .5752→.6176(+.0424), 답변 정확도 .5923→.6307(+.0384) — 방향은 가설대로(축-혼동 오류가
줄었다: wrong_row 113→91, wrong_column 143→126) 이나 크기가 작다. 0.8/0.9 목표에는 한참
못 미친다 — "문장 생성 프레임 유지 + no training" 제약 안에서 이 레버가 낼 수 있는 크기가
애초 예상(수 %p)대로였다는 뜻이지, 개입이 잘못됐다는 뜻은 아니다.
