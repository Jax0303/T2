# 사전등록: MultiHiertt에서 MT2Net 학습된 문서범위 재정렬기 (2026-09-20)

## 배경

HiTab에서 이미 완료(`PREREG-2026-09-20-mt2net-reranker-hitab.md`): 학습된 BERT pairwise
재정렬기가 임베딩-only mt2net 템플릿보다 유의하게 더 나빴다(McNemar p≈0, n=1581). MultiHiertt는
HiTab과 달리 문서 하나에 표 여러 개가 실제로 들어있어(`mh_arms.py`의 `uid`가 문서, `scope="doc"`가
이미 그 범위로 후보를 제한함) MT2Net의 "문서 범위로 좁혀서 찾는다"는 전제가 여기서는 실제로 성립한다.

`results/SUMMARY_TABLES-2026-09-18.md` §2의 `mt2net_desc` 행은 문장은 진짜(데이터셋 원문
`table_description`)이지만 검색기는 학습 없는 임베딩+BM25 하이브리드다(오늘 세션에서 "학습된
베이스라인" 오표기를 지우고 각주②로 캐비엇 달아둠). 이번 작업은 같은 문장으로, 실제로 학습된
분류기를 문서 범위 재정렬에 얹는다.

## 개입

- 학습: `BAAI/bge-reranker-base`를 파인튜닝(`train_mt2net_reranker_hitab.py`와 동일한 학습
  루프 재사용). negative는 질의 자신의 문서(uid) 안에서 기성품 하이브리드(default_encoder +
  BM25, α=0.7)가 상위에 올린 오답 후보로 채굴.
- 학습 데이터 분할: **MultiHiertt `validation` split 전체**(338질의/338문서, table-only)를
  학습에 쓴다. 평가는 §2가 이미 쓰는 **`train` split**(2908건 population)에 그대로 — 두
  분할의 문서 집합은 실측 **완전 분리**(교집합 0, 코드로 확인 완료).
  - 캐비엇: HiTab(7,388질의/2,517표)보다 학습 질의 수가 훨씬 작다(338질의) —
    MultiHiertt의 공식 validation split 크기가 원래 작아서다. 이 데이터량 제약은 결과
    해석에 남긴다.
- 평가: `mh_arms.py`에 `--rerank-model` 옵션 추가(HiTab의 `retrieval_accuracy.py
  --rerank-model`과 동일한 패턴) — `scope="doc"`(uid 범위)로 제한한 후보를 하이브리드
  점수 대신 학습된 재정렬기 점수로만 순위 매김. corpus/table 범위는 재정렬기 적용 대상
  아님(코퍼스 전체 스캔은 계산상 불가능, HiTab과 동일 이유).
- §2와 동일 조건: `--split train --unit mt2net_desc --budget 20`, 4개 지표(단일조회/
  다중조회/단일산술/다중산술) 그대로.

## 예측 (실행 전 고정)

- **HiTab 전례를 고려해, 이번에도 학습된 재정렬기가 mt2net_desc(임베딩만)보다 나쁘거나
  최소 비슷할 것으로 예측한다** — HiTab에서 나쁜 원인이 태스크 자체(짧고 정형화된 mt2net
  스타일 문장에 대한 `bge-reranker-base`의 사전학습 편향, 혹은 1 epoch 과소학습)라면
  같은 원인이 여기서도 작동해야 한다. 다만 이번엔 학습 데이터가 훨씬 적어(338 vs 7,388
  질의) 과소학습이 더 심할 수 있어 격차가 HiTab보다 클 것으로 예측한다.
- **mt2net_desc가 이미 ours를 앞서는 상황(§2, 4개 지표 전부 근소 우위)이므로, 학습된
  재정렬기가 mt2net_desc보다 나쁘더라도 ours보다는 나을 가능성이 있다** — 이 경우
  "학습을 더 해도 ours를 못 이긴다"가 아니라 "학습이 임베딩 검색보다는 낫지만 원문
  문장 자체(mt2net_desc)의 이점을 재정렬이 깎아먹는다"는 다른 해석이 필요함을 미리 적어둔다.
- 어느 쪽이든 결과를 보고 규칙을 고르지 않는다 — 나온 그대로 표에 싣는다.

## 고정

- 인코더/α: HiTab과 동일(`BAAI/bge-base-en-v1.5`, α=0.7). 재정렬 베이스: `BAAI/bge-reranker-base`,
  1 epoch, negative 7개/질의, lr 2e-5, seed 42 — HiTab과 하이퍼파라미터 전부 동일(사후 조정 없음).
- 채점: `mh_arms.py`의 기존 accuracy 로직(`q["gold"] <= got`) 그대로.
- 커밋 기준: `bee2e24`(HiTab 결과 커밋) 다음 작업, 미커밋 상태에서 시작.
