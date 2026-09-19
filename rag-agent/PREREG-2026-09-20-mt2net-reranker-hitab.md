# 사전등록: HiTab에서 MT2Net의 실제 채점 메커니즘(학습된 BERT pairwise 분류기) 재현 (2026-09-20)

## 왜

`results/SUMMARY_TABLES-2026-09-18.md` §1/§3의 "mt2net" 행은 오늘 세션에서 라벨을 정정했다 —
실제로는 ours와 동일한 임베딩+하이브리드 검색기(α=0.7, `--corpus gold`)를 쓰고 셀→문장
템플릿만 논문 예시 1개에서 역추정해(미검증, 재현 실패 확인됨) 바꿔 끼운 것이지, MT2Net
(Zhao et al. 2022, arXiv:2206.01347)의 핵심 메커니즘(질문+후보 문장 concat → BERT 이진
분류기로 점수, top-n 유지)을 재현한 것이 아니었다.

**정정만으로는 부족하다** — 이 실험은 실제로 학습된 분류기를 만들어 정직한 두 번째
"mt2net" 행을 추가한다. 코퍼스 전체(58,759셀)에 매 질의마다 BERT를 돌리는 것은
`CLAUDE.md §2`에 이미 계산상 불가능하다고 적혀 있고, HiTab은 표 540개 중 537개가
"출처 문서당 표 1개"라 MT2Net이 전제하는 "문서 범위로 좁히기"가 원래 형태로는 의미가
없다(오늘 세션에서 직접 확인). 그런데 `results/SUMMARY_TABLES-2026-09-18.md` §1/§3의
비교표는 이미 **모든 arm을 `--corpus gold`(질의 자신의 표 안에서만 검색)로 재고 있다**
(`PREREG-2026-09-19-selector-gold-fairness.md`가 이미 이렇게 기록해둠, 오늘 세션에서
`t_mt2net_gold_records.jsonl`을 n=300 pilot population으로 필터링해 재확인: 0.9433 ≈
표의 0.9500, 오차 2건). 즉 §1/§3의 "표 1개 안에서 찾기"라는 스코프 자체가 이미 MT2Net의
전제(범위를 좁혀서 찾는다)를 만족한다 — HiTab에 "문서" 개념이 없다는 문제와 무관하게,
이 표의 프로토콜 안에서는 재현이 성립한다.

## 개입

`scripts/train_mt2net_reranker_hitab.py`(신규, `git show 575ff88~1:rag-agent/scripts/train_cell_reranker.py`의
학습 루프를 그대로 재사용하되 negative 채굴을 파인튜닝 retriever(삭제됨, §0.3) 대신
기성품 하이브리드(`default_encoder()` + BM25, α=0.7, `--corpus gold`와 동일한 표 단위
마스킹)로 교체):

- 기반 모델: `BAAI/bge-reranker-base` (미세조정 전 기성품 cross-encoder, MT2Net 자체와
  같은 "BERT류 pairwise 분류기" 계열).
- 학습 데이터: HiTab **train** split, gold 셀이 있는 질의 전부. 표 단위로 eval 모집단과
  절대 안 겹침(자동 확인: HiTab 공식 train/test 분할은 이미 표 단위로 분리돼 있음 —
  train 표 2,519개, eval 표 198개, 교집합 0건, 스크립트 안 assert로 재확인).
- 문장 템플릿: `mt2net`(기존과 동일, 미검증 캐비엇도 동일하게 적용됨 — 이 실험이 고치는
  것은 채점 메커니즘이지 문장 템플릿의 논문 충실도가 아니다).
- negative: 질의당 자기 표 안에서 하이브리드 검색기가 상위로 올린 오답 셀 상위 7개
  (`--mine-topk 20`에서 gold 제외 상위 7, 기존 스크립트와 동일한 하이퍼파라미터).
- 손실: `BinaryCrossEntropyLoss`, 1 epoch, batch 16, lr 2e-5 — 기존 스크립트 기본값 그대로,
  이번 실험 목적(문장 형태가 아니라 채점 메커니즘 비교)에 비춰 재튜닝하지 않는다.

`scripts/retrieval_accuracy.py`에 `--rerank-model PATH` 옵션 추가: `--corpus gold`
필수(그 외 값이면 에러 — 코퍼스 전체 스캔은 계산상 배제된 것으로 이미 결론남). 주어지면
질의당 후보 점수 `s`를 하이브리드 임베딩 대신 이 cross-encoder의 (질문, 셀 문장) 점수로
전량 교체한다(혼합 아님 — MT2Net 자체가 분류기 점수 하나로 순위를 매기므로). 이후
`budget_select` 등 기존 선택·채점·출력 경로는 완전히 그대로 재사용한다(회귀 위험을
최소화하려고 새 채점 스크립트를 따로 만들지 않음).

## 모집단

`results/ksweep_population_300.json`의 고정 300-쿼리(HiTab primary population, mode=all,
m=1, aggregation=none 991건 중 stratified_sample seed=42) — §1/§3과 동일 모집단.
`--corpus gold`, `--budget 20`, `--split test`.

## 예측 (실행 전 고정)

1. **학습된 재정렬기가 mt2net식 템플릿(임베딩만, 정정된 행, 검색정확도 .9500)보다
   유의하게 낫거나 최소한 비슷할 것으로 예측한다** — 학습된 분류기는 같은 도메인
   negative로 지도학습됐으므로 기성품 임베딩보다 표 안에서 정답 셀을 더 잘 가려낼
   개연성이 높다. 다만 학습 데이터가 크지 않아(train split 7,417건 중 gold 있는
   질의만) 극적인 차이는 아닐 것으로 본다.
2. **ours(cell, 검색정확도 .9667)를 넘어서지는 못할 것으로 예측한다** — ours의 우위는
   `CLAUDE.md §6`에 이미 정리된 "고유 라벨"에서 오고, mt2net 계열 문장(고유 라벨 없이
   헤더+값만)은 그 구조적 이점이 없다. 채점기를 아무리 잘 학습시켜도 문장 자체가 표를
   구별 못 하면 표 안에서는 의미가 없지만(이미 gold 표 안으로 좁혀서 찾으므로 표 구별은
   문제가 안 됨) — 오히려 이 설정에서는 ours와 격차가 좁혀질 수 있다고 본다. 방향을
   미리 걸지 않고 결과 그대로 적는다.
3. 어느 예측이든 벗어나면 그대로 적고 표를 그대로 싣는다.

## 산출물

`results/mt2net_reranker_hitab/` — `eval_tables.json`(학습 제외 표 목록, 이미 생성),
학습된 모델(`models/mt2net-reranker-hitab/`), 학습 메타(`train_meta.json`), 채점 결과
(`retrieval_accuracy.py --rerank-model ... --corpus gold ...` 표준 출력 4종).
`results/SUMMARY_TABLES-2026-09-18.md` §1/§3에 새 행 추가(기존 행 보존).
