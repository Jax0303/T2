# 다음 세션 시작 프롬프트 (붙여넣기용)

아래 블록을 그대로 복사해서 새 세션 첫 메시지로 붙여넣으면 됩니다.

---

OSC 완전성 연구 이어서 한다. 목표: **엄격한 동일-셀(equal-cell) 예산에서도 dense/hybrid를
유의하게 이기기.** (현재: same-depth+~6셀 주입으로 BM25/dense/hybrid 다 유의하게 이김. 단
엄격 동일-셀에선 BM25만 깨끗이 유의(p=0.014), dense/hybrid는 더 적은 셀로 비김 p≈0.65 borderline.)

## 환경 (중요 — 매 세션 확인)
- 이 환경의 `/usr/bin/python3`엔 numpy 없음. **작동 인터프리터 = `/home/user/T2/hart-table-retrieval/.venv/bin/python`** (numpy/torch/sentence_transformers/scipy 보유).
- 실행: `PYTHONPATH=. /home/user/T2/hart-table-retrieval/.venv/bin/python scripts/...`
- 작업 디렉토리: `/home/user/T2-1/rag-agent`. 모델은 ~/.cache/huggingface 캐시. GROQ_API_KEY는 env에 있음(70b는 rate-limit 심함).
- 모집단 기준: HiTab dev, arithmetic m≥2, n=161, seed 42, LLM-free.

## 지금까지 결과 (커밋됨, ~59ffd99까지)
- `scripts/dense_ceiling_diag.py` → 천장 진단: gold operand 28.5%가 합계행, 유사도 랭킹 39.5위 vs 8위, 천장 실패 76%가 합계행.
- `scripts/osc_total_augment.py` → 합계행 주입. same-depth k=10: BM25 .689→.814, dense .789→.863, hybrid .789→.882 (p≤.0005, 손해 0). 플래그: `--resolver-cols`(cross-encoder 열resolver로 1~2열만, ~6셀 — 권장), 기본값=blind(전 열, ~35셀), `--aug-k/--plain-k`(matched 비교), `--col-targeted`(no-op, 행청크가 전 열 덮음), `--cross-encoder`(기본 MiniLM).
- `scripts/plot_osc_frontier.py` → `docs/fig_osc_frontier.png`.
- 논문: `docs/PAPER_DRAFT.md` §5.1b(천장), §5.10(승리), contribution #4.

## 왜 엄격 예산서 dense/hybrid를 아직 못 이기나 (가설)
resolver-targeted 주입은 셀은 싸지만(~6) 열 resolver가 ~30% 틀려서(col-recall@2≈0.70) 그 열의
합계행을 못 넣어 완전성 일부 손실. 즉 **열 정밀도가 병목.**

## 이번에 시도할 것 (우선순위)
1. **열 resolver 정밀도 ↑** → 엄격 동일-셀서 dense/hybrid 유의 달성이 목표.
   - bge-reranker로 열 resolver 교체 비교(현재 MiniLM). col_select_bench에선 bge가 @2 더 좋았음.
   - top_n_cross=1 vs 2 vs 3 스윕(주입 셀 수 ↔ 완전성 트레이드오프).
   - 후보: 합계행이 있는 열만 후보로 제한(합계행 없는 열은 주입 의미 없음) → 정밀도↑.
2. **엄격 matched-cell paired McNemar 재측정**: aug@k vs plain@k' where cells(aug@k) ≤ cells(plain@k').
   osc_total_augment의 matched_budget_test를 셀 수 맞춰 비교하도록(현재는 k 기준). 셀-매칭 로직 추가.
3. 되면 PAPER §5.10 "엄격 예산서도 dense/hybrid 유의" 로 업그레이드 + figure 갱신.

## 정직성 가드 (잊지 말 것)
- WebFetch 자동요약은 환각함 → 논문 인용 전 원문 확인.
- "열거 단독은 dense보다 낮다, 승리는 증강에서 온다"를 숨기지 말 것.
- 셀 수 항상 같이 보고. strict-superset 승리는 "같은 깊이+오버헤드"임을 명시.

각 단계 결과 보여주고 유의성까지 보고해줘.
