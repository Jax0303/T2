# 다음 세션 시작 프롬프트 (붙여넣기용)

아래 블록을 그대로 복사해서 새 세션 첫 메시지로 붙여넣으면 됩니다.

---

OSC 완전성 연구 이어서 한다.

## ✅ 직전 세션(2026-06-29 밤) 달성 — 엄격 동일-셀 예산 목표 클리어
- **셀-매칭 paired 검정** 추가(`osc_total_augment.py`의 `cell_matched_test`): 쿼리별로 plain에게
  aug와 같은 셀 예산(cells(plain@k')≥cells(aug@10))을 주고 비교. 기존 matched_budget_test는
  plain@20에 셀을 과하게 줘서(98셀 vs 65셀) aug가 진 것처럼 보였던 **오설계**였음.
- **열 resolver를 bge-reranker-base로 교체**(기본값 변경). 셀-매칭에서 **세 검색기 다 유의**:
  BM25 0.876 vs 0.745 (p<1e-4), dense 0.876 vs 0.814 (p=0.002), hybrid 0.901 vs 0.839 (p=0.006).
- top-n-cross 스윕: **top-2가 sweet spot**(top-1은 분자/분모 열 못 덮어 붕괴). `--total-cols-only`는
  top-2에서 no-op(bge가 이미 합계행 보유 열을 집음). 증거 파일: `results/osc_aug_baseline_minilm.json`
  (MiniLM이면 hybrid borderline), `results/osc_aug_bge_t1_tco.json`(top-1 붕괴).
- 갱신됨: `results/osc_total_augment.json`(=bge top2), PAPER §5.10, LAB_MEETING_BRIEF ②③④.

## 다음 목표 (택1)
1. **end-to-end 정확도**: 검색 완전성 win이 실제 답 정확도로 이어지는지(솔버 rate-limit 우회 — 8b 로컬/배치).
2. **figure 갱신**: `plot_osc_frontier.py`에 strict cell-matched 패널 추가(aug@10 vs plain@cell-matched).
3. 다른 벤치(추가 HiTab split / 다른 계층표)로 일반화 확인.

## 환경 (중요 — 매 세션 *직접* 확인할 것)
- **작동 인터프리터 = `/usr/bin/python3`** (검증됨 2026-06-29: numpy 2.4.3 / scipy 1.17.1 / torch 2.10.0 / sentence_transformers / rank_bm25 전부 보유).
- ⚠️ `.venv/bin/python`(`hart-table-retrieval/.venv`)는 deps 없음(`No module named 'sentence_transformers'`) → **쓰지 말 것.** `/home/user/...` 경로는 이 머신에 아예 없음.
- 실행: `PYTHONPATH=. /usr/bin/python3 scripts/...`
- 작업 디렉토리: `/mnt/c/Users/ugh/T2/rag-agent`. 모델은 ~/.cache/huggingface 캐시. GROQ_API_KEY는 env에 있음(70b는 rate-limit 심함).
- 세션 시작 시 확인: `/usr/bin/python3 -c "import numpy,scipy,sentence_transformers,rank_bm25;print('OK')"`
- 모집단 기준: HiTab dev, arithmetic m≥2, n=161, seed 42, LLM-free.

## 결과·스크립트 요약
- `scripts/dense_ceiling_diag.py` → 천장 진단: gold operand 28.5%가 합계행, 유사도 랭킹 39.5위 vs 8위, 천장 실패 76%가 합계행.
- `scripts/osc_total_augment.py` → 합계행 주입. same-depth k=10(bge-reranker resolver): BM25 .689→.876, dense .789→.876, hybrid .789→.901 (p<1e-3, 손해 0, 주입 ~7셀).
  - 플래그: `--resolver-cols`(cross-encoder 열resolver로 2열만 — 권장), 기본값=blind(전 열, ~35셀),
    `--cross-encoder`(**기본 BAAI/bge-reranker-base**, 검증된 최적), `--top-n-cross`(기본 2; 1은 붕괴),
    `--total-cols-only`(합계행 보유 열로 후보 제한 — top-2에선 no-op), `--aug-k/--plain-k`(k-기준 matched, 레거시).
  - 검정 3종이 json에 다 들어감: `same_depth_test`(k=10), `cell_matched_test`(**엄격, 셀-매칭 — 헤드라인**),
    `matched_budget_test`(k-기준, 레거시·오설계였음).
- `scripts/plot_osc_frontier.py` → `docs/fig_osc_frontier.png` (아직 strict 패널 없음 — 추가 TODO).
- 논문: `docs/PAPER_DRAFT.md` §5.1b(천장), §5.10(승리 + 엄격 셀-매칭 표), contribution #4.

## 정직성 가드 (잊지 말 것)
- WebFetch 자동요약은 환각함 → 논문 인용 전 원문 확인.
- "열거 단독은 dense보다 낮다, 승리는 증강에서 온다"를 숨기지 말 것.
- 셀 수 항상 같이 보고. strict-superset 승리는 "같은 깊이+오버헤드"임을 명시.

각 단계 결과 보여주고 유의성까지 보고해줘.
