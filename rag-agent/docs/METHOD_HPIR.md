# METHOD_HPIR.md — Header-Path Intent Resolution

생성: 2026-06-07 · 코드: `rag_agent/query/header_path_resolver.py` ·
테스트: `tests/test_header_path_resolver.py` (11) ·
평가: `scripts/hpir_retrieval_eval.py`, `scripts/method_grounded.py --mode hpir`

---

## 1. 문제 (수치 근거)

분리-스토어 table-RAG(VDB ↔ 원본 헤더트리, 쿼리별 라우팅)의 두 병목은 **곱**으로 작용한다
(`docs/VERDICT.md`, `docs/FAILURE_ANALYSIS.md`):

| 병목 | 측정 | 핵심 원인 |
|---|---|---|
| 검색 | dense_header_path R@1 **0.49** ≫ plain 0.39 | 코퍼스가 *헤더경로*로 직렬화될 때 최고 → 그런데 검색기에 **원문 서술 질의**를 그대로 넣음(헤더 토큰 비율 낮음) |
| 답변 | gold 표를 줘도 NM **0.447**; 답변실패의 **82%**가 bucket⑥(코드 실행됨·값 틀림) | 헤더를 **자유추측**으로 바인딩 → 빗나가도 예외 없이 조용히 틀림 |

**공통 원인 한 줄:** *쿼리가 가리키는 계층 헤더경로가 파이프라인 어디에서도 명시되지 않는다.*

## 2. 방법 — 헤더경로를 두 스토어의 공유 IR로

HPIR는 들어온 질의를 **계층 헤더경로 의도**로 변환하는 단일 쿼리-이해 단계이며, 같은 목적함수를
두 국면에 적용한다.

```
                         ┌──────────────────────────────┐
   query  ────────────▶  │  Header-Path Intent Resolver │
                         └───────┬───────────────┬──────┘
            (corpus-free)        │               │   (table-grounded)
        expand_for_retrieval     │               │  resolve_against_table
                                 ▼               ▼
                    ┌───────────────────┐  ┌──────────────────────────┐
                    │ 헤더경로 정렬       │  │ 검색표의 실제 row/col      │
                    │ 의사문서 → 검색     │  │ 헤더경로에 바인딩 → codegen│
                    │ (R@1 병목 공략)     │  │ 힌트 (조용한 오류 공략)     │
                    └───────────────────┘  └──────────────────────────┘
```

* **Regime 1 — 검색용(코퍼스-프리, `expand_for_retrieval`).** 서술을 제거하고 헤더가 될
  법한 토큰만 남긴다(+연산 힌트). HiTab에서 **연도는 열 헤더**(`current $millions > 2014`)인
  경우가 많아 *유지*한다 — 이는 `VERDICT.md`가 해롭다고 한 "쿼리 숫자 ↔ 전체 셀 매칭"과 다르다
  (여기선 헤더경로 인덱스에 대한 토큰 신호). HyDE(Gao 2023)/query2doc(Wang 2023)의 구조화 변형.
* **Regime 2 — 답변용(표-그라운디드, `resolve_against_table`).** 검색된 표의 *실제* top/left
  헤더경로를 질의에 대해 랭킹하여 (row_path, col_path) 후보를 돌려준다. **반환 바인딩은 표에
  존재함이 보장**되며(스토어의 fuzzy scorer 재사용), 이것이 자유추측 추출기에 없는 신호다.

**노블티(차별점):**
1. 분리-스토어 검색기의 **두 단계(검색·실행)를 잇는 명시적 공유 IR로 "계층 헤더경로"를 도입**.
   기존 query expansion은 검색만, program-grounding(PAL/Self-Debugging)은 실행만 다룬다.
2. **검증된 LLM 정련**(`resolve_intent`): LLM이 헤더를 골라도 **실제 인벤토리와 교집합**만
   채택하고 결정론적 랭킹으로 백필 → LLM이 비존재 바인딩을 절대 주입 못함. self-debugging이
   못 잡는 *예외 없는 grounding 오류*를 사전에 차단.
3. 결정론적 코어(LLM 불필요) → **완전 단위테스트 가능**하고 저비용 베이스라인 제공.

기존문헌 토대(실재): HyDE(arXiv 2212.10496), query2doc(arXiv 2303.07678), DTR(Herzig NAACL 2021),
PAL(arXiv 2211.10435), Self-Debugging(ICLR 2024), LEVER(ICML 2023), HiTab(ACL 2022).

## 3. 수치 검증 프로토콜 (메인)

두 병목을 **각각 격리**해 측정한다(곱 구조이므로 분리 측정이 필수).

### 3.1 검색 — 쿼리 처리 효과 격리
`scripts/hpir_retrieval_eval.py`. 동일 검색기·동일 풀에서 **raw 질의 vs HPIR-확장 질의**만 다르게.

```bash
HITAB_DIR=/path/to/hitab CHROMA_DIR=/path/to/chroma_db \
python scripts/hpir_retrieval_eval.py --device cuda --out results/hpir_retrieval.json
```
- 비교: `dense` vs `dense_hpir`, `keyword` vs `keyword_hpir`.
- 지표: R@1/R@5/MRR/nDCG@10 + **paired bootstrap 95% CI** + **exact McNemar(R@1)**.
- 층화: 난이도 6클래스 per-class R@1.
- 판정선: HPIR-확장이 raw 대비 R@1을 **CI가 0을 제외**하는 폭으로 올리면 검색 기여 성립.

### 3.2 답변 — 바인딩 힌트 효과 격리 (gold 검색 고정)
`scripts/method_grounded.py`. 검색을 gold로 고정해 **답변 단계만** 본다. 3-way:

```bash
python scripts/method_grounded.py --mode naive    --per-class 20   # 하한
python scripts/method_grounded.py --mode grounded --per-class 20   # 스키마+trace+repair
python scripts/method_grounded.py --mode hpir     --per-class 20   # +HPIR 바인딩 힌트
```
- 지표: NM(±2%) overall + per-class(5 hard) + bootstrap CI(`scripts/bootstrap_ci.py`).
- ablation 축: naive → grounded(스키마/trace/repair) → hpir(헤더경로 사전해소) 단조 비교.

### 3.3 기존 측정 기준점 (동일 하니스, N=100, gold, Qwen2.5-7B-4bit, seed=42)
| mode | NM | multi_op | arith_agg | pair_topk | single_arg | cmp_count |
|---|---|---|---|---|---|---|
| naive (`results/method_naive_pc20.json`) | 0.41 | 0.00 | 0.10 | 0.75 | 0.85 | 0.35 |
| grounded (`results/method_grounded_pc20.json`) | 0.48 | 0.20 | 0.25 | 0.75 | 0.70 | 0.50 |
| **hpir** | *측정 예정(로컬)* | | | | | |

> 본 컨테이너에는 HiTab 데이터·GPU·Qwen·Chroma가 없어 수치는 **로컬(RTX 3060 Ti)에서 산출**한다.
> 코드/하니스/단위테스트는 이 저장소에서 완결, 실행만 데이터 보유 머신에서.

## 4. 한계 / 위협 타당성
- HPIR 확장이 연도 토큰을 유지 → 연도 헤더가 없는 표에서는 noise일 수 있음(per-class로 점검).
- gold 검색 격리는 답변 상한을 측정; end-to-end는 검색×답변 곱이라 3.1·3.2 결합 해석 필요.
- 결정론 resolver의 fuzzy threshold(0.4)는 스토어 기본값 재사용 — 민감도는 후속 ablation 대상.

## 5. 재현 자산
- 코어: `rag_agent/query/header_path_resolver.py` (+`__init__.py`)
- 테스트: `tests/test_header_path_resolver.py` — `python -m pytest tests/ -q` (데이터 불필요, 11+19 통과)
- 검색 평가: `scripts/hpir_retrieval_eval.py`
- 답변 평가: `scripts/method_grounded.py --mode {naive,grounded,hpir}`
- 설정: `configs/hpir.yaml`
