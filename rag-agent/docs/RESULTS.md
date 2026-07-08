# Operand-Set Completeness Retrieval — Results (W7)

Consolidated results for the operand-set completeness (OSC) study on hierarchical
tables. All numbers measured on HiTab `dev`, seed=42, paired bootstrap 95% CI.
Retrieval experiments (E1, E2, W4b) are LLM-free except W4b's decomposition step.

> Status: E1 (H1) ✅ · E2 (H2) ✅ · W4b (LLM decomposition lever, 8b+70b) ✅ ·
> E3 (synthetic depth) ✅ · E4 (generation format) ✅ · E6 (structural scope
> selection: row-failure diagnosis + total/sibling treatments) ✅ ·
> §5.1b ceiling diagnosis ✅ · §5.10 injection case study ✅ · E8/E9 (vs OHD) ✅ ·
> §5.11 generalization scope (FinQA/WikiSQL) ✅ · H6 answer-accuracy payoff 🔶 (partial sample).

---

## ⚠ 2026-07-08 정합성 노트 — 어떤 수치가 지금 유효한가

2026-07-07 전수 코드 감사(8개 버그 수정 — m=1 gold-offset 오염, 행축 내부노드 헤더 소실 등,
`e4390c1`/`394ba9b`)와 gold 재빌드로 **아래 E1–E6 역사 섹션의 수치는 감사 이전 측정치(당시
n=158)** 가 됐다. 실험 서사(가설→진단→치료)와 결론의 방향은 유지되지만, 개별 숫자를 밖에
인용할 땐 이 스냅샷 또는 `PAPER_DRAFT.md` §5를 쓸 것. E2 열거-단독 수치는 아직 재측정 전
(**재검증 필요** — `e2_osc_enum.py --split dev` 재실행 예정).

**지표 정책 (2026-07-07부터):** 타 논문과 비교하는 답변 정확도는 반드시
**`hitab_exact_match`** (HiTab 공식 채점기 포팅, `rag_agent.eval.metrics`) 사용.
`numeric_match`(±2%, %/비율 스케일 관대)는 자체 진단 전용 — 논문에 "정확도"로 인용 금지.

### 현재 스냅샷 (HiTab dev arith m≥2, **n=161**, post-audit)

**§5.1b 천장 진단 (LLM-free):** gold operand의 28.5%가 이름 없는 총합행(비율 분모);
유사도 중앙랭킹 39.5위 vs 일반 operand 8위(~5×); top-50 도달률 0.59 vs 0.91;
dense full-set 완전성 @50=0.714에서 정체하고 **그 실패의 76%(35/46)가 미도달 총합행**.
구조적 미도달(예산 문제 아님)이 핵심 주장(기여 2). (`dense_ceiling_diag`)

**§5.10 총합-행 주입 — 사례연구 (same-depth k=10, 탐지 = keyword ∪ structural 하이브리드,
bge-reranker 열 resolver, 주입 4.4셀/쿼리, 07-08 post-audit 정본):**

| baseline | plain OSC | +주입 | Δ | flip:손해 | McNemar p | recall plain→aug |
|---|---|---|---|---|---|---|
| BM25 | 0.770 | **0.876** | +0.106 | 17:**0** | **1.5e-5** | 0.887→0.945 |
| dense | 0.832 | 0.863 | +0.031 | 5:**0** | 0.0625 (n.s.) | 0.920→0.946 |
| hybrid | 0.839 | **0.907** | +0.068 | 11:**0** | **0.001** | 0.927→0.964 |

> 주의: 새벽(07-08) 세션이 기록한 0.814/0.863/0.888은 브랜치 분기 사고로 **MiniLM resolver**로
> 측정된 값 — 검증된 승리 구성(bge-reranker, §5.4)의 위 수치가 정본.

recall 열이 핵심 전시물: plain도 평균 per-cell recall은 0.89–0.93인데 완전 회수(OSC)는
0.77–0.84 — 레퍼런스들이 쓰는 관대한 지표가 주입이 겨냥하는 실패를 가림. 모집단의
35%(56/161)만 total-row operand를 실제로 필요로 함 — 그 서브그룹에선 BM25 gap **65%**
폐쇄(0.536→0.839, p=2e-5), hybrid **58%**(0.661→0.857, p=.001), dense 31%(p=.0625 n.s.);
나머지 65%는 정확히 Δ=0(p=1.0). `is_total_row_structural`(자식/형제 행 합 산술검증,
언어독립)이 keyword 정규식과 union으로 기본 탐지기가 됨 — 단 structural *단독*은 keyword보다
약함(일반성 "부분" 해소). **프레이밍: 진단(기여 2)이 실행가능함을 보이는 사례연구(기여 5)이지
일반 검색기 개선이 아님.** 정직 단서: deeper-budget 비교(plain@20)에선 dense/hybrid는 plain이
유의하게 이김(BM25는 동률 p=1.0); strict cell-matched(쿼리별 동일 셀 예산)에선 **BM25만
유의**(+0.106, p<1e-4; dense/hybrid +0.019 n.s.) — 06-29 "strict 셋 다 승리"는 감사 이전
수치로 폐기. (`osc_total_augment`)

**E8/E9 — vs OHD 전체직렬화 (07-08 post-audit 재측정):** 전체표 직렬화는 35%가 8k 컨텍스트
초과(평균 8.5k 토큰, ours 753–953). E9 같은-지표 OSC@토큰예산: hybrid+inject가 B=250–8k
**전 구간 유의**(p≤0.004), @8k **1.000 vs 0.870**(관대 truncation 변형 기준, 21:0 flip;
faithful dual-OHD 대비 ~0.5 격차). 전체표는 B≥16k에서만 동률. **recall 병기(신규):** 평균
per-cell recall로는 @2k에서 hybrid plain이 0.92인데 OSC는 0.84 — 레퍼런스 지표(부분 recall)가
집계 실패를 가리는 것 자체가 전시물. 단 B≤500 기아예산에선 주입이 plain보다 손해
(@500 dense 0.497 vs 0.441, crossover ≈1k). (`e9_osc_token_budget`)

**H6 — OSC 이득 → 답변 정확도 전환 (공식 EM, dense@10 vs +주입, paired, 부분표본;
07-08 오후 `--resume` 확대 후):**

| solver | 평가 n | base | treat | Δ | flips | McNemar p |
|---|---|---|---|---|---|---|
| llama-3.1-8B | 134/161 | 0.112 | 0.119 | +0.008 | 1:0 | 1.0 |
| llama-3.3-70B | 52/161 (쿼터컷) | 0.346 | 0.404 | +0.058 | 3:0 | 0.25 |
| **gpt-oss-120b** | **86/161 (쿼터컷)** | 0.395 | **0.500** | **+0.105** | **11:2** | **0.0225** |

8B는 솔버 병목으로 무반응(E7 결론 유지); **gpt-oss-120b는 표본을 37→86으로 2.3배 늘려도
유의 유지** — OSC-flip 11개 쿼리에선 정확도 0.00→0.73(8:0). Δ가 n=37 시점(+0.243)보다 준 것은
flips-first 정렬의 예상된 희석(주입 수혜 쿼리를 먼저 평가)이므로 모집단 효과 추정치는 +0.105쪽이
정직함. 여전히 부분표본 — 잔여 75개는 쿼터 회복 후 `--resume`. 주의: treat 컨텍스트 4/86
truncation 감지됨. (`results/h6_rerun_20260707/`)

**§5.11 일반화 스코프:** WikiSQL은 메커니즘 **부적용**(named total row 0% — 집계가 저장되지
않음), FinQA는 **불필요**(중앙값 5행, k≥10이면 전체표 포함 → 주입 평균 0.1셀, ΔOSC=0; 천장
0.315는 operand 분해 문제). 기여의 정확한 스코프 = "저장된 집계행을 가진, 예산에 안 들어가는
큰 계층표". (`finqa_total_inject`, `total_inject_generalization_summary`)

**전 데이터셋 검색 스윕(테이블 랭킹, 참고):** R@1 — hitab: dense 0.703 최강 / finqa: hybrid
0.168 / wikisql: hybrid 0.534(dense 0.357 최약). (`multidataset_retrieval_summary`)

## Evaluation population

Gold operands are resolved from HiTab `linked_cells.quantity_link` by value-matching
into data space (`rag_agent/bench/hitab.py:resolve_gold_operands`). Value-matching
yields a clean operand set **only for arithmetic aggregations** (the answer is a
computed number). For selection/comparison queries (argmax/argmin/greater_than…)
the answer is a *header label*, so there is no value to match and the gold operand
set is empty — these are reported as a limitation, not evaluated.

| population | n | use |
|---|---|---|
| arithmetic, operands resolved | 214 | E1/E2 curve (incl. m=1 anchor) |
| **arithmetic, m≥2 (true scope)** | **158 → 161(현재)** | **primary OSC population** |
| selection/comparison | (excluded) | value-matching cannot build gold (limitation) |

> n=158은 초기 gold 빌드 기준(아래 역사 섹션들의 측정 시점). 2026-07 gold 재빌드 +
> offset-버그 수정 후 현재 모집단은 **n=161** — 위 스냅샷과 `PAPER_DRAFT.md`의 기준.
> 좁힘 자체가 가정이 아니라 측정 결과임은 dev 전수(n=1,301) 스윕으로 확인: lookup·m=1·
> selection은 유사도로 이미 OSC 0.94–1.00, 무너지는 유일한 구간이 arith m≥2
> (`osc_population_expansion`, 2026-07-03).

Integrity: of 1671 dev queries, 22% have empty operand sets — diagnosed as
label-answer queries (349) + 21 genuine value-resolution failures, **not** a
retriever/LLM failure (`results/operand_gold_report.json`).

## Metric

**OSC** (Operand-Set Completeness) = fraction of queries where *every* gold operand
cell is retrieved (all-or-nothing subset containment). Necessary condition for a
correct aggregation answer; strictly harder than averaged per-cell recall.
Implementation + unit tests: `rag_agent/eval/operand_set.py`,
`tests/test_operand_set.py` (10/10).

---

## H1 — dense baseline OSC collapses with scope size (E1)

Baseline: dense single-vector retrieval (`mode="plain"`, bge-small, S2 row-chunks).

**OSC vs scope size m, at fixed budget k** (the collapse):

| k ＼ m | 1 | 2 | 3–4 | 5–8 | 9+ |
|---|---|---|---|---|---|
| 1 | 0.68 | 0.20 | 0.19 | 0.13 | 0.14 |
| 5 | 0.89 | 0.60 | 0.62 | 0.53 | 0.29 |
| 10 | 0.96 | 0.79 | 0.85 | 0.67 | 0.43 |
| 20 | 0.98 | 0.92 | 0.92 | 0.87 | 1.00\* |

\*n=7. **m≥2 (n=158) overall OSC:** k=1 → 0.19, k=5 → 0.58, k=10 → 0.77, k=20 → 0.92.

**Verdict: H1 supported.** At realistic budgets (k≤10) OSC falls monotonically as
the aggregation scope grows. A larger budget partially rescues completeness, but
only by dumping ~20 chunks into context — completeness is bought with budget, not
targeting. First-order independence (r^m) is a reasonable fit at tight budget.
Detail: `results/e1_osc_baseline_summary.md`, `results/e1_osc_baseline.json`.

---

## H2 — header-tree enumeration re-localizes the bottleneck (E2)

Treatment: deterministic header-tree scope enumeration — resolve the query to
header-path predicates, then enumerate every numeric leaf under the matched scope
nodes (`rag_agent/retrieve/header_enum.py`, tests 4/4). Paired vs dense baseline,
m≥2, n=158.

| metric | value |
|---|---|
| OSC enumeration | 0.335 (mean 17.2 cells) |
| **OSC \| decomposition correct** | **1.000** (n=53) |
| row-axis coverage | 0.544 |
| col-axis coverage | 0.728 |
| ΔOSC vs dense k=5 | −0.247, CI [−0.335, −0.158] |
| ΔOSC vs dense k=10 | −0.437, CI [−0.519, −0.348] |

OSC | decomposition-correct, **by scope size**: 1.0 at m=2, 3–4, 5–8, 9+ — flat.

**Verdict: H2 revised, not naively confirmed.**
1. On *raw* OSC, enumeration **loses** to the dense baseline (ΔOSC significantly
   negative): a missed header predicate zeroes a query, whereas similarity ranking
   degrades gracefully.
2. But the **mechanism is fully validated**: conditional on correct decomposition,
   enumeration recovers the complete operand set **100% of the time, independent of
   scope size**. The H1 collapse curve is *eliminated*.
3. `OSC_enum (0.335) = decomposition success rate (53/158)` exactly. Enumeration
   **converts operand-set completeness into a header-path decomposition problem**
   and localizes the bottleneck to the **row axis** (0.544 vs col 0.728).

The contribution is the **re-localization of the bottleneck** — from the
theoretically-hard arbitrary-subset-selection limit (Weller et al. 2508.21038,
which H1 exhibits) to a separable, measurable decomposition problem — not a raw
OSC win. Detail: `results/e2_osc_enum_summary.md`, `results/e2_osc_enum.json`.

---

## W4b — the decomposition bottleneck is largely model-agnostic

Lever: refine decomposition with an LLM choosing header paths from the real
inventory (`resolve_intent`), to raise row-axis coverage. Tested at two scales.

| metric | deterministic | 8b | 70b |
|---|---|---|---|
| row-axis coverage | 0.544 | 0.506 | **0.595** |
| OSC enum | 0.335 | 0.285 | **0.380** |
| n decomp correct | 53/158 | 45/158 | **60/158** |
| mean enum cells | 17.2 | 16.0 | **8.9** |
| OSC \| decomp correct | 1.000 | 1.000 | 1.000 |
| ΔOSC vs k=10 | −0.437 | −0.487 | **−0.392** |

- A *weak* 8b model **degrades** decomposition below the deterministic fuzzy ranker.
- A *strong* 70b model **partly lifts** it (row-axis +0.05, n-correct 53→60) and is
  far more precise (17→9 cells), but a ~9× larger model still **does not beat the
  dense baseline** (ΔOSC significantly negative).

The row-axis ceiling is **not closed by LLM scale** in the available range — it is a
representation/matching problem, not a model-capacity one. The next lever is the
decomposer's representation, not a bigger model. The enumeration invariant
(OSC | decomp = 1.0) holds across all three. Detail: `results/e2_osc_enum_summary.md`.

---

## E3 — header depth is a method-specific liability (causal)

Holding data, leaf vocabulary, and scope fixed, leaf-flatten every table to depth 1
(drop ancestor header levels) and re-measure, paired (n=158, m≥2, LLM-free).

| flatten (d→1) effect | OSC original → flat | Δ |
|---|---|---|
| enumeration | 0.335 → 0.570 | **+0.234** |
| dense baseline | 0.772 → 0.703 | **−0.070** |

Removing the header tree (same words, same data) **raises enumeration OSC by +0.23**
(col-axis coverage 0.73→0.93) but **lowers the dense baseline by −0.07**. The two
methods respond to depth in *opposite* directions: depth is not intrinsic to the
completeness problem (the baseline is depth-robust) — it is a **method-specific
liability of resolve-then-enumerate**, because the fuzzy resolver cannot map queries
onto deep header paths. Caveat: the flattened enum scope is 2.2× larger (37.9 vs
17.2 cells), so part of its OSC gain trades precision for completeness. Detail:
`results/e3_depth_summary.md`, `results/e3_depth.json`, `results/e3_depth_dense.json`.

Together with W4b, the open problem is sharpened to **depth-robust
query→header-path resolution** — a representation problem, not model scale or budget.

## E4 — structured context cuts silent grounding errors (H3)

Retrieval held fixed at the oracle operand set; only the context *format* varies
(same numbers, same header words). Codegen, Groq llama-3.1-8b, n=158, m≥2.

| arm | NM accuracy | silent-wrong rate |
|---|---|---|
| flat dump | 0.335 | 0.665 |
| **(header-path = value)** | **0.576** | 0.424 |

ΔNM = **+0.241** CI [0.158, 0.323]; McNemar 49:11. **H3 supported** — making the
binding explicit nearly doubles numeric-match accuracy and cuts the silent-error
rate, even with perfect retrieval. Residual silent-wrong 0.42 is the 8b model's
own grounding/arithmetic limit (non-number rate 0). Detail:
`results/e4_format_summary.md`, `results/e4_format.json`.

## Idea follow-up — embedding tree-node resolver + recall-first

Tests the idea "represent row headers as a tree and match query→header by semantic
embedding" against the row-axis bottleneck. LLM-free, n=158.

| resolver | row-cov | col-cov | OSC enum |
|---|---|---|---|
| lexical (fuzzy) | 0.544 | 0.728 | 0.335 |
| embed (tree-node) | 0.582 | 0.677 | 0.361 |
| **hybrid (row=embed, col=lexical)** | 0.582 | 0.728 | **0.380** |

- Embedding fixes the **row axis** (vocabulary mismatch) but hurts the **column
  axis** (years/codes match better lexically) → hybrid keeps both. The hybrid is
  LLM-free yet **matches the 70b LLM exactly** (OSC 0.380, n_decomp 60): a targeted
  representation fix equals a 9× larger model — nailing "representation, not scale".
- **Depth re-test (E3 embed):** the embedding resolver halves the flatten benefit
  (+0.234→+0.127) and on the row axis flips depth from liability to slight asset
  (row-cov 0.582→0.551 when flattened). The idea's hypothesis holds on the
  bottleneck axis.
- Still below the dense baseline: the residual is **structural scope selection**
  (which/how-many sibling rows), not vocabulary.

**Recall-first (E5) — meeting a 100%-completeness requirement:**

| config | OSC | mean cells (whole table ≈162) |
|---|---|---|
| enum precise (hybrid) | 0.380 | 19 |
| enum axis-complete | 0.930 | 87 |
| dense top-20 | 0.918 | 99 |
| **union(axis-complete, dense)** | **1.000** | 123 |
| whole table | 1.000 | 162 |

100% completeness **is** achievable (union), which similarity ranking alone cannot
guarantee — but it costs ~76% of the whole table. Completeness and precision are in
tension; "100% within a *small* set" remains the open problem. Detail:
`results/e2e3_embed_resolver_summary.md`.

## E6 — structural scope selection: what's left after lexical/depth is fixed

The residual row-axis gap (after the embedding idea closed lexical/depth) was
attributed to "structural scope selection". We **diagnosed it before treating it**
(`scripts/diag_row_failures.py`, current rebuilt gold n=161, hybrid row-cov 0.615):

| row-axis failure structure | share of 62 failures |
|---|---|
| **total_pairing** (share/ratio query needs a table-level total it can't name; 68% are `div`) | **68%** |
| sibling_subset (gold ⊂ children of one parent) | 15% |
| cross_parent (genuine multi-entity cross-cut) | 11% |
| parent_expandable (gold = all children of one parent) | 6% |

The dominant cause is **not** sibling selection (the pre-registered hypothesis) but a
**missing total/denominator row** with an empty/top-level header path. Diagnosis-
driven treatments (row augmentations over the same hybrid scope), paired:

| arm | OSC | ΔOSC vs hybrid base | ΔOSC vs dense k10 | mean cells | row-cov |
|---|---|---|---|---|---|
| base (hybrid enum) | 0.416 | — | −0.373 [−0.453, −0.286] | 19 | 0.615 |
| **T_total** (+total rows) | 0.596 | **+0.180** [0.124, 0.242] | −0.193 | 30 | 0.845 |
| T_subtree (+sibling group) | 0.460 | +0.043 [0.012, 0.075] | −0.329 | 31 | 0.665 |
| **T_both** | **0.652** | **+0.236** [0.174, 0.304] | **−0.137** [−0.236, −0.037] | 40 | 0.888 |

**Verdict.** Total-row augmentation is the single highest-value lever (the diagnosis
was right); sibling expansion is significant but minor (the pre-registered guess
targeted the 6–15% minority). Both are **pure-superset** gains (McNemar b:0; OSC
monotone), so the cost is precision: cells 19→40 (~2×). The levers **close ~63% of
the gap to the dense baseline** (−0.373→−0.137) and **T_total ties dense k=5 exactly**
(0.596), but **still do not beat dense k=10 on raw OSC** — H2's "no raw win" stance
holds. The binding constraint is now the **untouched column axis** (col-cov 0.733
across all arms), not row scope. Detail: `results/e6_scope_treatments_summary.md`,
`results/diag_row_failures_summary.md`.

## Differentiation gate (W0)

All four nearest works verified (method sections, `docs/RELATED_DELTA.md`):
DCTR and Huawei-TableRAG **exclude aggregation from retrieval** and defer it to SQL
on flat relational schemas; T-RAG's "hierarchical" is a corpus index (its benchmark
lacks operand labels); HD-RAG models the internal header tree but only for top-1
*document* retrieval, never enumerating the scope at retrieval time. **No prior work
puts header-tree scope enumeration / operand-set completeness as a retrieval-time
objective.** Gate passes.

## Hypothesis scorecard (numbers only)

| H | claim | verdict |
|---|---|---|
| H1 | dense single-vector OSC degrades with scope size m | **supported** (E1) |
| H2 | header-tree enumeration improves operand-set completeness | **revised**: removes scope-size dependence (OSC\|decomp=1.0 flat) and re-localizes the bottleneck to row-axis decomposition; does **not** beat raw baseline OSC under the deterministic/8b/70b decomposer (E2, W4b) |
| H2-causal | the enumeration effect is hierarchy-caused, not domain | **supported, with a twist** (E3): depth causally suppresses enumeration OSC (flatten→ +0.234) but the dense baseline is depth-robust (−0.070) — depth is a method-specific liability of resolve-then-enumerate, not intrinsic to completeness |
| H3 | structured (header-path, value) context reduces silent grounding errors | **supported** (E4): oracle-fixed retrieval, ΔNM +0.241 [0.158, 0.323], silent-wrong 0.66→0.42 |
| H4 | the residual row-axis gap is structural scope selection, fixable by targeted enumeration | **partly supported** (E6): diagnosis shows it is dominated by *total-row pairing* (68%), not sibling selection; total-row augmentation +0.180 OSC [0.124, 0.242] paired vs hybrid, T_both +0.236 — closes ~63% of the dense gap and ties dense k=5, but still does **not** beat dense k=10 (Δ −0.137); residual cap moves to the column axis |
| H6 | injection's completeness gain converts to answer accuracy at a capable solver | **first significant signal, partial sample** (2026-07-08 rerun, official EM): 8B flat (p=1.0, solver-bound); 70B directional (n=23); **gpt-oss-120b 0.405→0.649, Δ+0.243, 10:1 flips, p=0.012 — at n=37/161 (quota cut)**; not confirmatory until the sample is extended (`--resume`) |

## Threats / limitations

- OSC(operand 라벨) 측정은 여전히 HiTab(+MultiHiertt 어댑터)뿐. 다만 외적 타당성은 부분
  확보: 열 선택 cross-encoder는 AITQA에서 순서 재현, MultiHiertt는 예측된 null(총합에 이름이
  붙는 재무표), FinQA/WikiSQL은 주입 메커니즘의 적용범위 경계를 확정(§5.11 스냅샷 참조).
- OSC is upper-bounded by header-path decomposition accuracy; we report it
  conditionally (OSC | decomp) to separate enumeration from decomposer quality.
- Selection/comparison aggregations excluded from gold (value-matching limitation).
- Baseline k=10 uses a larger effective cell budget than enumeration's ~17 cells;
  part of the raw-OSC gap is budget, not targeting.
- W4b tested 8b and 70b; LLM scale does not close the row-axis ceiling, but only
  Groq-hosted Llama models were tried (no frontier model / no fine-tuned decomposer).

## Reproduce

```
PYTHONPATH=. python scripts/build_operand_gold.py --split dev
PYTHONPATH=. python scripts/e1_osc_baseline.py   --split dev
PYTHONPATH=. python scripts/e2_osc_enum.py       --split dev
PYTHONPATH=. python scripts/e2_osc_enum.py       --split dev --llm groq:llama-3.1-8b-instant
PYTHONPATH=. python scripts/e3_depth.py          --split dev [--dense]
PYTHONPATH=. python scripts/e4_format.py         --split dev --llm groq:llama-3.1-8b-instant
PYTHONPATH=. python scripts/diag_row_failures.py --split dev          # E6 diagnosis
PYTHONPATH=. python scripts/e6_scope_treatments.py --split dev --dense  # E6 treatments
PYTHONPATH=. python scripts/dense_ceiling_diag.py  --split dev          # §5.1b diagnosis
PYTHONPATH=. python scripts/osc_total_augment.py   --split dev --resolver-cols  # §5.10 injection
PYTHONPATH=. python scripts/e9_osc_token_budget.py                      # E9 OSC@budget vs OHD
PYTHONPATH=. python scripts/finqa_total_inject.py                       # §5.11 generalization
PYTHONPATH=. python scripts/answer_accuracy_injection.py --solver-model openai/gpt-oss-120b \
  --codegen-max-tokens 1024 --flips-first --resume \
  --out results/h6_rerun_20260707/gptoss120b.json \
  --records results/h6_rerun_20260707/gptoss120b_records.jsonl         # H6 (quota-gated)
```

Note: this repo's working Python is the system interpreter (`/usr/bin/python3`,
pandas + sentence-transformers); run from `rag-agent/` with `PYTHONPATH=.`. There is
no `pytest` — run the unit tests by importing each `tests/test_*.py` and calling its
`test_*` functions (the row-failure diagnosis / E6 add 7 tests in
`tests/test_header_enum.py`).
