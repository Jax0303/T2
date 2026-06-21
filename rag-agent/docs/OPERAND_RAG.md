# OPERAND_RAG.md — Operand-targeted RAG over (hierarchical) tables

생성: 2026-06-18 · 코드: `rag_agent/{bench,serialize,query,retrieve,generate}/` ·
러너: `scripts/operand_rag_eval.py` · 테스트: `tests/test_{serialize,bench_loaders,operand_decomposer,operand_retriever,coverage,answerer}.py`

> 본 문서는 `Claude Code Prompt Engineering_apa.docx` / `클로드코드 구현_apa.docx`의 구현
> 스펙을 기존 `rag-agent` 자산 위에 올린 결과물이다. 핵심 주장: **복잡한 계층 표에서 RAG의
> 병목은 검색 완전성(operand recall)이며, 그 상한은 질의를 헤더경로로 분해·정합하는 능력
> (decomposition ceiling)이다.** [[METHOD_HPIR]]의 헤더경로 IR을 operand 단위로 일반화한다.

---

## 1. 파이프라인 (스펙 5 컴포넌트 ↔ 모듈)

| 스펙 컴포넌트 | 모듈 | 비고 |
|---|---|---|
| ① 직렬화 S1/S2 | `serialize/serializers.py` | 행 단위 청크, 셀 커버리지 기록 |
| ② HPIR operand 분해 + 헤더경로 정합 | `query/operand_decomposer.py` | fuzzy/embedding/hybrid, **천장 지표** |
| ③ operand-targeted 검색 | `retrieve/operand_retriever.py` | BM25+dense(RRF), `operand_recall@k` |
| ④ coverage 체크 + fallback | `retrieve/coverage.py` | 신뢰도 기반, 전체표 폴백 |
| ⑤ 생성 + 평가 | `generate/answerer.py` | direct / codegen(가드 실행), EM/numeric |

세 벤치마크(HiTab 계층 / FinQA 재무 / WikiSQL 평면 대조군)는 단일 스키마
`bench/schema.py`(`BenchTable`/`BenchQuery`/`GoldOperand`)로 통일되어 모든 단계가 한 번 작성으로
세 곳 모두 동작한다(`bench/{hitab,finqa,wikisql}.py`, dispatcher `bench/registry.py`).

## 2. Gold operand 정의 (벤치마크별)

operand = 정답이 의존하는 **데이터 셀**, 헤더경로 `left_path > top_path`로 식별.

- **HiTab**: `linked_cells.quantity_link`의 셀. *좌표 주의* — 원본 좌표는 data 행렬과 불일치
  (실측: 좌표 일치 1/235). **값 매칭**으로 복원(382/384=99.5%, 고유 72%), `entity_link` 헤더로 tie-break.
- **WikiSQL**: gold SQL(`sel`/`agg`/`conds`) 실행. operand = 매칭 행의 `sel`·조건 컬럼 셀. 평면 → 행 경로 없음.
- **FinQA**: `gold_evidence`의 `the {row} of {col} is {val}` 절을 파싱해 셀 복원. parquet 브랜치 로드.

## 3. 측정 지표

1. **header_path_match_accuracy** (분해 천장): 질의로 표의 헤더경로를 랭킹했을 때 gold operand의
   헤더경로가 상위에 오는 비율. operand-targeted 검색의 상한.
2. **operand_recall@k**: gold operand를 덮는 청크가 회수집합에 있는 비율. `k=1,3,5,10`.
   모드 `plain`(no-HPIR) / `operand` / `oracle`(gold 경로).
3. **answer_accuracy**: direct·codegen × {full, no-fallback, no-HPIR, S1-only} (LLM 필요).
4. **coverage_rate 분포** + **fallback 발동률**.

## 4. baseline / ablation 매핑

- BM25-only → `plain`, S2 (행 평면 대조는 S1).
- TableRAG식 schema+cell → (후속) 컬럼 헤더 검색 + 셀 검색 변형.
- Oracle → `oracle`(gold operand 경로).
- ablation: `plain↔operand`(HPIR 기여) · `S1↔S2`(구조보존 기여) · `full↔no-fallback`(폴백 가치).

## 5. 핵심 측정 결과 (seed=42, HiTab dev n=241, BGE-small dense)

전체 실행값: `results/operand_rag/hitab/summary.{json,md}`.

### 5.1 분해 천장 (header_path_match_accuracy)
| matcher | HiTab |
|---|---|
| fuzzy | 0.303 |
| embedding | **0.486** |
| hybrid | 0.409 |

→ 헤더경로를 **이름 붙이는 능력**이 계층 표에서 낮다(≤0.49) = 근본 병목.
([[thesis-preprocessing-complexity-diagnosis]]의 메타데이터 이득 flat≫hier와 일관.)

### 5.2 operand_recall@k (HiTab, dense) — 3중 신호
| 직렬화 | mode | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|---|
| S1(flat) | plain | 0.530 | 0.707 | 0.797 | 0.897 |
| S1(flat) | operand | 0.616 | 0.707 | 0.790 | 0.885 |
| S2(header-path) | plain | 0.693 | 0.839 | 0.905 | 0.952 |
| S2(header-path) | operand | 0.695 | 0.786 | 0.830 | 0.901 |
| S2(header-path) | **oracle** | **0.896** | 0.957 | 0.974 | 0.994 |

해석(정직한 그림):
1. **구조보존 효과(S2≫S1)**: plain R@1 0.693 vs 0.530 — header-path 직렬화 자체가 가장 큰 이득.
2. **거대한 상단 여유(oracle≫operand)**: R@1 0.896 vs 0.695 — *분해만 완벽하면* operand 검색이 압도.
   현재 operand가 plain과 비슷한 이유는 **분해 천장이 0.30–0.49로 낮기 때문**(§5.1). → 병목은 검색기가
   아니라 **operand 분해**임을 oracle–operand 격차가 직접 증명.
3. operand는 약한 직렬화(S1)에서 plain보다 이득(R@1 0.616 vs 0.530)이나, 강한 S2+dense는 raw query가
   이미 신호를 흡수해 고-k에서 operand가 소폭 손해(상위 4개 operand 쿼리 union이 다양성 제한).

### 5.3 coverage + fallback (HiTab, score-based)
mean coverage 0.849, fallback **15.8%** 발동, `operand_recall@5` 0.836 → **0.868**(+0.031) 회복.
self-supervised coverage 신호는 약하므로(미스 min-score 0.354 vs 정상 0.393) 폴백 가치는 답변정확도
ablation으로 확정한다. coverage 히스토그램: {0.0:28, 0.2:4, 0.5:6, 0.8:10, 1.0:193}.

### 5.4 세 벤치마크 종합 (전체 실행, dense, S2)
n: HiTab 241 / FinQA 238 / WikiSQL 300. 전체값 `results/operand_rag/{hitab,finqa,wikisql}/`.

| 벤치마크 | 천장 hybrid | plain R@1 | operand R@1 | oracle R@1 | fallback율 | R@5 폴백전→후 |
|---|---|---|---|---|---|---|
| HiTab(계층) | 0.409 | 0.693 | 0.695 | **0.896** | 0.158 | 0.836→0.868 |
| FinQA | 0.544 | 0.596 | **0.744** | 0.903 | 0.147 | 0.971→0.983 |
| WikiSQL(평면) | 0.787 | **0.756** | 0.268 | 0.197 | 0.713 | 0.526→0.865 |

세 가지 정직한 결론:
1. **FinQA = 메서드 명확한 승리**: operand R@1 0.744 ≫ plain 0.596 (+0.15). 행 라벨이 헤더경로를 형성해
   operand 타게팅이 곧장 작동.
2. **HiTab = 천장에 묶인 잠재력**: operand≈plain이나 oracle 0.896이 거대한 상단 여유를 드러냄 →
   병목은 검색기가 아닌 **operand 분해**(천장 ≤0.49). 분해 개선이 다음 레버.
3. **WikiSQL = 음성 대조군(설계대로)**: 평면 표는 행을 조건값으로 찾으므로 헤더경로 operand 무력
   (operand·oracle 모두 plain보다 낮음). 메서드 이득이 **계층 복잡도에 특이적**임을 반증으로 확인.
   (평면이라 S1≡S2 — 직렬화 동일성 sanity check 통과.)

## 6. 재현

```bash
VENV=/home/user/T2/hart-table-retrieval/.venv/bin/python
# 검색 측(LLM 불필요) 전체 표 생성
$VENV scripts/operand_rag_eval.py --bench hitab  --max-samples 300 --device cuda
$VENV scripts/operand_rag_eval.py --bench finqa  --max-samples 300 --device cuda
$VENV scripts/operand_rag_eval.py --bench wikisql --max-samples 300 --device cuda
# 답변 측(LLM) — GROQ 일일한도 시 로컬로
$VENV scripts/operand_rag_eval.py --bench hitab --llm local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit \
      --answer-samples 100
# 단위테스트 (데이터 불필요)
$VENV -m pytest tests/test_{serialize,bench_loaders,operand_decomposer,operand_retriever,coverage,answerer}.py -q
```

출력: `results/operand_rag/<bench>/summary.{json,md}` + `records.jsonl`(쿼리별 로그).

## 7. 한계 / 위협 타당성
- WikiSQL operand 검색 무력은 의도된 대조이나, 평면 표에서 조건값을 operand 쿼리에 포함하면 개선 여지.
- FinQA gold_evidence 파싱 의존 → 미해소 절은 drop(로그). operand 결측 ~17%.
- self-supervised coverage 신호 약함 → fallback 트리거 보수적; 답변정확도 ablation이 최종 판정.
- 천장은 BGE-small·top-|gold| 예산 기준 — 임베더/예산 민감도는 후속 ablation.
