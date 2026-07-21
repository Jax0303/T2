# rag-agent

표 RAG 연구 패키지. 현재 주제는 **다중 표 코퍼스에서 집계 질의의 피연산자 충돌
진단과 구조 직렬화 처방** — 전체 서사와 헤드라인 수치는
[루트 README](../README.md)와 [`docs/LAB_MEETING_BRIEF.md`](docs/LAB_MEETING_BRIEF.md)
(발표용 정본) 참조.

## 핵심 수치 (2026-07-13 확정)

- 충돌 진단: flat에서 충돌라벨 피연산자 중앙랭크 **180 vs 일반 16** (hybrid, p=6e-18)
- 처방 효과: all-covered@50 **0.458 → 0.593** (hybrid flat→S3, +45/−5, p=4.2e-9)
- 충돌 규모: 표 240→1,203개로 늘리면 충돌 피연산자 12.7%→**34.8%**
- 계층 자가복원 (**2026-07-21 실측치로 갱신**): HiTab 실제 소스 격자에서
  열 **97.6%** / 행 **58.2%** (train n=2,043, 경계 gold). 이전에 적혀 있던
  96.7/88.2는 `flatten_to_grid`로 합성한 격자를 되돌린 **라운드트립**
  수치라 알고리즘 자기일관성만 보여준다 — 근거는
  [`docs/RECONSTRUCTION_VALIDITY.md`](docs/RECONSTRUCTION_VALIDITY.md).
  행 58.2%는 디코더 결함이 아니라 표현 격차: 행 계층이 stub 열 블록에
  들어가는 표는 **99.5%**, 한 열 안에 쌓인 표(41%)는 **9.7%**.

## 모듈

| 경로 | 역할 |
|---|---|
| `rag_agent/reconstruct/` | raw HTML/그리드 → 행·열 헤더 경로 자가복원 (forward-fill, 정답 트리 안 씀) |
| `rag_agent/serialization/` | 셀 직렬화 3방식: `flat.py`(라벨만) / `header_path.py`(S2 트리매핑) / `caption.py`(S3 캡션 문장, short·medium·long) |
| `rag_agent/stores/` | `original_store.py`(원본 2D 보관 — 값 읽기·계산·검증, 벡터와 분리) + `vector_store.py` |
| `rag_agent/retrieve/` | 하이브리드 검색·재랭크·헤더트리 열거(`header_enum.py`)·커버리지 폴백 |
| `rag_agent/extract/` | LLM 셀 선택 + 안전 AST 계산 (이전 단계 파이프라인, 유지) |
| `rag_agent/eval/metrics.py` | R@k · MRR · nDCG@10 · EM · NM (문헌 정렬), OSC 계열은 실험 스크립트에서 |

## 주요 스크립트

```bash
# 충돌 진단+처방 (핵심 실험): flat/S2/S3 × bm25/dense/hybrid × 캐스케이드
PYTHONPATH=. python scripts/operand_collision_multihiertt.py \
    --max-queries 300 --out results/operand_collision_multihiertt_n300.json

# 유의성 검정: ①MWU(진단) ②Wilcoxon(처방) ③exact binomial(완전성 flip)
PYTHONPATH=. python scripts/operand_collision_significance.py \
    results/operand_collision_multihiertt_n300_records.jsonl

# 계층 복원 정확도 — 인용할 수치는 _raw 쪽(실제 소스 격자)이다
PYTHONPATH=. python scripts/tree_reconstruct_hitab_raw.py --split train  # 실측
PYTHONPATH=. python scripts/tree_reconstruct_hitab.py        # 라운드트립(자기일관성 점검용)
PYTHONPATH=. python scripts/tree_reconstruct_multihiertt.py  # raw HTML 3.1만 표 근사 검증

# 교수님 3단계 베이스라인: 표 검색(1테이블=1청크) → 표 내부 셀 검색 → 답 채점
PYTHONPATH=. python scripts/s3_table_chunk_baseline_multihiertt.py \
    --scheme S3 --rerank-k 10   # S2로 바꾸면 트리매핑, --rerank-k 1이면 재랭크 OFF

# 에이전트 파이프라인 엔드투엔드 (검색→symbolic/reader→채점). 답변 지표는 hmtEM.
PYTHONPATH=. python scripts/run_eval.py --config configs/v3.1_baseline.yaml \
    --data-dir data/hitab --chroma-dir data/chroma_db \
    --llm groq:llama-3.3-70b-versatile --retriever-device cpu
PYTHONPATH=. python scripts/smoke_test.py \
    --data-dir data/hitab --chroma-dir data/chroma_db --device cpu  # LLM 없이 배선 점검
```

결과 JSON은 전부 `results/`에 저장되며 per-record `.jsonl`(셀 단위 랭크)이 함께
남아 유의성 검정을 재현할 수 있다.

## 문서

- [`docs/LAB_MEETING_BRIEF.md`](docs/LAB_MEETING_BRIEF.md) — **발표용 정본** (최신 수치, 예상 질문 포함)
- [`docs/PAPER_DRAFT.md`](docs/PAPER_DRAFT.md) — 논문 초안
- [`docs/CITATIONS_VERIFIED.md`](docs/CITATIONS_VERIFIED.md) — 인용 논문 원문 대조 로그
- [`EXPERIMENTS.md`](EXPERIMENTS.md) — 이전 단계(적응형 라우팅 에이전트) 상세 실험 기록
- [`STATE.md`](STATE.md) — 세션 간 작업 상태

## 이전 단계 기록

HiTab 하드 40쿼리 적응형 라우팅 에이전트(NM 0.475 vs 벤치 0.250), 검색 논제
(keyword 0.646 vs dense 0.618), codegen 평가 등 이전 단계의 상세 수치·trace는
`EXPERIMENTS.md`와 git 이력에 보존.
