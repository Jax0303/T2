# STATE.md — Phase 0 현재 상태 점검

생성: 2026-06-05 · 브랜치: `main` · 환경: GPU 사용 가능 (RTX 3060 Ti, 8GB, CUDA 12.6 / torch 2.12.0+cu126)

> 본 문서는 T2 검색 라우팅 실험 프로토콜(v1)의 Phase 0 산출물이다. 목적: 재발명 방지를 위한
> 기존 자산 목록화와 "재사용 / 신규" 구분.

---

## 0. 환경 사실 (측정값)

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Ti, 8192 MiB, CUDA 12.6 |
| Python venv | `/home/user/T2/hart-table-retrieval/.venv` (torch 2.12.0+cu126, cuda=True) |
| 핵심 패키지 | chromadb 1.5.9, sentence-transformers 5.5.0, scipy 1.17.1, pandas |
| 미설치 | `rank_bm25`, `faiss` (Phase 2 BM25용 rank_bm25 설치 필요) |
| API 키 | `GROQ_API_KEY` 존재 (LLM 답변 생성용). ANTHROPIC/OPENAI 없음 |
| 임베딩 모델 캐시 | `BAAI/bge-large-en-v1.5`, `intfloat/multilingual-e5-large-instruct`, `Qwen/Qwen2.5-7B-Instruct` |

> 프로토콜 §0-6은 CPU 전용을 가정하나, 사용자 지시로 임베딩/인덱싱 등 병렬 가능 구간은 GPU 사용.
> 통계·검색지표·답변 파이프라인은 결과 동일.

## 1. git 상태

- `git log --oneline -20`: 최근 커밋은 `codegen_eval` / retrieval-only eval 계열.
  최신: `62e8fd8 Retrieval-only eval (no LLM): numeric signal hurts; lexical original-store beats dense VDB`
- 미커밋: `.claude/settings.local.json` (M), `results/codegen_eval_*` 4개 (untracked).
- 신규 실험 산출물은 untracked 상태로 누적 중.

## 2. HiTab 원본 데이터 위치 및 포맷

- 위치(심볼릭 링크 생성): `rag-agent/data/hitab` → `/home/user/T2/hart-table-retrieval/data/hitab`
- 포맷: cell-level JSON 트리. 표 키 = `title, top_root, left_root, data, table_id`
  (계층 헤더 트리 `top_root`/`left_root` 보존).
- 질문 샘플 키 = `id, table_id, table_source, sentence_id, sub_sentence, question, answer, aggregation, linked_cells, answer_formulas, reference_cells_map`
- 규모(측정): 표 `hmt`/`raw` 각 **3597개** · 질문 train **7417** / dev **1671** / test **1584**.
  → 프로토콜 목표 "3,597개 표"와 일치.

## 3. 사전 계산된 인덱스 (재임베딩 불필요 부분)

`rag-agent/data/chroma_db` (→ T2/.../data/chroma_db) 내 컬렉션:

| 컬렉션 | 임베더 | 직렬화 | count |
|---|---|---|---|
| plain_markdown_bge_large_en_v1_5 | bge-large | plain_markdown | 540 |
| json_kv_bge_large_en_v1_5 | bge-large | json_kv | 540 |
| header_path_bge_large_en_v1_5 | bge-large | header_path | 12605(청크) |
| plain_markdown_multilingual_e5_large_instruct | e5-large | plain_markdown | 540 |
| json_kv_multilingual_e5_large_instruct | e5-large | json_kv | 540 |
| header_path_multilingual_e5_large_instruct | e5-large | header_path | 12605(청크) |

> **540 = HiTab dev 1671개 질문이 가리키는 고유 정답표 수(dev pool).** 3 직렬화 × 2 임베더가 이미 인덱싱됨.
> 단, 프로토콜 Phase 1은 **전체 3597개 표 코퍼스**를 요구 → dev pool(540) 외 나머지 표는 미임베딩.
> GPU로 전체 코퍼스 재임베딩은 저비용(아래 결정 사항 참조).

## 4. 이미 측정된 결과 (요약)

- `results/retrieval_eval_full.json` (검색 전용, dev pool=540, n_eval=1671):
  - R@1 — keyword 0.646 · structural_h0 0.641 · **vdb(dense bge) 0.618** · structural_full 0.503
  - paired bootstrap + McNemar 첨부됨. 결론(커밋 메시지): 숫자 신호(w_num=0.4)가 검색을 해치고,
    어휘(lexical) original-store가 dense VDB를 이김.
- `results/ablations_summary.json` (답변 codegen, 소표본 n=50~100):
  - gold_ceiling nm=0.28, adaptive nm=0.18 — **현재 codegen 답변 정확도 자체가 낮음**(상한 0.28).
- `results/codegen_eval_*`, `results/local_qwen7b_*`, `results/groq_*` 등 다수 답변 실행 로그 존재.

---

## 5. 재사용 / 신규 모듈 표 (Gate 0 핵심)

| 프로토콜 요구 | 기존 자산 | 판정 |
|---|---|---|
| HiTab 로더 | `rag_agent/data/loader.py` (load_samples/load_table/load_hitab) | **재사용** |
| 원본 2D 구조 store + 셀 값 조회 (structured 경로) | `rag_agent/stores/original_store.py` | **재사용** |
| dense 벡터 검색 (FAISS↔Chroma) | `rag_agent/stores/vector_store.py` (Chroma+ST, GPU 임베딩) | **재사용** (FAISS 대신 Chroma IndexFlat 동급) |
| Verifier (A3 ablation) | `rag_agent/retrieve/verifier.py` | **재사용** |
| 라우터 (Adaptive-RAG식 휴리스틱) | `rag_agent/router/query_classifier.py` + `router/policy.py` | **재사용** (휴리스틱 명시) |
| 난이도 층화 라벨 | `rag_agent/eval/metrics.py::difficulty_class` (6类: simple_lookup/single_arg/comparison_or_count/arithmetic_agg/pair_or_topk_arg/multi_op_formula) | **재사용** |
| 검색지표 R@k/MRR/nDCG + paired bootstrap + McNemar | `scripts/retrieval_eval.py` | **재사용·확장** (bm25/hybrid/oracle/nocontext 추가) |
| 답변 EM/F1 codegen 파이프라인 | `scripts/codegen_eval.py`, `rag_agent/extract/*`, `rag_agent/llm/*` | **재사용** |
| bootstrap CI 유틸 | `scripts/bootstrap_ci.py` | **재사용** |
| ablation 러너 | `scripts/run_ablations.py` | **재사용·재구성** (A1–A4/FULL 네이밍 정렬) |
| --- 신규 필요 --- | | |
| Phase1: `corpus/tables.jsonl`, `queries.jsonl`, `corpus/serialized/{plain_markdown,json_kv,header_path}/` | 없음(데이터만 있음) | **신규** |
| `docs/RECONSTRUCTION.md` | 없음 | **신규** |
| Phase2: `bm25` 베이스라인 (k1×b grid 튜닝) | 없음 (rank_bm25 미설치) | **신규** |
| Phase2: `hybrid` RRF(k=60), `oracle`, `nocontext` 답변 베이스라인 | 부분(retrieval만) | **신규** |
| `results/phase2_baselines.csv` + `phase2_summary.md` | 없음 | **신규** |
| Phase3: `results/phase3_routing.csv` (A1–A4/FULL), 층화표 | 부분(ablations_summary) | **신규·재구성** |
| `docs/leakage_note.md`, `docs/VERDICT.md` | 없음 | **신규** |
| 전체 3597표 코퍼스 임베딩(dev pool 외) | dev pool 540만 존재 | **신규(GPU)** |
| `configs/*.yaml` (seed/모델/k1·b/RRF k/top-k 명시) | 일부 존재(decomposition/ablation 계열) | **신규·보강** |

---

## 6. 중복 구현 위험 점검

중복 신규 구현 계획 **없음**. 모든 검색/저장/검증/라우팅/지표 코어는 기존 모듈 재사용.
신규 작업은 (a) Phase1 코퍼스 직렬화 산출물 생성, (b) Phase2 BM25/RRF/oracle/nocontext **베이스라인 래퍼**,
(c) docs 4종, (d) 전체 코퍼스 임베딩으로 한정 — 기존 코어 로직 위에 얹는 어댑터 성격.

## 7. 결정 필요 / 다음 Phase 계획

- **코퍼스 범위 결정**: 기존 검색 eval은 dev pool=540 위에서 측정(distractor=정답표들만).
  프로토콜은 전체 3597표 코퍼스(더 큰 distractor 풀)를 요구. → Phase 1에서 **전체 3597 코퍼스** 구축하고
  검색 풀을 3597로 확장(더 현실적·더 어려움). GPU로 e5/bge 임베딩 재계산.
- 다음(Phase 1): 3597표 → `corpus/tables.jsonl` + 3직렬화 + `queries.jsonl`(원본 정렬 그대로) + `RECONSTRUCTION.md`,
  orphan=0 검증.

## 9. 진단 트랙 — 전처리×복잡도 (2026-06-15, artifact 보고서 기반)
새 프레임: "검색 전 표 전처리 효과 × 표 복잡도(flat↔hierarchical) 통제 비교" (연구 공백 정면).
- [x] flat 데이터셋 OpenWikiTable 24,680표 재구성 (`scripts/diag_build_flat.py`, orphan=0)
- [x] C0(raw)/C1(+meta)/C2(+schema·header-path) 직렬화 양 축 (`scripts/diag_serialize.py`)
- [x] BGE-small 임베딩+검색평가+paired bootstrap (`scripts/diag_embed_eval.py`)
- [x] 층화 분해 (`scripts/diag_stratify.py`)
- [x] **1단계 진단 → `docs/DIAGNOSIS.md`**, **2단계 방법론 → `docs/METHOD_PROPOSAL.md`**
- 핵심수치: 메타데이터 R@1 이득 flat **+0.599** vs hier **+0.178**(3.4×). hier 천장 0.373(<flat 0.864 절반).
  계층 헤더경로 평탄화 C2−C1=+0.018(무력). 무너지는 지점=계층×집계질의(arithmetic_agg R@1 0.30).
- [ ] C3(좌표정렬 합성질문) 생성·검증 H1–H3 → `diag_synthetic_q.py` (다음 단계)
- 산출 JSON: `results/diag_flat_test.json`, `diag_hier_dev.json`, `diag_hier_stratified.json`

## 8. 진행 로그 (2026-06-07 갱신)
- [x] Phase 0 — STATE.md, Gate 0 통과
- [x] Phase 1 — 코퍼스 3597 + 3직렬화 + queries 10672, orphan=0, RECONSTRUCTION.md → Gate 1 통과
- [x] Phase 2 검색 — phase2_retrieval.json/baselines.csv/summary.md/by_class, BM25 grid, leakage_note.md
- [x] Phase 2 답변 — phase2_answers.json (전체 dev 1671, local Qwen 4bit, top-1) → Gate 2 상한·하한 성립
- [x] Decision Gate 2·4 판정 → docs/VERDICT.md (둘 다 미트리거; 풀확대 시 dense 우위·ρ=+1.0)
- [ ] Phase 3 — 라우팅 A1–A4/FULL + paired bootstrap + 층화 (codegen 경로, 추가 LLM 실행 필요)
- [ ] Gate 1·3 판정, VERDICT.md 완성

## 9. 메인 기여 — HPIR (Header-Path Intent Resolution) (2026-06-07)
주제 확정: 쿼리를 **계층 헤더경로 IR**로 변환해 분리-스토어의 검색·답변을 동시에 끌어올리는
쿼리-이해 모듈. 두 병목(검색29%/답변71%)의 공통 원인("헤더경로 미명시")을 한 모듈로 공략.
- [x] 코어 `rag_agent/query/header_path_resolver.py` — 결정론 + 검증형 LLM 정련(인벤토리 교집합)
- [x] 단위테스트 `tests/test_header_path_resolver.py` (11) — 데이터 불필요, 전체 30 통과
- [x] 검색 하니스 `scripts/hpir_retrieval_eval.py` — raw vs HPIR확장, paired bootstrap+McNemar
- [x] 답변 하니스 `scripts/method_grounded.py --mode hpir` — naive/grounded/hpir 3-way ablation
- [x] `docs/METHOD_HPIR.md`, `configs/hpir.yaml`
- [x] `stores/vector_store.py` chromadb import 지연화(bare 환경 테스트 가능하게)
- [ ] **수치 산출(로컬 RTX 3060 Ti)**: hpir_retrieval.json + method_hpir_pc20.json → METHOD_HPIR §3 표 채우기
      (본 원격 컨테이너엔 데이터/GPU/Qwen/Chroma 부재 → 실행만 데이터 보유 머신에서)
