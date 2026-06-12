# PREP — 검색 전 표 전처리 × 표 복잡도 통제 실험 프로토콜

근거 조사: [`docs/PREP_SURVEY.md`](PREP_SURVEY.md) (선행연구·연구 공백·수치 출처).
연구 공백: **전처리 효과가 표 복잡도(flat vs hierarchical)에 따라 어떻게 달라지는지를
검색 recall@k 단위로 통제 비교한 연구는 없다.** 이 프로토콜이 그 공백을 정면으로 채운다.

---

## 1. 설계

### 독립변수 1 — 전처리 조건 (누적 사다리, `rag_agent/prep/conditions.py`)

| 조건 | 인덱싱되는 텍스트 | 선행연구 근거 |
|---|---|---|
| **C0** raw | markdown 표만 (헤더 + 행, `--max-rows`로 절단) | 베이스라인 |
| **C1** +metadata | C0 + page/section title, caption | TARGET(arXiv:2505.11545): 제목 유무가 OTT-QA BM25 R@10 0.967↔0.592를 가름 |
| **C2** +schema | C1 + 컬럼명·추론 타입·예시값 설명 | Pneuma(arXiv:2504.09207), RASL |
| **C2h** +schema-hier | C1 + root-to-leaf **헤더 경로** 명시 (계층 표 전용) | HiTab·API-assisted codegen(arXiv:2310.14687)의 경로 평탄화 |
| **C3** +synthetic Q | C2(또는 C2h) + 합성 질문 | QGpT(arXiv:2508.06168) |

조건은 **누적**(C0 ⊂ C1 ⊂ C2 ⊂ C3)이라 각 단계 Δ가 "그 정보를 추가한 한계 이득"으로 읽힌다.
포맷(markdown/CSV/HTML)은 변수에서 제외 — 파일럿에서 BGE-small+flat 조건의 병목이 아님을
확인했고, arXiv:2604.24040과의 차별점도 "포맷이 아니라 정보 증강"에 있다.

### 독립변수 2 — 표 복잡도 (데이터셋)

| 데이터셋 | 복잡도 | 코퍼스 | 질문 | 비고 |
|---|---|---|---|---|
| OpenWikiTable | flat | 24,680 | test 6,602 (기본 1,000 샘플) | `scripts/prep_build_owt.py`로 구축 |
| HiTab | hierarchical (98.1%) | 3,597 | dev 1,671 | 기존 `data/loader.py` 재사용 |

### 종속변수·통계

- **R@1 / R@5 / R@10** (주지표) + MRR. gold rank는 전 코퍼스 대상 full ranking.
- **paired bootstrap** (질문 단위 재표집 10,000회, seed=42, 95% CI) —
  C0 대비 및 직전 단계 대비 Δ. CI가 0을 제외하면 유의(`*`).
- 질문 샘플링·bootstrap 모두 seed=42 고정. 동률 점수는 gold 우대로 일관 처리
  (조건 간 비교엔 영향 없음).

### 리트리버 2종 (전처리 효과의 retriever 의존성 확인)

- `bm25` — rank_bm25 Okapi, CPU-only. 어휘 기반 상한/대조군.
- `dense` — sentence-transformers `BAAI/bge-small-en-v1.5` + 내적 full-rank
  (FAISS IndexFlatIP와 동일 결과). `--device cuda`로 로컬 3060 Ti 실행.

---

## 2. 실행 명령

```bash
# 0) 데이터 구축 (1회)
python rag-agent/scripts/prep_build_owt.py            # → data/openwikitable/*.jsonl

# 1) flat × BM25 (CPU, 원격 컨테이너에서도 실행 가능)
python rag-agent/scripts/prep_retrieval_eval.py \
    --dataset owt --retriever bm25 \
    --conditions C0,C1,C2,C3 --synth template \
    --n-queries 1000 --seed 42 \
    --out rag-agent/results/prep/owt_bm25_n1000.json

# 2) flat × dense BGE-small (로컬 GPU 머신)
python rag-agent/scripts/prep_retrieval_eval.py \
    --dataset owt --retriever dense --model BAAI/bge-small-en-v1.5 \
    --device cuda --conditions C0,C1,C2,C3 --synth template \
    --n-queries 1000 --seed 42 \
    --out rag-agent/results/prep/owt_dense_n1000.json

# 3) hierarchical × dense (로컬 GPU 머신, C2h 포함)
python rag-agent/scripts/prep_retrieval_eval.py \
    --dataset hitab --data-dir rag-agent/data/hitab --retriever dense \
    --device cuda --conditions C0,C1,C2,C2h,C3 --synth template \
    --n-queries 0 --seed 42 \
    --out rag-agent/results/prep/hitab_dense.json

# (선택) C3를 LLM 합성질문으로 업그레이드 — 캐시 jsonl 재사용
python rag-agent/scripts/prep_retrieval_eval.py \
    --dataset owt --retriever dense --device cuda \
    --conditions C2,C3 --synth llm:local:Qwen/Qwen2.5-7B-Instruct \
    --synth-cache rag-agent/results/prep/owt_synthq_cache.jsonl \
    --n-queries 1000 --out rag-agent/results/prep/owt_dense_llmsynth.json
```

단위 테스트(데이터 불필요): `python rag-agent/tests/test_prep_conditions.py`

---

## 3. 결과 — flat (OpenWikiTable test 1,000 질문, 코퍼스 24,680, seed=42)

### 3.1 BM25 (rank_bm25 Okapi, 원격 컨테이너 실측 2026-06-12, `results/prep/owt_bm25_n1000.json`)

| 조건 | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| C0 raw | 0.180 | 0.298 | 0.349 | 0.2399 |
| C1 +metadata | 0.634 | 0.831 | 0.887 | 0.7217 |
| C2 +schema | 0.627 | 0.820 | 0.876 | 0.7135 |
| **C3 +synthetic(template)** | **0.682** | **0.867** | **0.909** | **0.7638** |

paired bootstrap Δ (10k iters, seed=42, `*`=95% CI가 0 제외):

| 대비 | ΔR@1 | ΔR@5 | ΔR@10 |
|---|---:|---:|---:|
| C1 − C0 | **+0.454*** | **+0.533*** | **+0.538*** |
| C2 − C1 | −0.007 [−.023,+.009] | −0.011 [−.023,+.001] | −0.011 [−.023,+.000] |
| C3 − C2 | **+0.055*** | **+0.047*** | **+0.033*** |
| C3 − C0 | **+0.502*** | **+0.569*** | **+0.560*** |

**판독.**
1. **메타데이터가 지배적** — C1 한 단계가 R@10 +53.8pp (0.349→0.887). TARGET의
   OTT-QA 결론(제목 유무가 sparse retriever를 0.967↔0.592로 가름)을
   OpenWikiTable에서 재확인. 전처리의 1차 변수는 직렬화가 아니라 메타데이터.
2. **스키마 설명(C2)은 BM25에서 한계 이득 0** — Δ가 전 지표에서 음수지만 95% CI가
   0을 포함(ns). 컬럼명은 이미 C0 헤더행에 있으므로 타입/예시값 추가는 어휘
   매칭에 새 정보를 주지 못하고 문서 길이만 늘린다.
3. **결정론 template 합성질문(C3)조차 유의한 추가 이득** (+5.5pp R@1) — 합성질문이
   질문-문서 어휘 분포를 정렬한다는 QGpT 주장의 어휘-리트리버 버전을, LLM 없이도
   재현. LLM 합성질문(C3-llm)과 dense에서의 재현 여부가 다음 셀.

### 3.2 dense BGE-small — 로컬 실행 대기

(로컬 3060 Ti에서 §2-2 명령 실행 후 이 표를 채울 것. 기대: arXiv:2604.24040·QGpT에
근거하면 dense에서는 C2·C3의 부호가 BM25와 다를 수 있음 — 그 자체가 발견.)

## 4. 결과 — hierarchical (HiTab dev 1,671 질문, 코퍼스 3,597)

(로컬 실행 대기 — §2-3. 핵심 비교: **C2h − C2** = "계층 경로 명시의 한계 이득",
그리고 같은 사다리의 Δ들이 flat(§3)과 어떻게 달라지는가 = 논문의 메인 표.)

| 조건 | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| C0 | | | | |
| C1 | | | | |
| C2 | | | | |
| C2h | | | | |
| C3 | | | | |

## 5. 방어 논리 (lab meeting)

- *"다른 논문은?"* → DTR=구조 인코딩, OpenWikiTable=decontextualization,
  TableRAG=schema/cell 분해(단일 거대 표), QGpT=합성질문, TARGET=메타데이터.
  전부 flat 또는 단일 표 — 복잡도 축의 통제 비교는 없음 (PREP_SURVEY.md §3).
- *"너는 뭐가 다른가?"* → 동일 전처리 사다리를 flat↔hierarchical에서 paired
  bootstrap으로 통제 비교. 변수를 포맷이 아닌 정보 증강에 두는 근거는 파일럿 +
  TARGET + OpenWikiTable(BERT≈TAPAS).
- *"C2가 이득이 없는데 사다리가 무의미한 것 아닌가?"* → 그 자체가 결과: flat 표
  +BM25에서는 스키마 설명의 한계 이득이 0(ns)이고 합성질문은 유의(+5.5pp R@1).
  같은 사다리를 dense·hierarchical에 교차하면 "어떤 전처리를 어떤 리트리버·
  복잡도에 쓸 것인가"의 처방표가 완성됨 — C2h가 계층 표에서 비로소 유의해지는지가
  논문의 메인 가설.

## 6. Caveats

- C3-template는 합성질문의 하한선(어휘 중복만 추가). QGpT의 주장은 LLM 합성질문
  기준이므로 C3-llm 실행 전에는 "합성질문 무용" 결론 금지.
- OpenWikiTable 질문은 decontextualized(제목 포함)라 C1 이득이 구조적으로 큼 —
  HiTab 질문은 그렇지 않으므로 flat↔hier 비교 시 이 차이를 함께 보고할 것.
- BM25 동률 처리(gold 우대)는 낙관 편향이 있으나 조건 간 동일 적용이라 Δ에는 중립.
