# METHOD_PROPOSAL.md — 2단계(방법론): 무너지는 지점을 겨냥한 학위논문 설계

생성: 2026-06-15 · 근거: `docs/DIAGNOSIS.md` (1단계 진단)

> 진단이 고정한 무너지는 지점:
> **계층 표 × 헤더 좌표를 가리키는 질의(특히 집계). 병목은 구조의 부재가 아니라,
> 구조가 질의와 정렬되지 않은 채 직렬화된다는 것.** (DIAGNOSIS §4)
> 본 문서는 그 지점을 정면으로 겨냥한 방법론을 정의한다.

---

## 1. 논문 한 줄 (thesis statement)

> **계층 헤더를 "경로의 나열"로 직렬화하는 대신, 각 헤더 좌표 (top-path × left-path)를
> 질의 형태의 자연어 명제로 변환해 인덱싱하는 *좌표 정렬형 전처리(coordinate-aligned preprocessing)*가,
> 계층 표 검색에서 무너지던 집계·좌표 질의의 recall을 회복시킨다.**

핵심 전환: **직렬화(serialize)가 아니라 질의-정렬(align).** C2(헤더경로 평탄화)가 +0.018로 무력했던 이유 —
구조를 텍스트로 폈을 뿐 질의가 쓰는 표현과 같은 공간에 두지 않았기 때문 — 을 직접 제거한다.

---

## 2. 방법 개요

표 `T`의 의미 있는 셀 좌표 `(top_path_i, left_path_j)` 각각에 대해:

1. **좌표 명제 생성**: 헤더경로를 자연어로 결합한 명제/질문 단위를 만든다.
   예) top=`record > w`, left=`sv sandhausen > 2013–2016` →
   *"How many wins (record) did sv sandhausen have during 2013–2016?"* (값 35와 함께).
   - 생성기: 무료/로컬 LLM(Qwen2.5-7B 또는 Groq llama-3.1-8b). **템플릿+LLM 혼합**으로 비용 통제.
   - **좌표 그라운딩**: 생성 단위는 반드시 자신의 (top-path, left-path, value)에 태깅 → 검색 후 셀 역참조 가능.
2. **선택적 단위화**: 모든 셀이 아니라 (a) 집계/총계 행·열, (b) 질의빈출 헤더를 우선(예산 통제).
3. **인덱싱**: 좌표 명제들을 표 본문(C1/C2)과 **함께** dense 인덱스에 넣되, 표 단위로 그룹핑(max-pool 또는 표별 best-unit 점수).
4. **검색**: 질의 → 좌표 명제 매칭 → 그 명제가 속한 표를 후보로. 계층 좌표가 질의와 같은 공간에 있으므로 정렬됨.

산출 조건명: **C3-grounded** (좌표 정렬형). 대조군: **C3-qgpt**(좌표 비그라운딩 합성질문, QGpT 재현).

---

## 3. 선행연구와의 차별 (artifact 경고 반영)

artifact는 "요약/합성질문 붙이기"는 이미 있으니 단순 재탕 금지라고 경고한다. 차별점:

| 선행 | 무엇을 | 본 방법과 차이 |
|---|---|---|
| **QGpT** (TRL@ACL 2025) | partial table에서 합성질문 생성·임베딩 | 표 *크기*에 대응, 헤더 *계층 좌표*에 그라운딩 안 됨. 본 방법은 좌표 태깅 + 계층 경로 결합 |
| **C2 헤더경로 평탄화** (본 진단) | root-to-leaf 경로 bag 직렬화 | 질의와 정렬 안 됨(+0.018). 본 방법은 경로를 *질의형 명제*로 변환 |
| **TARGET / Pneuma** | 표 단위 제목·요약 인덱싱 | 표 *전역* 서술. 본 방법은 *셀 좌표* 단위, 구조에서 파생 |
| **TableRAG** | schema/cell 분리 인덱싱(단일 거대표 내부) | 표 *코퍼스에서 표 선택*이 본 태스크. 셀 분리를 표선택용 좌표명제로 재목적화 |

→ 신규성: **"계층 좌표 → 질의정렬 단위" 변환을 표-검색(table selection) 전처리로 정식화하고,
복잡도·질의유형 축에서 통제 비교한 최초 연구.**

---

## 4. 진단이 만든 검증 가능한 가설 (반증 가능)

1. **H1 (회복)**: C3-grounded는 hier `arithmetic_agg`에서 C2 대비 R@1을 유의하게 올린다.
   (진단: 이 클래스가 천장 0.30·헤더경로 수익 ≈0 → 가장 큰 개선 여지.)
2. **H2 (복잡도 특이성)**: C3-grounded−C2 이득은 hier ≫ flat.
   (진단: flat은 C1에서 이미 0.86 포화 → 추가 여지 거의 없음.)
3. **H3 (정렬 > 합성)**: C3-grounded > C3-qgpt(좌표 비그라운딩) on hier.
   (좌표 정렬이 합성질문 *존재* 자체보다 중요하다는 본 논문의 메커니즘 주장.)
4. **H4 (음성결과도 기여)**: 만약 C3-grounded ≈ C3-qgpt면, "합성질문이면 충분, 그라운딩 불필요"라는
   QGpT 결론을 계층 표로 확장하는 음성결과로 보고(여전히 발표 가능).

---

## 5. 평가 프로토콜 (기존 자산 재사용)

- **검색**: `scripts/diag_embed_eval.py` + `diag_stratify.py` 그대로. 조건에 C3-grounded/C3-qgpt 추가.
  지표 R@1/5/10, C2 기준 paired bootstrap(seed=42, B=10,000), 층화(difficulty_class).
- **공정성**: C3 단위는 **코퍼스 전체 표**에 부여(gold만 부여 시 누수). 예산상 hier 3,597 전체에 생성
  (로컬 Qwen, 표당 top-k 좌표만 → 호출 수 통제). flat은 포화 확인용 축소 실행.
- **end-to-end 연결**: 본 repo VERDICT — HiTab 전체풀에서 검색↔답변 Spearman ρ=+1.0, 즉
  **검색 개선이 답변 개선으로 직결**. 따라서 R@1 회복분을 `scripts/codegen_eval.py`(oracle NM 상한 0.447)에 통과시켜
  end-to-end NM 향상으로 환산 → 검색-only 기여를 답변 기여로 보강.
- **베이스라인 정렬**: dense_header_path(bge-large R@1 0.490, VERDICT)와 BGE-small C2(0.373)를
  동일표에 병기해 모델·전처리 효과 분리.

---

## 6. 리스크·완화

- **생성 비용**: 표당 모든 셀 명제는 폭발 → (집계행/빈출헤더) 우선 + 표당 좌표 상한. 템플릿 우선·LLM 보강.
- **계층 baseline 희소**(artifact caveat): hierarchical 검색 전처리 선행이 적음 → C2(평탄화)·C3-qgpt를
  자체 baseline으로 세워 비교 공백을 메움.
- **2604.x 프리프린트 의존**: 비교 인용은 조건부로.
- **단일 임베더**: bge-large/e5-large 재확인을 robustness 절로.

---

## 7. 다음 실행 단계 (바로 가능)

1. `scripts/diag_synthetic_q.py` — hier 코퍼스에 C3-grounded / C3-qgpt 좌표명제 생성(로컬 Qwen, 표당 좌표 상한).
2. `diag_embed_eval.py --conditions C0,C1,C2,C3-grounded,C3-qgpt` 로 hier 재평가 + 층화.
3. H1–H3 검정표 작성 → `results/diag_hier_c3.json`, `docs/DIAGNOSIS.md`에 C3 행 추가.
4. (선택) 회복된 R@1을 codegen_eval에 통과시켜 end-to-end NM 환산.
