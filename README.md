# 표 RAG의 피연산자 충돌: 진단과 구조 직렬화 처방

석사논문 프로젝트. **다중 표 문서에서 집계(계산형) 질의의 피연산자 셀을 검색이 놓치는
구조적 실패 모드**를 정량 진단하고, 셀에 계층 문맥을 새기는 직렬화가 그 실패를
통계적으로 유의하게 고친다는 것을 보인다.

> **문제 한 줄**: 재무보고서처럼 표가 수십 개인 문서에서 "Total" 같은 셀은 모든 표에
> 같은 이름으로 존재한다. BM25(IDF≈0, 수천 동점) · dense(임베딩 공간에서 전부 뭉침) ·
> hybrid(둘 다 계승) 모두 **어느 표의 Total인지 구분하지 못해** 그 셀을 놓치고,
> 피연산자를 하나라도 놓치면 집계 답은 틀린다.
>
> **해결 한 줄**: 셀 텍스트에 소속(계층 경로 또는 캡션 문장)을 새기면 충돌이 사실상
> 소멸하고, 필요한 셀 전부 회수 성공률이 **45.8% → 59.3% (p=4.2×10⁻⁹)** 로 오른다.

---

## 프레임워크

교수님 지침 3단계(원본표 별도보관 → 계층 파악 → 캡션 문장 → 1테이블=1청크)를 전부
구현·비교한 뒤, 성능이 확인된 구성으로 채택한 파이프라인:

```
 raw 표 (HTML / 병합 풀린 그리드)
   │
   ├──────────────► [원본표 저장소]  벡터와 분리 보관 — 값 읽기·계산·검증 전담
   │                      ▲                  (rag_agent/stores/original_store.py)
   │                      │ 찾은 위치만 넘겨받아 값을 읽음
   ▼                      │
 계층 자가복원 ──► 셀 문장화(S3 캡션) ──► 임베딩 ──► hybrid 검색  ← "찾기" 전담
 (reconstruct/)    (serialization/caption.py)         (+ 쌍둥이 표 있을 때만 재랭크)
```

- **계층 자가복원**: 정답 트리를 읽지 않고 forward-fill로 행·열 헤더 경로를 복원
  (`rag_agent/reconstruct`). MultiHiertt는 정답 트리 자체가 없어 이것이 유일한 방법.
- **직렬화 = S3 캡션 문장**: `"For {행경로}, {열경로} is {값}."` 형태의 템플릿 문장.
  S2 트리매핑(`행경로 > 열경로: 값`)과 성능 동률 — **소속을 새기는 것 자체가 핵심**이고
  표기 형식은 부차적이라는 것이 메커니즘 규명의 일부.
- **검색 = hybrid** (BM25 + dense 융합): 세 검색기 중 전 지표 최고.
- **재랭크 = 조건부 모듈**: top-K 표 후보를 셀 단위로 재경합. 쌍둥이 표가 있는
  데이터에서 R@1 +30%, 없는 데이터에선 오히려 손해 → 진단 후 켜는 장치.

## 핵심 결과 (MultiHiertt, 쿼리 297개 / 표 1,203개 / 셀 42,715개)

세 단계 주장 전부 통계적으로 유의 (`results/operand_collision_multihiertt_n300*.json`):

| 주장 | 검정 | 결과 |
|---|---|---|
| ① 문제가 실재한다 | Mann-Whitney U (flat, 충돌 vs 일반 라벨 랭크) | 세 검색기 모두 p<1e-10 — hybrid 중앙랭크 **180 vs 16**, bm25 471 vs 35 (p=3.4e-19) |
| ② 처방이 그 셀을 끌어올린다 | Wilcoxon signed-rank (충돌 피연산자 315쌍 paired) | hybrid S3 p=2.0e-4, dense p<2e-6 (bm25는 중앙랭크 471→90이나 paired 비유의 → ③으로 주장) |
| ③ 최종 검색 지표가 오른다 | exact binomial (쿼리 단위 all-covered@50 flip) | **hybrid flat→S3: 0.458→0.593, +45/−5, p=4.2e-9** · S2: 0.579, p=1.7e-8 · bm25 전 k p<2e-6 |

추가 발견:

- **충돌은 코퍼스가 클수록 커지는 구조적 문제**: 표 240개일 때 충돌 피연산자 12.7% →
  1,203개일 때 **34.8%**. "소수 케이스" 반론이 데이터로 막힘.
- **계층 자가복원 정확도**: HiTab 정답 채점 기준 열 96.7% / 행 88.2%
  (경계 자동추정 정확도 99.7%). 표가 커져도 거의 안 떨어짐(>20행: 95.5%/87.1%).
  raw HTML 31,507개 표 근사 검증 93~94%. (`results/tree_reconstruct_*.json`)
- **1테이블=1청크 캐스케이드는 이 세팅에서 최하위** (all@50 0.28~0.30): 같은 문서의
  거의 동일한 쌍둥이 표들 때문에 표 선택 단계가 병목(정답 표 중앙값 2등, 0.001 차 역전).
  단 표만 맞으면 타 표 Total이 후보에서 구조적으로 배제되는 조건부 강점은 실재.
  재랭크로 표 찾기 R@1 0.41→0.53 회복. (`results/s3_table_chunk_baseline_multihiertt.json`)
- **원본 분리보관 덕분에 "찾기 vs 읽기" 병목이 분리 측정됨**: 표를 맞게 찾은 경우로
  한정하면 정답률이 2배(0.12→0.25~0.27) — 병목은 읽기가 아니라 찾기.
- **문장 길이(short/medium/long)는 real 데이터에서 잡음 수준** (R@1 차이 ≤0.007).
  HiTab에서 보였던 "길수록 나쁘다"는 일반화 불가.

## 병행 줄기 — HiTab 합계행 주입 (완전성 천장의 다른 처방)

같은 "완전 회수(OSC)" 천장을 HiTab(정형 통계표)에서 진단하면 원인이 다르다:
이름 없는 합계 행. 처방도 다르다: 합계행 주입.

- 주입 효과 (dev n=161, 감사 후 정본): BM25 OSC 0.770→**0.876** (p=1.5e-5),
  hybrid 0.839→**0.907** (p=0.001), 손해 쿼리 0.
- 완전성 이득이 정답 정확도로 전환 (gpt-oss-120b EM 0.395→**0.500**, p=0.022,
  부분표본 86/161 병기).
- OHD 전체직렬화와 동일 지표 정면비교: 현실적 토큰예산 전 구간 유의 승
  (@8k OSC **1.000 vs 0.870**).

두 줄기의 통합 서사: **완전성 천장의 구조적 원인을 두 데이터셋에서 진단하고,
데이터셋 성격에 따라 다른 처방(HiTab=합계행 주입 / MultiHiertt=구조 직렬화)이
듣는다는 것을 보인다.**

## 진행 중 — 문장 직접읽기 답변 (코드 생성 대체)

셀 문장화의 최종 목적(교수님 피드백): 셀 의미가 자연어로 풀려 있으면 답변 LLM이
**코드 생성 단계 없이** 문장만 읽고 직접 답할 수 있다. 다음 실험: 같은 질문·같은
검색 결과에서 ⟨코드 생성 경로⟩ vs ⟨문장 직접읽기 경로⟩ 정답률 정면 비교.

---

## 저장소 구조

```
.
├── README.md                          이 파일
└── rag-agent/
    ├── rag_agent/
    │   ├── reconstruct/               계층 자가복원 (HTML→그리드, forward-fill 경로)
    │   ├── serialization/             flat / S2 header_path / S3 caption 직렬화
    │   ├── stores/                    원본표 저장소(값 읽기·검증) + 벡터 저장소
    │   ├── retrieve/                  검색·재랭크·verifier
    │   ├── extract/                   셀 추출 + 안전 AST 계산 (이전 단계, 유지)
    │   └── eval/metrics.py            R@k, MRR, nDCG, EM, NM (문헌 정렬 지표)
    ├── scripts/
    │   ├── operand_collision_multihiertt.py    ⭐ 충돌 진단+처방 실험 (핵심)
    │   ├── operand_collision_significance.py   ⭐ MWU/Wilcoxon/binomial 검정
    │   ├── tree_reconstruct_{hitab,multihiertt}.py   계층 복원 정확도 채점
    │   ├── s3_table_chunk_baseline_multihiertt.py    3단계 베이스라인 (S2/S3, 재랭크)
    │   └── ...
    ├── results/                       모든 실험 결과 JSON (per-record jsonl 포함)
    └── docs/
        ├── LAB_MEETING_BRIEF.md       ⭐ 최신 결과 정리 (발표용, 수치는 여기 기준)
        ├── PAPER_DRAFT.md             논문 초안
        └── CITATIONS_VERIFIED.md      인용 논문 원문 대조 검증 로그
```

## 재현

```bash
pip install -e rag-agent/
cd rag-agent

# 1) 계층 복원 정확도 (HiTab 정답 채점 / MultiHiertt 근사 검증)
PYTHONPATH=. python scripts/tree_reconstruct_hitab.py
PYTHONPATH=. python scripts/tree_reconstruct_multihiertt.py

# 2) 충돌 진단 + 처방 실험 (핵심 결과; 표 ~1,200개 임베딩, 1시간+)
PYTHONPATH=. python scripts/operand_collision_multihiertt.py \
    --max-queries 300 --out results/operand_collision_multihiertt_n300.json

# 3) 유의성 검정 (①MWU ②Wilcoxon ③binomial)
PYTHONPATH=. python scripts/operand_collision_significance.py \
    results/operand_collision_multihiertt_n300_records.jsonl

# 4) 3단계 베이스라인 (S2/S3 × 문장길이 × 재랭크)
PYTHONPATH=. python scripts/s3_table_chunk_baseline_multihiertt.py --scheme S3 --rerank-k 10
```

데이터: MultiHiertt는 HuggingFace `bevaya/MultiHiertt`에서 자동 로드.
HiTab 계열 스크립트는 `microsoft/HiTab` dev split 경로 필요.

## 지표 정책

문헌(FT-RAG, Topo-RAG, STAR 등)이 쓰는 표준 IR 지표(R@1/5/10, MRR, nDCG@10)와
HiTab 공식 exact match를 **항상 병기**하되, 핵심 주장 지표는 all-operands-covered
(OSC) — 집계 질의는 피연산자를 하나라도 놓치면 틀리므로 부분점수 지표(Hit Rate,
nDCG)가 가리는 실패를 드러내는 완전성 지표가 필요하다는 것이 논문 기여의 일부.
(per-cell recall 0.92인 지점에서 OSC 0.84 — 부분 recall이 집계 실패를 가림.)

## 프로젝트 이력

이 저장소는 단계적으로 발전해 왔으며 이전 단계의 코드·결과는 git 이력과
`rag-agent/EXPERIMENTS.md`에 보존되어 있다:

1. **HART (§5, 폐기)**: 벡터 점수와 헤더 정렬 점수의 α-블렌딩 — 음성 결과.
   현재의 "두 저장소 분리" 설계의 동기.
2. **적응형 라우팅 에이전트**: HiTab 하드 40쿼리에서 쿼리 분류 → 단계 라우팅 →
   심볼릭 계산(LLM은 셀 선택만, 계산은 AST). NM 0.475 vs 기존 벤치 0.250.
3. **검색 논제 + 트리매핑**: 원본 구조 직접 검색 및 트리매핑 직렬화 (10×10까지 검증,
   교수님 우려로 방향 전환).
4. **현재**: 위 프레임워크 — 충돌 진단·처방 + 합계행 주입의 통합 서사.

## License

[MIT](https://spdx.org/licenses/MIT.html)
