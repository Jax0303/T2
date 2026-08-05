# 연구 구조 확정본

작성 2026-08-04. 저장소 감사 보고서(`EXPERIMENT_STATUS.md`, 2026-08-04)와 발명신고서를
대조하여 확정한 학위논문 구조. 저장소가 참조하지만 실재하지 않는
`RESEARCH_SPEC_operand_set_retrieval.md`를 대체한다.

**우선순위: 학위논문.** 학술대회 투고(ECIR 2027, 10/2 마감 등)는 부차이며,
학위논문 일정과 충돌하면 투고를 미룬다. 2026-08-05 확인.

---

## 0. 폐기 이력

| 방향 | 상태 |
|---|---|
| TSR (TATR, Row-NMS, GriTS) | 폐기 |
| 표 이미지 인지 연구 (구 Phase 1) | 폐기 |
| 2단계(Phase 1 이미지 → Phase 2 RAG) 구성 | 소멸. **폐기한 것은 이미지 기반 축이고**, 단계 분리 자체가 문제였던 것이 아니다. Phase 1이 사라져 결과적으로 단일 트랙이 됐다 |
| WTQ 벤치마크 | 폐기 |

현재 연구는 **텍스트·구조 기반 표 RAG 단일 트랙**이다.

---

## 1. 문제 정의

표를 RAG 파이프라인에 투입할 때, 집계 질의(합계·평균·비율 등)에 필요한
**피연산자 셀 전체를 검색 단계에서 회수하는 문제.**

### 1.1 관련도 검색과 완전성의 불일치

기존 리트리버는 후보 $c_i$ 각각에 대해 $s(q, c_i)$를 독립적으로 계산하고 상위 $k$개를
반환한다. 집계 질의의 정답은 셀 집합 $S^*$ 전체를 요구하며,
$|S^* \setminus \text{top-}k| \geq 1$이면 답은 확정적으로 틀린다.
개별 관련도 최적화와 집합 완전성 최적화는 서로 다른 목적함수다.

### 1.2 주 지표 — OSC (Operand-Set Completeness)

전부-아니면-무 판정. `rag_agent/eval/operand_set.py:301`

```
OSC(q) = 1  iff  gold_operands(q) ⊆ retrieved(q)
       = 0  otherwise
```

- 순위 형태: `set_recall_at_k` / `all_covered@k` / `set_em@k` (동일 개념)
- 대조 지표: `per_cell_recall`, `coverage_at_k` (부분 회수율)
- **빈 gold는 공허참으로 1 처리** → 상류 필터 필수 (HiTab dev 214건 중 7건)

### 1.3 답변 정확도 지표

| 지표 | 용도 |
|---|---|
| `hitab_exact_match` (HiTab 공식) | **논문 인용용** |
| NM (numeric match, ±2% 관대) | 내부 진단 전용. **인용 금지** |

---

## 2. 두 개의 실패 원인 (실측 확보됨)

### 2.1 표면형 충돌

`Total`, `Overall` 등 비변별 라벨이 다수 표에 반복 존재하여
BM25는 IDF가 0에 수렴하고 dense는 임베딩 공간에서 군집화된다.

| 근거 | 값 | 출처 |
|---|---|---|
| 피연산자 중앙 순위 (충돌 vs 고유, flat) | 264.5 vs 14.0 | `operand_collision_multihiertt_n300.json` |
| 동일, S2 / S3 | 62.5 vs 11.0 / 48.5 vs 12.0 | 동일 |
| Mann-Whitney 단측 p (hybrid) | 3.95e-18 | `..._significance.json` |
| 동일 (bm25 / dense) | 1.95e-19 / 1.40e-10 | 동일 |

### 2.2 독립 채점의 구조적 한계

크로스 인코더 리랭커는 후보별 정밀도를 올리는 대신 집합의 마지막 셀을 예산 밖으로 밀어낸다.

| 대조 (k=10) | 성공 변화 | gain:loss | p |
|---|---|---|---|
| flat hybrid → flat rerank | 93 → 63 | 12:42 | 5.21e-5 |
| S3 hybrid → S3 rerank | 100 → 79 | 18:39 | 7.51e-3 |

출처 `operand_collision_rerank_n300.json` (n=293, pool=100, `bge-reranker-large`)

~~**미해소 모순**~~: 문서 내 풀(`operand_collision_within_doc_n300.json`, 풀 중앙값 134셀)에서는
리랭커가 flat @50 (.8840→.9010)과 S2 @10 (.6382→.6553)을 **개선**하는 것으로 보였다.

**✅ E-A로 해소 (2026-08-05).** `results/ea_pool_size_sweep_{flat,S3}.json`,
유의성 `results/ea_crossover_stats.json` (n=293, 풀은 gold 포함·크기 간 중첩,
exact binomial sign test, family 18건 Holm 보정).

| pool | flat ΔOSC@10 | S3 ΔOSC@10 | p_holm (S3) |
|---|---|---|---|
| 50 | −.003 | +.027 | 1.00 |
| 100 | −.003 | +.027 | 1.00 |
| 200 | −.027 | −.007 | 1.00 |
| 500 | −.055 | −.079 | **.043** |
| 1000 | −.085 | −.106 | **.0012** |

**모순이 아니었다.** 리랭커의 OSC 효과는 풀 크기에 대해 단조 감소하며,
**해로운 쪽만 유의**하다(≥500). 작은 풀의 이득은 방향은 있으나 유의하지 않다
(무보정 p=.096 / .152). 문서 내 풀(중앙값 134셀)은 pool 100–200 구간,
즉 효과가 0과 구별되지 않는 영역에 있고, 거기서 보고됐던 +.017은 본 스윕에서
n.s.로 나오는 변동폭과 같은 크기다. → 부호가 뒤집힌 것이 아니라
**0 근처의 추정 두 개를 양쪽에서 본 것**이다. k=50에서는 어느 지점도 유의하지 않다.

⚠️ 위 표와 이 절 상단 표는 **직접 비교 금지**. 상단(`operand_collision_rerank_n300`)의
"pool=100"은 전체 코퍼스에서 1단계 검색기가 뽑은 상위 100개라 gold가 풀 밖일 수 있어
풀 회수 실패와 순위 실패가 섞인다. E-A의 풀은 gold를 항상 포함하므로 **순수 순위 문제**만
측정한다. 즉 E-A는 상단의 93→63을 재현하는 실험이 아니라, 그 열화 중 순위 성분이
풀 크기에 따라 어떻게 커지는지를 분리하는 실험이다.

---

## 3. 실험 구성

### 3.1 무대 분리

| 무대 | 데이터셋 | 역할 | 모집단 |
|---|---|---|---|
| **검색 필요성** | MultiHiertt 다중표 코퍼스 | 주 무대. 표 전체 투입이 불가능한 규모 | n=293 (1,203표 / 43,324셀), lookup 대조 n=203 |
| **하류 정확도** | HiTab | 구조 주입이 답변으로 전환되는지 | dev 산술집계 214 → m≥2 161 / train 1,006 → 750 |
| **직렬화 하류 효과** | RealHiTBench | 피연산자 주석 없음 → OSC 측정 불가, 답변만 | n=94 (seed 0) |

**HiTab을 검색 필요성의 근거로 사용하지 않는다.** 근거:
`e9_osc_token_budget.json`에서 dense 4,000토큰 예산 OSC .9441인데
HiTab 표 전체 토큰 중앙값이 2,995(단일)/5,991(이중)이다.
즉 표를 통째로 넣는 편이 토큰도 적고 OSC는 1.0이다.
`e5_recall_first.json`의 `E_whole_table` arm도 OSC 1.000.

**test split은 미소진.** 최종 보고용으로 보존한다.

### 3.2 조작 변인

| 축 | 수준 |
|---|---|
| 직렬화 | flat / S2 (행 단위 + 헤더 경로) / S3 (셀 단위 + 캡션 + 행·열 양축 경로) / S2_shuf (토큰 동일, 순서 파괴) |
| 검색기 | bm25 / dense / hybrid / cross |
| 예산 | k ∈ {1,3,5,10,20,50}, 또는 토큰 예산 |
| 해석기 | 어휘(`header_path_resolver`) / 의미(`header_embed_resolver`) |
| 구조 주입 | 있음 / 없음 |
| 게이트 | 있음 / 없음 |
| **풀 범위** | **코퍼스 전체 / 문서 내 / 표 오라클** ← 고정값이 아니라 변인으로 취급 |
| 솔버 | gpt-oss-120b / llama-3.3-70b / llama-3.1-8b / gpt-5.1 / gpt-4.1-mini |

### 3.3 시스템 구성과 기여 구분

| 발명신고서 | 코드 | 논문 위치 |
|---|---|---|
| ① 헤더 구조 자가복원 | `reconstruct/header_grid.py` | **구현 기반** (3장). 선행 연구 인용 후 "표준 설정을 따른다" |
| ② 셀 단위 색인 | `serialization/caption.py` | **구현 기반** (3장). 동일 |
| ③ 구조 기반 보완검색 | `retrieve/structure_attr.py`, `header_enum.py` | **기여** (4장) |
| ④ 완전성 판정 | `retrieve/completeness_gate.py` | **기여** (4장). 단 5.2 참조 |
| (추가) 헤더 경로 해석기 | `query/header_{path,embed}_resolver.py` | 신고서 미기재. 논문에는 포함 |

①②를 기여로 서술하면 방어 불가능하다. 대응 선행 연구:

- Oguz et al. (2022), *UniK-QA*, NAACL Findings, arXiv:2012.14610 — 표 셀의 문장화 후 텍스트 리트리버 적용
- Chen et al. (2024), *TableRAG*, NeurIPS, arXiv:2410.04739 — schema-cell retrieval
- Herzig et al. (2021), *Open Domain QA over Tables via Dense Retrieval*, NAACL, arXiv:2103.12011 — 행/열 특징 주입
- Cheng et al. (2022), *HiTab*, NAACL, arXiv:2108.06712 §2.6 — 계층 헤더 트리를 휴리스틱으로 추출, 표본 100건 중 94% 정확
- Cao et al. (2026), *Orthogonal Hierarchical Decomposition (OHD)*, arXiv:2602.01969 — 행·열 양축 경로로 셀 계보 복원. 단 **셀이 Row Header/Column Header/Data로 사전 분류된 입력을 전제**
- Jiang, Vitagliano, Hameed & Naumann (2022), *Aggregation Detection in CSV Files*, EDBT — 집계행 검출. 키워드 방식이 실제 합 집계의 약 60% 포착이라고 보고
- Asai et al. (2024), *Self-RAG*, ICLR, arXiv:2310.11511 / Jiang et al. (2023), *FLARE*, EMNLP, arXiv:2305.06983 / Yan et al. (2024), *CRAG*, arXiv:2401.15884 / Joren et al. (2025), *Sufficient Context*, ICLR, arXiv:2411.06037 — 검색 충분성의 실행 시 신호화

**남는 공백**: 집계 검출 결과를 오프라인 색인의 구조 속성으로 물질화한 뒤
검색 후보 집합에 합집합 주입하는 용도로 사용한 보고는 확인되지 않는다.
데이터베이스 커뮤니티(EDBT/VLDB)의 집계 검출과 IR/NLP의 표 검색이 분리되어 있다는 사실이 근거.
①은 기여가 아니라 **OHD류 방법이 전제하는 입력을 주석 없이 생성하는 필요조건**으로 서술한다.

### 3.4 다음 실험

| ID | 내용 | 해소 대상 |
|---|---|---|
| ~~**E-A**~~ | ✅ **완료 2026-08-05.** 풀 {50,100,200,500,1000} × {flat, S3} × {bm25,dense,hybrid,cross}, `scripts/ea_pool_size_sweep.py` + `scripts/ea_crossover_stats.py`. 결과는 §2.2. 전체 코퍼스(43k셀) 지점은 293질의×43k = 12.7M 쌍이라 미시도 — 스윕이 구간을 괄호친다 | 2.2의 모순. 논문 핵심 그림 |
| **E-B** | 메인 실험 3종 seed 고정 후 재실행 (`operand_collision_multihiertt`, `resolver_osc_matched`, `pipeline_osc_asdescribed`) | 재현성 부채 |
| **E-C** | 문장 템플릿 통일 후 길이별(short/medium/long) OSC 비교 | 3종 템플릿 공존 + 배포 설정이 최저 정확도 |

---

## 4. 근거 확보된 주장

| # | 주장 | 근거 |
|---|---|---|
| 1 | 충돌 라벨 피연산자는 순위가 유의하게 밀린다 | p=3.95e-18 (hybrid) |
| 2 | 헤더 경로 포함 직렬화가 OSC를 개선한다 | flat→S3 @50, 141→163, p=3.13e-4 (Holm 보정 후 4.06e-3, 수작업 확인) |
| 3 | **리랭커가 코퍼스 규모에서 OSC를 낮춘다** | p=5.21e-5 / 7.51e-3. **+ E-A(§2.2): 순수 순위 조건에서도 풀 크기 단조 감소, pool 500/1000에서 Holm 후 유의(.043 / .0012)** |
| 4 | 이득은 조상 헤더 **단어**에서 오고 계층 **순서**의 기여는 0 | hybrid에서 S2와 S2_shuf가 소수점 4자리까지 동일 (.6382 / .9283). cross에서만 +.024 |
| 5 | 구조 주입이 HiTab 답변 정확도로 전환된다 | 공식 EM .4423→.5192, McNemar p=.03857 (104/161) |
| 6 | 의미 해석기가 어휘 해석기 대비 OSC 우위 | train n=1,006 전 구간 유의 (p=.000477~.006224), 예산 매칭 대조 기준 |
| 7 | 복원 손실은 작다 | S2_gold .4060 vs S2_recon .3932 @10 |

주장 3이 논문의 중심이다.

---

## 5. 근거가 반대이거나 부재한 주장 (한계 章에 명시)

| # | 신고서·초기 가설 | 실측 |
|---|---|---|
| 1 | 완전성 게이트가 실행 시 제어 신호로 기능 | **k≥3에서 발동 0회** (0/214). 결과 파일 자체가 "병목은 검색이 아니라 분해 정확도"로 기록 |
| 2 | 구조 주입의 일반성 | **MultiHiertt 전이 실패.** @10/@20/@40 전부 Δ=0. 원인: 정답 피연산자 631개 중 total 행 소재 90개(14.3%) |
| 3 | 계층 순서 보존이 기여 | hybrid에서 기여 0 (주장 4) |
| 4 | HiTab에서 검색이 필요 | 표 전체 투입이 토큰 적고 OSC 1.0 |
| 5 | 리랭커 악화의 일관성 | ✅ **해소(§2.2, E-A).** 작은 풀의 "개선"은 유의하지 않다(Holm 후 1.00). 악화는 단조이고 ≥500에서만 유의. **단, 작은 풀에서 리랭커가 "무해하다"고까지는 말할 수 없다** — 검정력 부족이며 S3 +.027은 부호가 양이다. 표현은 "풀이 작을수록 효과가 0에 수렴"까지만 |
| 6 | 해석기 이득이 답변으로 전환 | 현 표본(42/161)에서 OSC 오히려 −.024, EM 동률. 표본 부족 |

게이트는 기여가 아니라 **병목 위치를 특정하는 진단 결과**로 재배치한다.

---

## 6. 재현성 부채

| 항목 | 상태 | 조치 |
|---|---|---|
| seed | 결과 179개 중 48개 기록(실측 재확인, `_smoke` 17개 포함). 메인 3종은 여전히 미기록이나 `--seed`(기본 42)와 `env` 블록은 배선 완료 → 다음 실행부터 기록 | E-B 실행만 남음 |
| 문장 템플릿 | **실질 2종.** 스크립트 내부 사본은 `caption.py` medium과 683셀 전부 완전일치(`diag/template_divergence.md`). 배포는 long이라 달라 보였을 뿐. 남는 진짜 차이는 `serialize/verbalize.py`의 값 표기(`21` vs `21.0`, 21.5%) | E-C는 길이 비교로 축소 |
| BM25 k1/b | ✅ 해소. 13개 호출부에 동일값을 복붙하는 대신 `rank-bm25==0.2.2` 핀(k1=1.5 / b=0.75 / eps=0.25가 그 버전 시그니처로 고정). 수치 불변 | — |
| **S2_shuf 비결정성** | ✅ 해소. `hash(tuple(segs))` → `zlib.crc32`(내용 파생, 프로세스·머신 무관). `tests/test_s2_shuf_determinism.py`가 `PYTHONHASHSEED` 0/1/12345 서브프로세스로 검증 | ⚠️ **순열이 바뀌므로 S2_shuf 수치는 E-B 재실행에서 이전 값과 달라진다.** 4절 주장 4는 E-B 결과로 재확인할 것 |
| 임베딩 모델 리비전 | HF revision 미고정 | revision hash 핀 |
| 융합 경로 | 가중합(alpha=0.5)과 RRF(k=60) 공존 | 논문 보고 시 어느 쪽인지 명시 |
| 실행 환경 | ✅ 해소. `rag_agent/runenv.py::run_env`가 결과 JSON에 `env.{seed,device,torch,embed_model,run_started_utc}`를 기록. 실측 환경 = **RTX 3060 Ti 8GB / torch 2.12.1+cu130 / Python 3.12.3, `rag-agent/.venv`** | 메인 3종 배선 완료 |
| 의존성 | scipy·scikit-learn 미선언 → 선언함. **bs4·lxml은 어디서도 안 쓴다**(HTML 파싱은 표준 `html.parser`, `bench/multihiertt.py:29` / `reconstruct/header_grid.py:26`) — 선언 불필요. venv의 pandas는 2.3.3으로 `<3` 제약 준수 | ✅ `pyproject.toml` 갱신 완료 |
| 유의성 검정 | 무보정 p값만 출력. Holm/BH·효과크기 없음 → E-A는 `scripts/ea_crossover_stats.py`에 Holm 내장(family 18) + ΔOSC 보고 | 나머지 스크립트에도 확산. `operand_collision_significance.py`가 다음 대상 |
| LLM 레그 | Groq TPD 한도로 다수 중단 (44/94, 52/161, 86/161, 104/161, 42/214) | `--resume`으로 완주 |

---

## 7. 지식재산 확인 사항

- 발명신고서 제출은 공개가 아니나, 학위논문의 도서관 등록·공개는 공지에 해당한다.
  출원 완료 전 논문 공개가 예정된 경우 산학협력단에 유예 가능 여부를 사전 확인할 것.
- 발명자 명의(유권환)와 학위논문 저자가 다르므로 구성별 기여 구분을 문서로 남길 것.

---

## 8. 미결 사항

| 항목 | 상태 |
|---|---|
| 학위논문 심사 일정 | 미확인 |
| `HPIR_changes.patch` 적용 여부 | 미확인 |
| `results/operand_collision_within_doc_n300.json` 재실행 필요 여부 | `NEXT.md`와 파일 내용이 상충 |
| ~~지도교수 확인 대기: 문제 범위를 코퍼스 전체로 볼지 문서 내로 볼지~~ | ✅ **해소 (2026-08-05).** 어느 한쪽으로 좁히지 않는다. 문제 범위는 **"표가 RAG에 투입되는 상황"** 자체이고, 풀 범위는 고정 전제가 아니라 **변인**이다(§3.2). → E-A(풀 크기 스윕)가 이 방침의 직접적인 실험 형태 |
