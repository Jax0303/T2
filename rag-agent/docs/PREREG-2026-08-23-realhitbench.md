# 사전 등록 — RealHiTBench 제목 대조 (2026-08-23)

**이 문서는 답변 EM을 한 건도 보기 전에 작성해 커밋한다.** 커밋 시각이 실험 실행보다
앞선다는 것이 이 문서의 전부다. 예측이 빗나가면 빗나간 대로 보고한다.

## 왜 사전 등록인가

지금까지 제목의 몫은 **데이터셋 사이의 비교**로만 존재했다 — HiTab 99.3% 대
MultiHiertt·AIT-QA 0%. 그 대조에서 제목은 도메인, 리더 난이도, 문장 길이, 표 크기와
완전히 교란돼 있다. 리뷰어가 "제목이 아니라 HiTab이 쉬운 것"이라고 말하면 반박할 수단이
없다.

RealHiTBench는 **한 코퍼스 안에서 38.2%의 표만 제목을 가진다.** 같은 도메인 분포, 같은
인코더, 같은 리더, 같은 예산, 같은 채점기에서 제목만 갈린다. 이 대조는 교란이 없다.

단, 돌려보고 나서 설명하면 사후 해석이다. 그래서 먼저 적는다.

## 코퍼스 (실행 전 확정)

`scripts/corpus_dump_vs_cell.py::realhitbench_corpus`, HF `spzy/RealHiTBench`
(Zhang et al., ACL Findings 2025; arXiv:2506.13405), `data/realhitbench/`.

| | 값 |
|---|---|
| 표 | 536 |
| 색인 셀 | 148,140 (값이 빈 칸 제외) |
| 질의 | **243** |
| 제목 보유 표 | 205 / 536 = **38.2%** |
| **제목 있는 표의 질의** | **100** |
| **제목 없는 표의 질의** | **143** |
| 행 경로 깊이 ≥2 | 54.1% |
| 열 경로 깊이 ≥2 | 46.1% |

**제목은 주석 필드가 아니다.** 스프레드시트 맨 위의 전폭 행("Table A-6. Time spent in
primary activities 1 and the percent of married mothers...")이라서 격자에서 복원해야
한다 — 이 논문의 기여 ①과 같은 작업이다. 복원된 제목 행은 헤더 재구성 **전에** 격자에서
떼어낸다. 남겨두면 `guess_n_header_rows`가 헤더 한 층으로 세고 `_hierarchical_carry`가
모든 열 경로 앞에 제목을 붙여서, **S2가 제목을 공짜로 갖게 되고 대조가 무의미해진다.**

**모집단은 라벨이 아니라 답 매칭이 정한다.** 답 문자열이 데이터 셀 집합으로 유일하게
풀리는 질의만 남는다(AIT-QA와 같은 규칙, 애매하면 버림). Value-Matching으로 먼저 거르면
모집단이 243 → 116으로 줄고 제목 분할이 n=45가 되어 검정이 불가능하다.

## 고정한 설계 (실행 전)

| 항목 | 값 |
|---|---|
| 검색기 | dense (`BAAI/bge-small-en-v1.5`) — 논문 실험은 dense 고정 |
| 예산 | **512 (주)**, 1024 (부) |
| arms | dump, cell, row, flat |
| 리더 | `local:Qwen/Qwen2.5-7B-Instruct` 4-bit, `--answer-mode direct` |
| 채점 | `hitab_exact_match_text` (다른 세 데이터셋과 동일) |
| 주 분석 | cell arm **S3 대 S2**, query_id로 짝지어 exact McNemar, **gold 표의 제목 유무로 분할** |
| 도구 | `scripts/paired_em_between_runs.py` |

예산 512를 주로 삼는 이유는 세 데이터셋 정면대결 표가 512라서다. 1024는 부차이며,
**둘 중 좋은 쪽을 골라 보고하지 않는다** — 512를 먼저 보고하고 1024는 부록에 둔다.

## 측정된 예측 변수 (LLM 0회, 인코더 0회)

`scripts/cell_sentence_collision.py` → `results/cell_sentence_collision.json`.
셀 문장에서 **값을 뺀 주소**가 다른 셀과 겹치는 비율. 질문은 값을 물어보는 것이라 값을
담고 있지 않으므로, 주소가 같은 두 셀은 어떤 인코더로도 구별되지 않는다.

**전체 코퍼스:**

| | flat | S2 | S3 |
|---|---|---|---|
| 주소 중복 | 85.6% | 56.6% | 50.3% |
| 표를 넘는 중복 | 41.3% | 30.4% | 20.2% |

**제목 유무로 나누면 — 이것이 예측의 근거다:**

| | flat | S2 | S3 | **S2→S3 (제목의 몫)** |
|---|---|---|---|---|
| **제목 있는 표** (셀 61,767) | 82.4% | 48.5% | **38.3%** | **−10.2점** |
| 제목 없는 표 (셀 86,373) | 85.4% | 58.9% | 59.0% | +0.1점 |
| **제목 있는 표, 표를 넘는 중복** | 30.7% | 26.8% | **10.3%** | **−16.5점** |
| 제목 없는 표, 표를 넘는 중복 | 37.3% | 27.2% | 27.3% | +0.1점 |

제목 없는 쪽의 +0.1점은 알려진 인공물이다 — 제목이 비면 `render()`가 문장 전체에
`.capitalize()`를 걸어 대소문자가 뭉갠다.

## 예측

### P1 (주 예측) — 제목 있는 표에서 S3 > S2

제목 있는 표의 질의 **n=100**에서 cell arm의 S3 답변 EM이 S2보다 높다.
**크기 예측 +.05 ~ +.15.** (HiTab에서 제목은 +.15~.18을 벌었으나, 여기는 건초더미가
148k 셀로 HiTab의 2.5배이고 제목 보유율도 38.2%뿐이므로 더 크게 잡지 않는다.)

### P2 — 제목 없는 표에서는 움직이지 않는다

제목 없는 표의 질의 **n=143**에서 |S3 − S2| ≤ .03, 부호는 0 또는 **음(−)**.
음일 수 있는 이유는 정보 없이 문장만 길어져 같은 예산에 셀을 덜 담기 때문이다.

### P3 (가장 까다로운 예측) — 두 반쪽의 차이가 갈린다

**(P1의 Δ) > (P2의 Δ).** 이것이 진짜 대조다. P1만 맞고 P3이 틀리면(양쪽 다 오르면)
제목이 아니라 다른 무언가가 오른 것이다.

### P4 — 경로의 몫은 제목과 무관하게 양쪽에 있다

두 반쪽 모두에서 cell(S2) > flat. 경로가 고쳐주는 몫이 제목 있는 표 82.4→48.5(33.9점),
없는 표 85.4→58.9(26.5점)로 양쪽에 다 있기 때문이다.

### P5 — 전체 이득은 HiTab이 아니라 AIT-QA·MultiHiertt 구간

flat→S2 주소 중복 감소가 **29.0점**으로 AIT-QA(32.7) · MultiHiertt(30.0)와 같은
구간이고 HiTab(71.4)과는 멀다. 따라서 flat 대비 전체 EM 이득은 **+.03 ~ +.08**이지
HiTab의 +.32가 아니다.

### P6 — 미리 인정하는 위험 (예측이 아니라 경고)

AIT-QA에서 `row`(행 청크)가 우리를 이겼다(.295 대 .191). RealHiTBench도 통계 스프레드시트라
행이 자연스러운 단위일 수 있다. **`row`가 `cell`을 이길 가능성을 실행 전에 명시한다.**
그렇게 나오면 검색 단위 주장이 4개 중 2개에서 깨지는 것이고, 그대로 보고한다.

### 부차 — OSC

건초더미가 148k 셀(HiTab 58.8k의 2.5배)이라 OSC 절대값은 세 데이터셋보다 낮게 나올 것이다.
**이것은 방법에 대한 주장이 아니라 무대 난이도의 기술이다.** OSC로 방법을 고르지 않는다.

## 무엇이 예측을 깨는가

| 관측 | 함의 |
|---|---|
| P1이 반대 부호 | 제목이 EM을 만든다는 주장이 이 코퍼스에서 깨진다 |
| P3이 깨짐(양쪽 다 오름) | 제목이 아니라 문장 길이·다른 요인이 원인. §제목이 전부다를 다시 써야 한다 |
| P4가 깨짐 | 경로 주장 자체가 이 코퍼스에서 성립하지 않는다 |
| P5보다 훨씬 큰 이득 | 충돌률 예측자의 보정이 틀렸다는 뜻이므로, 예측자를 다시 세워야 한다 |

## 실행 명령 (이 문서 커밋 후에 실행)

```
PYTHONPATH=. .venv/bin/python scripts/corpus_dump_vs_cell.py \
  --dataset realhitbench --retriever dense --budget 512 \
  --cell-scheme S3 --arms dump,cell,row,flat \
  --reader "local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit&dtype=float16" \
  --out results/rhb_dense_512_s3.json

PYTHONPATH=. .venv/bin/python scripts/corpus_dump_vs_cell.py \
  --dataset realhitbench --retriever dense --budget 512 \
  --cell-scheme S2 --arms dump,cell,row,flat \
  --reader "local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit&dtype=float16" \
  --out results/rhb_dense_512_s2.json

PYTHONPATH=. .venv/bin/python scripts/paired_em_between_runs.py \
  results/rhb_dense_512_s2_records.jsonl results/rhb_dense_512_s3_records.jsonl \
  --out results/rhb_dense_512_s2_vs_s3.json
```
