# 사전등록 — 인코더 교체 강건성 검사 (bge-base → bge-large)

작성 시각: 2026-09-09, **실행 전**. 사용자 결정: 선택지 A(강건성 검사)만 한다.

## 0. 이것은 점수 올리기가 아니다

인코더는 12개 arm 전부가 **공유하는 통제 변인**이다 (`BAAI/bge-base-en-v1.5`,
α=0.7, 예산 20 — 각 arm 의 `*.json` 에 기록돼 있다). 우리만 바꾸면 표 1·2b 의
"조작 변인은 색인 단위 하나"라는 전제가 깨진다. 그러므로 **전 arm 을 같이 바꾼다.**

검정하려는 주장은 하나다:

> H1. 본 방법과 baseline 사이의 격차는 **bge-base 때문이 아니다.**
> 인코더를 키워도 부호와 유의성이 유지된다.

**점수가 오르는 것은 결과가 아니다.** 다 같이 오르면 논문 문장은 한 글자도 안 바뀐다.
**baseline 이 우리보다 더 오르면 그것도 그대로 보고한다** — 이 문장을 실행 전에
적는 것이 이 설계의 핵심이다.

## 1. 조작 변인

`--embed-model BAAI/bge-large-en-v1.5` 하나. 335M vs 110M, 같은 계열이라
쿼리 접두어 규칙(`rag_agent/retrieve/encoders.py: _QUERY_PREFIXES`)이 동일하게
적용된다 — 접두어 관행이 바뀌는 e5/gte 대신 같은 계열을 고른 이유가 이것이다.
학습은 없다 (§0.3). 색인 단위·템플릿·α·예산·리더·프롬프트 전부 고정.

## 2. 모집단·지표

- 표 1: 검색 정확도, 주지표 **조회 m=1 n=991**. 10개 arm 전부.
- 표 2: 답변 EM. **본 방법과 MT2Net 둘만** — 나머지 arm 은 리더 20분씩이 들고,
  격차가 구조적이라 답변 레그가 판정을 바꾸지 않는다(TableRAG·청킹은 애초에
  셀을 색인하지 않는다). 비용 대비 정보가 없는 실행은 하지 않는다.

## 3. 실행 전 값 (bge-base, 주지표 991 검색 정확도)

| arm | 현재 | 본 방법과의 격차 |
|---|---:|---:|
| **본 방법 (셀+헤더경로)** | **0.9142** | — |
| 행 단위 (우리 ablation, `t_row_hybrid`) | 0.8890 | +0.0252 |
| MT2Net 색인 단위 | 0.8153 | +0.0989 |
| TableRAG Huawei (논문 2,400자) | 0.8012 | +0.1130 |
| 고정 청킹 1,000자 | 0.7669 | +0.1473 |
| 행 단위 (RowColRetrieval 행 절반) | 0.7134 | +0.2008 |
| TableRAG Huawei (코드 1,000자) | 0.6872 | +0.2270 |
| RowColRetrieval (행×열 온전) | 0.5146 | +0.3996 |
| TableRAG NeurIPS'24 (path) | 0.1433 | +0.7709 |
| TableRAG NeurIPS'24 (leaf) | 0.0686 | +0.8456 |

## 4. 예측 (숫자를 먼저 박는다)

- 본 방법 검색 **0.925** [0.905, 0.945]
- MT2Net 검색 **0.830** [0.800, 0.860]
- 본 방법 − MT2Net 격차 **+0.095** [+0.050, +0.140] (현재 +0.0989)
- TableRAG NeurIPS'24 (leaf) **0.075** [0.05, 0.11] — 구조적 결손이라 거의 안 움직인다
- 본 방법 답변 EM **0.720** [0.690, 0.750] (현재 0.7164)

근거: 격차의 기전이 이미 밝혀져 있고 둘 다 인코더 품질이 아니다 —
(a) TableRAG·청킹은 개별 숫자 셀을 색인하지 않는다(구조),
(b) MT2Net 선형화는 표를 식별하는 제목을 문장에 담지 않는다(문자열에 없는 정보는
인코더를 키워도 복원되지 않는다).

## 5. 판정 규칙 (실행 후 바꾸지 않는다)

**강건하다** = 아래 둘을 **모두** 만족.

1. 본 방법의 검색 정확도가 9개 baseline **전부**보다 높다.
2. 각 격차의 짝지은 정확 McNemar 가 Holm 보정 후에도 p < 0.05 를 유지한다
   (현재 유의하지 않은 행은 그대로 유의하지 않아도 된다 — 부호만 본다).

하나라도 깨지면 **그 arm 에 대해 "격차는 인코더 의존적"이라고 보고한다.**
전체를 살리거나 죽이지 않고 arm 단위로 적는다.

**표 1·표 2 의 보고 운영점은 bge-base 그대로다.** 이 실행은 강건성 절(부록)에만
싣는다. bge-large 가 더 높게 나오더라도 본표를 갈아끼우지 않는다 — 그렇게 하면
지금까지의 모든 판정(사전등록 6건)이 다른 인코더 위에서 나온 값이 된다.

## 6. 재구성 검증 (실행과 동시에 확인되는 것)

arm 별 명령은 각 `*.json` 의 인자에서 복원했는데, 오래된 실행 셋은
`row_text`·`chunk_chars` 키가 없다(`scripts/retrieval_accuracy.py:555` 가 그래서
지금은 인자를 기록한다). 복원이 맞았는지는 **`n_units` 가 기존 값과 일치하는가**로
확인한다 — 단위 수는 색인 구성의 지문이다. 어긋나는 arm 은 수치를 쓰지 않고
"복원 실패"로 적는다.

| arm | 복원한 인자 | 기대 `n_units` |
|---|---|---:|
| `t_s3c_hybrid` | `--unit cell --template s3c` | 67,664 |
| `t_row_hybrid` | `--unit row --row-text sentence` | 8,530 |
| `t_row_values` | `--unit row --row-text values` | 8,530 |
| `t_rowcol_values` | `--unit rowcol --row-text values` | 12,950 |
| `t_mt2net_hybrid` | `--unit cell --template mt2net` | 67,664 |
| `t_trag_hetero` | `--unit trag_hetero --chunk-chars 1000` | 1,143 |
| `t_trag_hetero_tok` | `--unit trag_hetero --chunk-chars 2400` | 618 |
| `t_tablerag_leaf` | `--unit tablerag --tablerag-colmode leaf` | 17,425 |
| `t_tablerag_path` | `--unit tablerag --tablerag-colmode path` | 18,576 |
| `t_chunk1000` | `--unit chunk --chunk-chars 1000` | 922 |

## 7. 산출물

- `results/retrieval_accuracy/{tag}_lg.json` / `_lg_records.jsonl` (10 arm)
- `results/retrieval_accuracy/t_s3c_hybrid_lg_answer_retrieved.*`,
  `t_mt2net_hybrid_lg_answer_retrieved.*`
- 판정 `results/retrieval_accuracy/VERDICT_ENCODER.md`
