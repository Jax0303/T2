# 사전 등록 — 후보 12개를 거르는 두 번째 신호 (2026-08-28)

**EM을 한 건도 보기 전에 커밋한다.** 모든 수치는 커밋된 결과 파일에서 나왔고 출처를 함께 적는다.

## 0. 왜 이 개입인가 — 오라클 두 개가 같은 곳을 가리킨다

| 오라클 | 하는 일 | ΔEM | 출처 |
|---|---|---|---|
| `goldcell` | 정답 셀 문장 **하나만** 남긴다 | **+.1044** (.6036 → .888) | `results/reading_ceiling_verdict.json` |
| adaptive 꼬리 절단 (S2) | gold **아래** 꼬리만 자른다 | **+.1157** | `results/cell_budget_policy_verdict.json` |
| 그것을 검색 신호로 예측한 예측기 | `cv50`·`gap12` 등 특징 7개 | **+.0000** (53:53, p=1.0) | 같은 파일 |

**예측기가 정확히 0을 받은 것이 진단이다.** 자를 자리를 정하는 데 쓴 특징이 전부
bi-encoder 코사인의 파생물이다. 코사인이 이미 틀린 순서로 준 것을 코사인 통계로 되돌릴 수 없다.

오답 분해가 같은 말을 한다 — OSC=1 오답 160건 중 **56.9%가 정답 표 *안*의 다른 셀**이다
(`diag/errors_hitab_s3c_512.xlsx`). 표 단위 장치(`cascade`, `--max-context-tables`,
`capped`)가 전부 죽은 것도 이 때문이다: 혼동이 표 사이가 아니라 표 안에서 일어난다.

→ 필요한 것은 **문맥에 이미 들어온 후보 12~14개에 대한, 코사인이 아닌 두 번째 신호**다.

## 1. 개입 — `cellfilter`

`cell` arm이 만든 문맥을 그대로 받아서, 각 문장에 cross-encoder 점수를 매기고
**상위 3개만 남긴다.** 그 외에는 아무것도 하지 않는다.

조건 세 개를 못 어긴다. 어기면 이미 기각된 arm의 재탕이다:

* **재랭킹이 아니라 제거만.** §C-1·C-2에서 죽은 것은 풀 100~1000을 **재정렬**한 것이고
  (`RESULTS.md` §C-2: pool 500 −.079, pool 1000 −.106, Holm 유의), 정답을 밀어내서 OSC가
  깎였다. 여기서는 순위를 바꾸지 않고 **버리기만** 한다.
* **비운 예산을 다시 안 채운다.** `capped`(표당 3셀)가 죽은 이유가 이것이다 — 빈 자리를
  새 표가 채워 문맥의 표가 3.0 → 9.1로 늘었다(`CLAUDE.md` §기각된 가설). 문맥은 짧아진
  채로 둔다. 천장 `goldcell`이 **48토큰**으로 .888을 낸 것이 근거다.
* **후보 생성은 건드리지 않는다.** 색인·검색기·예산·스킴 전부 커밋된 런과 동일.

모델은 **`BAAI/bge-reranker-large`** — §C-1이 랭커로 써서 죽인 바로 그 모델이다.
실패하든 성공하든 모델 탓으로 돌리지 못하게 같은 것을 쓴다. 58,759셀에 못 돌린다는
MT2Net 기각 사유(`CLAUDE.md` §MT2Net과의 관계)는 여기 해당하지 않는다 — 질의당 14쌍이다.

## 2. 손익분기를 먼저 적는다 — 이 개입은 얇다

OSC=1인 655건에서만 움직인다(OSC=0에서 버리면 어차피 오답이다). 정답 문장을 잘못 버릴
확률을 x, 살아남았을 때의 정화된 EM을 U라 하면

```
ΔEM|OSC=1 = (1−x)·U + x·.034 − .7557        (.034 = OSC=0 조건부 EM, .7557 = 현재)
손익분기:  x < (U − .7557) / (U − .034)
```

`U = .888`(= `goldcell`, 방해 0)이면 **x < .155**, 보수적으로 `U = .80`이면 **x < .058**이다.
**즉 정답 유지율이 최소 94%는 돼야 안전한 내기다.** 이 계산을 실행 전에 박아두는 이유는,
결과를 보고 keep-k를 고르는 것이 이 저장소가 금지하는 짓이기 때문이다.

## 3. T0 — CPU에서 먼저 건다. 통과 못 하면 GPU를 안 태운다

HiTab dev `hitab_dev_lookup_all` n=830, **dense α=1.0**, S3c, 512.
커밋된 `results/s3c_hitab_512_records.jsonl`의 문맥을 `scripts/error_worksheet.py
--also-correct`로 재생하고(재생 불일치 0건 가드가 이미 붙어 있다), 그 문장들에만
cross-encoder를 돌린다. LLM 없음.

bi-encoder 기준선: 코퍼스 R@1 = **.4458** (`diag/ceiling_hitab.json`), OSC=1 조건부로는
.4458/.7892 = **.565**. 문맥은 top-k 접두이므로 "코퍼스 1등"과 "문맥 동료 중 1등"은 같다.

| | 무엇 | 예측 |
|---|---|---|
| **T0-A** | CE 1위가 정답 문장인 비율 (OSC=1, n=655) | **≥ .75** (기준선 .565 대비 +.19) |
| **T0-B (관문)** | 정답 문장이 CE 상위 3 안에 드는 비율 | **≥ .95** |
| **T0-C (통제)** | 문맥 재생 불일치 | **0건.** 1건이라도 나오면 런 폐기 |
| **T0-D** | T0-A − .565 (신호 독립성) | **≥ +.10** |

### 판정 규칙 (결과 보기 전에 정한다)

| 관측 | 다음 행동 |
|---|---|
| T0-B ≥ .95 **and** T0-D ≥ +.10 | keep-3으로 T1(GPU) 진행 |
| .90 ≤ T0-B < .95 | keep-5로 **한 번만** 후퇴해 재측정. 그것도 .95 미만이면 중단 |
| T0-B < .90 | **축 종료.** "필터도 코사인 이상을 못 본다"로 기록하고 GPU를 안 태운다 |
| T0-D < +.10 | 신호가 독립이 아니다. keep-k와 무관하게 **중단** |

## 4. T1 — 답변 레그 (GPU, 홈 머신)

같은 모집단·같은 리더 스펙(`local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit&dtype=float16`)
·dense·512·S3c. **hybrid는 섞지 않는다** — 별개의 축이고, 섞으면 어느 쪽이 벌었는지 못 가른다.

| | 예측 | 판정 |
|---|---|---|
| **C (통제)** | `cell` = **.6036**, `flat` = **.1602** 재현 | 다르면 T1 해석 전에 원인부터. `flat`은 템플릿을 안 읽으므로 움직이면 리더·환경 교란 (불일치 ≤5 · p 비유의까지 허용) |
| **T1-A (주)** | `cellfilter` EM **.64 ~ .68** | 짝지음 exact binomial p<.05 **and** Δ ≥ +.02면 채택 |
| **T1-B** | OSC **≥ .75** (현재 .7892에서 4점 이내 손실) | .75 미만이면 필터가 정답을 너무 버린 것 — §2의 x가 관문을 깬 것이므로 기각 |
| **T1-C** | 문맥 토큰 509 → **250 이하** | 짧아지지 않으면 필터가 일을 안 한 것 |

**T1-A가 .68을 넘으면 예측 실패로 적는다.** 천장이 .708(= .6036 + .1044)이고 실현 가능한
이득은 오라클보다 작아야 정상이다. 넘으면 오라클보다 잘한 것이므로 재확인부터 한다.

## 5. 무엇이 예측을 깨는가

| 관측 | 함의 |
|---|---|
| OSC는 유지되는데 EM이 떨어짐 | 방해 셀이 값을 치른다는 §읽기 천장의 진단과 모순. 문맥이 짧아진 것 자체가 손해라는 뜻이므로 프롬프트(다중 셀 전제)와의 상호작용부터 본다 |
| T0은 통과했는데 T1-A가 null | 필터가 고른 3개가 `goldcell`(1개)과 다른 체제. keep-1로 한 번 내려가 보고, 그것도 null이면 종료 |
| AIT-QA에서 부호가 뒤집힘 | AIT-QA는 정답 문장이 **바이트 동일한 질의가 19.1%**다(`diag/ceiling_aitqa.json`). CE도 구별 못 하므로 여기서는 애초에 기대하지 않는다 — T3는 T1 통과 후에만 |

## 6. 실행

```
# T0 (CPU, 이 머신)
PYTHONPATH=. python3 scripts/error_worksheet.py \
    --records results/s3c_hitab_512_records.jsonl --also-correct \
    --dataset hitab --population hitab_dev_lookup_all --cell-scheme S3c \
    --retriever dense --budget 512 --out diag/ctx_hitab_s3c_512_all.xlsx
PYTHONPATH=. python3 scripts/candidate_filter_t0.py \
    --contexts diag/ctx_hitab_s3c_512_all.xlsx \
    --reranker BAAI/bge-reranker-large --keep 3 \
    --out diag/candidate_filter_t0_hitab.json

# T1 (GPU, 홈 머신) — T0 통과 후에만
PYTHONPATH=. python scripts/corpus_dump_vs_cell.py \
    --dataset hitab --split dev --population hitab_dev_lookup_all \
    --retriever dense --budget 512 --cell-scheme S3c \
    --arms flat,cell,cellfilter \
    --reader 'local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit&dtype=float16' \
    --out results/cellfilter_hitab_512.json
```

`scripts/candidate_filter_t0.py`와 `--arms cellfilter`는 아직 없다. **T0 스크립트만 먼저
쓴다** — T1 arm을 T0 판정 전에 배선하면 결과를 보고 규칙을 고르게 된다.

## 7. 이 문서가 닫는 것

T0-B가 관문을 못 넘으면 **"문맥 구성으로 EM을 올린다"는 축이 닫힌다.** `goldcell`의 +.104는
천장으로만 남고, 논문은 그것을 "실현 불가능한 상한"으로 서술한다. 그 경우 남은 것은 리더뿐이고,
`RESULTS.md` §읽기 천장이 이미 그 문장을 준비해 두었다.
