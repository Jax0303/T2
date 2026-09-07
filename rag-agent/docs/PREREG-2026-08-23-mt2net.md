# 사전 등록 — 발표된 색인 단위(MT2Net)와의 정면대결, 검정력 확보판 (2026-08-23)

**결과를 보기 전에 커밋한다.** `PREREG-2026-08-23-realhitbench.md`와 같은 규율이다.

## 왜 다시 하는가

n=100에서 우리 .610 대 MT2Net .540, **17:10, p=.2478**
(`corpus_dump_vs_cell_{s3,mt2net}_dense_512.json`). 점추정은 우리가 +.07 앞서는데
불일치 쌍이 27개뿐이라 판정이 안 된다. **진 것이 아니라 잴 수 없는 것이다.**

`hitab_dev_lookup_single`은 830건 풀에서 seed-0 셔플 후 **앞 100건만** 잘라 쓴다.
남은 730건은 버려져 있었다.

## 표본을 830건 전부로 하는 이유 — 멈출 지점을 없애기 위해

250건에서 애매하면 400, 그다음 600으로 늘리는 것이 **유의해질 때까지 늘리는 조작**이다.
**가용한 것을 전부 쓰면 더 늘릴 것이 없어 멈출 지점을 고를 여지 자체가 사라진다.**

`populations/hitab_dev_lookup_all.txt`, n=**830**. 기존 100건은 이 830건의
**부분집합임을 확인**했다(`a <= b` True). test split은 건드리지 않는다.

기대 검정력 — 100건에서 관측된 구조(불일치 27%, 우리 쪽 63%)가 참이라면:

| 표본 | 예상 불일치 쌍 | 예상 p |
|---|---|---|
| 100 | 27 | .25 (실측) |
| **830** | ~224 | **~3e-5** |

**차이가 진짜면 압도적으로 유의해지고, 가짜면 그것도 드러난다.**

## 고정한 설계 (실행 전)

| 항목 | 값 |
|---|---|
| 모집단 | `hitab_dev_lookup_all`, n=830 (동결) |
| 코퍼스 | HiTab dev, 표 424 / 셀 58,759 |
| 검색기 | dense (`BAAI/bge-small-en-v1.5`) |
| 예산 | **512 (주)**, 1024 (부) |
| arms | `flat,cell` |
| 색인 단위 | `--cell-scheme S3` (우리) 대 `--cell-scheme mt2net` |
| 리더 | `local:Qwen/Qwen2.5-7B-Instruct` — **쿼리 문자열 없이.** 기존 런과 같은 bfloat16 기본값을 쓰기 위해서다(§reader_spec 참조) |
| 채점 | `hitab_exact_match_text` |
| 검정 | cell arm McNemar, **예산 2건에 Holm 보정** |

MT2Net 템플릿은 `rag_agent/serialization/templates.py::MT2NET`이고, 논문이 공개한
예시 한 문장을 글자까지 재현하는 것을 `tests/test_templates.py`가 고정한다.
채택한 해석 4개(`MT2NET_ROW_SEP=" of "`, `MT2NET_COL_SEP=", "`,
`MT2NET_LEAF_FIRST=True`, `MT2NET_TRAILING_PERIOD=False`)는 그 파일이 PROVISIONAL로
명시하며, 논문에도 그렇게 적는다. **"MT2Net 재현"이라고 무조건부로 쓰지 않는다.**

## 예측

**Q1 (주)** — 예산 512, n=830에서 **S3(우리) > MT2Net**, Holm 보정 후 p<.05.
크기 예측 **+.03 ~ +.10** (100건 관측치 +.07 주변).

**Q2 (부)** — 예산 1024에서도 같은 방향. 크기는 예측하지 않는다.
(HiTab 곡선은 512와 1024가 둘 다 .610으로 평평하나, MT2Net 문장이 우리보다 짧아
예산 확대의 이득이 어느 쪽에 더 가는지 근거가 없다.)

**근거**: MT2Net 템플릿에는 **제목 자리가 아예 없다.** HiTab은 표의 99.3%가 제목을
가지므로, 우리 단위는 그쪽 단위가 담을 수 없는 정보를 담는다. 100건에서 이미
우리 OSC가 더 높으면서(.760 대 .750) 문맥에 담은 표는 더 적었다(2.3 대 6.1) —
같은 예산으로 더 정확히 겨눴다는 뜻이고, 이것이 우연이 아니라면 830건에서 드러난다.

## 무엇이 예측을 깨는가

| 관측 | 함의 |
|---|---|
| Q1이 유의하지 않음 | **우리 색인 단위는 발표된 것과 구별되지 않는다.** 논문에서 "더 나은 색인 단위"를 주장할 수 없고, 기여는 ①복원 ②인과 측정 ③예측 셋으로 한정된다 |
| Q1이 반대 부호로 유의 | MT2Net 형태가 우리보다 낫다. 그대로 보고하고 채택을 검토한다 |
| Q1은 맞고 Q2가 반대 | 이득이 예산에 의존한다. 조건과 함께 서술한다 |

**어느 쪽이 나오든 네 런의 수치를 전부 싣는다.** 유리한 예산만 고르지 않는다.

## 실행 (이 문서 커밋 후)

```
for SCH in S3 mt2net; do for B in 512 1024; do
  PYTHONPATH=. .venv/bin/python scripts/corpus_dump_vs_cell.py \
    --dataset hitab --population hitab_dev_lookup_all --retriever dense \
    --budget $B --cell-scheme $SCH --arms flat,cell \
    --reader "local:Qwen/Qwen2.5-7B-Instruct" \
    --out results/h2h830_${SCH,,}_$B.json
done; done
```

## 이 문서가 다루지 않는 것

`render()`가 제목이 빈 문장에 `.capitalize()`를 걸어 대소문자를 뭉개는 결함은 **여기서
고치지 않는다.** HiTab은 99.3%가 제목을 가져 이 대조에 영향이 없고, 실행 중에
렌더러를 바꾸면 순차 실행되는 런들이 서로 다른 코드를 쓰게 된다. 별도 커밋에서 다룬다.
