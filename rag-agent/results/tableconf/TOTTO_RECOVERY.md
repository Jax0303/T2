# ToTTo 페이지 제목 복원 가능성 — **가능** — 2026-09-02

`results/titlemode/VERDICT.md`가 기각되면서 나온 실제 원인("질문이 묻는 개체명이
코퍼스에 없다")에 대한 후속 확인. 개입은 하지 않았다. 검색 실험 미실행.

## 매핑 규칙 (검증됨)

HiTab 표 id `<배치>_totto<N>-<k>` → **ToTTo `totto_train_data.jsonl`의 N번째 줄(0-based)**.

원본: `https://storage.googleapis.com/totto-public/totto_data.zip` (2.3GB 압축 해제,
train 120k줄). 스크래치패드에 받았고 저장소에는 넣지 않았다.

### 검증

HiTab의 totto 표 **1,851개 전부**에 대해 ToTTo 줄의 셀 값 집합과 HiTab `texts`의
셀 값 집합을 대조했다.

| 셀 겹침 | 표 수 |
|---|---:|
| 1.0 | 656 |
| 0.9 | 345 |
| 0.8 | 688 |
| 0.7 | 145 |
| 0.5~0.6 | 16 |
| **0.5 미만** | **1** |

**1,850 / 1,851 = 99.95%가 0.5 이상.** 1.0 미만인 것은 HiTab이 셀을 정제한 결과이지
매칭 오류가 아니다. 매핑은 확정으로 본다.

## 복원되는 정보

`results/tableconf/totto_page_titles.json` (1,851개, 이 확인에서 생성).
표당 `page_title`, `section_title`.

HiTab이 `title`로 쓰는 것은 ToTTo의 **section_title**이다 (`career statistics`).
**page_title은 버려졌고, 그것이 개체명이다** (`Ian Miller (footballer, born 1955)`).

## 페이지 제목이 질문과 맞는가 — 맞는다

dev 830건 중 gold 표가 totto 출신인 것 **86건**, 그중 제목 겹치는 표가 **72건**.

| | |
|---|---:|
| 페이지 제목 토큰 중 질문에도 나오는 비율 (중앙) | **.500** |
| 제목 전체가 질문에 나옴 | 26건 (30.2%) |
| 하나도 안 나옴 | 14건 (16.3%) |

| 질문 토큰 중 표 텍스트에 없는 비율 (중앙) | |
|---|---:|
| 페이지 제목 넣기 전 | **.667** |
| 페이지 제목 넣은 후 | **.429** |

```
Q: how many goals did ian miller score a total goals from 1975-78
   페이지: Ian Miller (footballer, born 1955)
Q: how many yards did matthew stafford complete in 2014?
   페이지: Matthew Stafford
Q: in 1984-85, how many goals did o'callaghan make for stoke ...
   페이지: Brendan O'Callaghan
```

## 기대 상한 (미리 못박음, 산술이지 측정이 아님 — MEASURED: NO)

- 영향받는 dev 질의 86건(10.4%), 그중 제목 겹침 72건.
- 그 72건의 표 오인율 .4533 → 실패 약 33건.
- **표를 맞혀도 표 내부 1등이 되는 비율은 .509** (`results/tableconf/VERDICT.md`).
- 33 × .509 ≈ **17건 → R@1 +.020.**

**+.02가 현실적 상한이다.** 앞서 적은 표 축 전체 천장 +.0663의 3분의 1 이하다.
16.3%의 페이지 제목은 질문과 토큰을 하나도 공유하지 않으므로 상한은 더 내려간다.

## 남은 함정

- **a0은 페이지 제목 없는 문장으로 학습됐다.** 인덱스만 바꾸면 학습/추론 불일치가
  섞인다. `PREREG-2026-09-02-title-collision.md` 1단계와 같은 이유로 기성품 인코더
  대조가 먼저 필요하다.
- **test 셋의 totto 비율 미측정.** dev의 10.4%가 test에서도 같은지 모른다.
- ToTTo 원본 2.3GB는 저장소에 넣지 않았다. 재현하려면 다시 받아야 한다.
  복원된 제목만 `results/tableconf/totto_page_titles.json`으로 남겼다.
