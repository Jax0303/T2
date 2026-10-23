# 사전등록 — 문서·표 특정 사람 판정 기준 (2026-09-26)

상태: **확정 (2026-09-26 사용자 결정). 판정 시작 전 커밋.** 판정을 시작한 뒤 바꾸는 것은 전부 §8 이탈 기록에 적는다.

## 1. 판정 파일

| 판정자용 파일 (`rag-agent/results/` 아래) | 원본 | 문항 수 | 판정 열 |
|---|---|---|---|
| `doc_identify_20260926/sample100_rater_{A,B}.csv` | `sample100.csv` (MultiHiertt train, 시드 20260926) | 100 | (가), (나) |
| `hitab_e_miss_20260926/judge40_rater_{A,B}.csv` + `judge40_tables/` 80개 | `judge40.csv` (HiTab test, 제목 접두어 문장, 표 538개 한 색인, 정답 표를 못 찾은 40건) | 40 | (가), (나), (다) |

- `sample100_rater_*` = `sample100.csv` 에서 자동 규칙 열(`question_has_company`, `question_company`, `same_as_gold_doc_company`)만 뺀 것.
- `judge40_rater_*` = `judge40.csv` 그대로.
- 빈 파일 sha256:

| 파일 | sha256 |
|---|---|
| `sample100_rater_A.csv`, `sample100_rater_B.csv` | `cd5b790735ccc0defa4130d48e37d41ddbfcffac336a2e5dca0c89071a9e504b` |
| `judge40_rater_A.csv`, `judge40_rater_B.csv` (= `judge40.csv`) | `b59d529d50faeeeb93bfc9af16bac48c67b48f072324b91b29a7edf3fcbaf913` |
| `judge40_tables/*.md` 80개를 이름순으로 이은 것 | `35fa78ab4c6d8a5f2e0dcdb04f369b0197f5e5b52533411846f07354bc868f47` |

## 2. 절차

- 판정자 2명: A(본인), B(연구실 동료). 서로의 판정을 보지 않는다. B 에게는 이 문서 §3·§4 와 자기 판정자용 파일만 준다
  (judge40 은 `judge40_tables/` 폴더 포함).
- 판정 근거는 판정자용 파일의 열과, judge40 의 경우 파일이 가리키는 표 파일만 쓴다. 검색 결과·그 밖의 표·인터넷 검색은 보지 않는다.
- (가)(다) 값은 `예` 또는 `아니오`. 빈칸을 남기지 않는다.
- 두 사람의 판정을 모두 받은 뒤 §6 일치도를 계산한다. 그다음 불일치 문항만 두 사람이 합의해 최종 파일
  (`*_final.csv`)을 만든다. 보고하는 κ 는 합의 전 값이다.

## 3. 판정 기준 — (가)(나), 두 파일 공통

질문 문장에 **글자로 나온 것만** 본다. 상식·추론으로 채우지 않는다.

- (가) **예** = 질문에 대상을 특정하는 **고유한 이름**이 하나 이상 있고, **기간**도 있다.
  **아니오** = 대상이 일반 명사뿐이다(예: revenue, total assets, the company), 또는 기간이 없다.
  - 고유한 이름 = 특정한 하나를 가리키는 이름: 회사, 사업부, 시설, 제도·계획 이름 등(사람·지역·기관·작품 이름 포함).
    여러 대상에 두루 쓰이는 말(revenue, total assets, the company, individuals, women, workers)은 일반 명사다.
  - 기간 = 연도·회계연도·분기·날짜(예: 2016, fy 2016, December 31, 2014, Third Quarter 2009).
    특정 시점이 아닌 표현(two seasons, the past 12 months, the years where …)은 기간이 아니다.
- (나) (가)가 예일 때만 적는다. 근거가 된 단어(고유한 이름과 기간)를 질문 **원문 그대로** 쉼표로 구분해 적는다.
  - 자동 검증(§5)은 쉼표로 나눈 조각을 각각 찾는다. 이름 안에 쉼표가 있으면(Bank of America, N.A.) 두 조각으로 나뉘어 찾는다.
- 예시(판정 파일에 없는 문항):

| 데이터 | 질문 | (가) | (나) |
|---|---|---|---|
| MultiHiertt | what is the growth rate in net revenue for entergy wholesale commodities in 2012? | 예 | entergy wholesale commodities, 2012 |
| MultiHiertt | What's the average of Bank of America Corporation in 2016? (in million) | 예 | Bank of America Corporation, 2016 |
| MultiHiertt | What's the increasing rate of Total assets in 2018? | 아니오 (일반 명사뿐) | |
| HiTab | how many touchdowns did roddy white finish the 2009 season with? | 예 | roddy white, 2009 |
| HiTab | what was the percentage of female lone parents that had a university degree in 2011? | 아니오 (일반 명사뿐) | |

## 4. 판정 기준 — (다), HiTab judge40 만

- 1위 표 파일(`1위 표 파일` 열) 전체를 본다. **예** = 1위 표 안에 질문이 묻는 대상·항목·기간(질문에 단위가 있으면 단위도)에
  **모두** 맞는 셀이 있고, 그 셀 값이 `정답 값` 과 같은 수다(쉼표·끝자리 0 같은 표시 차이는 무시). 그런 셀이 없으면 **아니오**.
- 뜻이 맞는 셀이 있어도 값이 다르면 아니오. 값이 같아도 대상·항목·기간이 어긋나면 아니오.

## 5. 자동 검증 (판정 뒤, 코드 판정 전에 고정)

- 스크립트: `scripts/judge_verify.py` (확인: `tests/test_judge_verify.py`).
- (가)=예 인 문항마다 (나)의 조각이 **모두** 들어 있는 검색 단위 수를 센다.
  - MultiHiertt: 단위 = train 문서 묶음 **1,105개**(본문 문단+표 내용이 같은 문서를 하나로 묶음. train 2,908문서 기준).
    글 = 본문 문단 + 표(HTML 태그 제거).
  - HiTab: 단위 = test 표 **538개**. 글 = 표 제목(색인에 쓴 제목) + 비어 있지 않은 셀마다 '행 경로 > 열 경로'.
  - 조각 찾기: 소문자·HTML 엔티티 풀기·공백 정리 뒤, 앞뒤가 영문자·숫자가 아닌 자리에서 문자열 일치.
- **검증 예** = 조각을 모두 담은 단위가 정확히 1개이고 그것이 정답 문서 묶음(정답 표). 0개·2개 이상·1개지만 정답 아님은 **검증 아니오**.
  (가)=아니오 는 **해당 없음**.
- 조각이 질문에 없으면 `질문에 모두 있음`=아니오 로 표시한다(검증은 그대로 한다).
- 실행: `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python scripts/judge_verify.py {mh|hitab} <판정.csv> <출력 stem>`
  → `<stem>.csv`(문항별 사람 판정·검증 나란히), `<stem>.json`(사람 (가) × 검증 표).
- A, B, 최종(합의) 파일 각각에 돌린다.

## 6. 일치도 (판정 전에 고정)

- 스크립트: `scripts/judge_kappa.py` (확인: `tests/test_judge_kappa.py`).
- 판정 열마다 Cohen's κ, 일치율, 2×2 표, κ 95% CI(문항 부트스트랩 2,000회, seed 20260926, 백분위).
- 대상: sample100 (가), judge40 (가)·(다). (나)는 자유 기술이라 κ 를 내지 않는다.
- 실행: `.venv/bin/python scripts/judge_kappa.py <A.csv> <B.csv> "(가)" ["(다)"]`

## 7. 보고

- 파일마다: 문항 수, (가)·(다) 예 수(A, B, 최종), κ·95% CI·일치율, 사람 (가) × 검증 표(최종 판정 기준, A·B 도 함께).

## 8. 결정·이탈 기록

- 2026-09-26 판정 시작 전: 초안의 (가) 기준(HiTab "대상·주제·기간 중 2개 이상", MultiHiertt "대상 필수 + 2개 이상")을
  사용자 결정으로 §3 의 공통 기준으로 바꿨다. 판정 전이라 이탈이 아니다.

## 9. 사후 변경

- 2026-09-26 판정자 A 가 (가)에서 **기간 조건을 적용하지 않음**(§3 의 "기간도 있다"를 쓰지 않고 판정).
- 2026-09-26 판정자 A 가 (나)를 쉼표 대신 **세미콜론**으로 구분함. `scripts/judge_verify.py` 가 쉼표와 세미콜론을 모두 조각
  구분자로 쓰도록 바꾸고, (a) 사전등록 규칙(근거 구절 단위) 검증을 이 구분자로 실행함.
- 2026-09-26 판정자 A 의 judge40 (다) 중 10건이 `판단불가: …`(예/아니오 아님).
- 2026-09-26 판정자 A 파일은 Excel 로 저장되어 judge40 `정답 값` 12칸의 끝자리 `.0` 이 빠짐(수는 같음). 다른 비판정 열은 그대로.
- 2026-09-26 사후 추가 분석 (b): 근거 구절을 단어로 나누고(BM25 토크나이저 `encoders._tokenize`) 불용어(sklearn
  `ENGLISH_STOP_WORDS`)를 뺀 뒤, 단위 글의 단어 집합에 모든 단어가 있는지로 셈(`judge_verify.py --words`).
  (a)(b) 모두 판정 예 전체와 **기간 있는 예**(질문에 앞뒤가 숫자가 아닌 연도 19xx·20xx 가 있음)를 나눠 셈.
  실행·결과: `results/judge_verify_20260926/`.
