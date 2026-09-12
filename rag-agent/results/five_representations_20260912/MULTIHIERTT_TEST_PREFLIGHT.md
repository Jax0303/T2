# MultiHiertt test 전용 평가 사전 확인 — 2026-09-12

현재 HiTab 다섯 표현 비교를 마무리하는 동안 자료와 평가 어댑터만 확인했다. MultiHiertt 신규 모델 평가·학습·프롬프트 선택을 하지 않았다.

## 확인된 자료

- 로컬 HF 캐시 `bevaya/MultiHiertt` dataset_info.json에는 train 7,830건과 validation 1,044건만 있다. test split은 없다.
- [공식 저장소](https://github.com/psunlpgroup/MultiHiertt)는 private test leaderboard를 안내하고, test 예측 ZIP을 [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/6738)에 제출해 최종 점수를 받도록 설명한다. 공개 페이지는 로그인 후 참가를 요구한다. 실제 제출 가능 여부까지 검증한 것은 아니다.
- 공식 README의 Google Drive 데이터 링크는 확인했지만 test 입력 파일 내용과 크기는 아직 검증하지 않았다. test 정답을 로컬에서 사용할 수 있다고 가정하지 않는다.

## 현재 코드로 곧바로 진행할 수 없는 이유

`scripts/retrieval_accuracy_mh.py`는 gold table_evidence 좌표를 읽어 문항을 분류하고 검색 성공을 채점한다. 현재 어댑터는 표 근거만 있는 문항으로 모집단을 제한하며 text_evidence가 필요한 문항을 제외한다. S3c에는 제목 대신 빈 문자열을 넣는다. 이는 전체 MultiHiertt 과제의 성능 평가가 아니며 다섯 표현의 HiTab v2 무결성 계약이 자동 적용되는 것도 아니다.

비공개 test 정답/근거 좌표 없이 로컬 검색 정확도와 답변 EM의 차이를 동일 문항에서 계산할 수 없다. 공식 서버에서 답변 점수를 얻을 수 있어도 질의별 검색 hit 판정이 자동 확보되지는 않는다. train/validation을 test라고 이름만 바꾸지 않는다.

## 후속 작업 조건

1. HiTab 비교를 공통 검색기·리더 아래의 표현 적응 실험으로 명명하고, 전달 셀 수·길이·임베딩 절단·제목/헤더 정보 차이를 공개한다.
2. MultiHiertt 공식 test 입력과 제출 경로를 확보하고, 필요한 평가지표(답변 점수 외 검색 hit·유형별 집계)를 실제 제공하는지 확인한다. 외부 제출 전 예측물과 제출 범위를 검토할 수 있게 준비한다.
3. 같은 형태의 검색–EM 감사가 필수라면 공개 test 정답과 근거 좌표를 갖춘 다른 자료가 필요하다. 사용자가 합의한 test 전용 원칙을 바꾸는 대안은 실행하지 않는다.

현재 결론: HiTab 결과 감사는 계속 진행할 수 있다. MultiHiertt test의 동일한 로컬 감사는 아직 준비되지 않았다.
