# MultiHiertt 공정 비교 보고 (2026-09-28)

사전등록 `PREREG-2026-09-28-fair-mh.md`(f693c7f, 본 실행 전). 결과 폴더 `results/fair_mh_20260928/`. HiTab·MT2Net 제외.

## 1. 어떤 문제가 있었는가 (파일·함수 기준 점검)

| 점검 항목 | 확인 결과 |
|---|---|
| ④ LLM 필터가 MultiHiertt 경로에 연결돼 있는가 | **아니었다.** 필터 코드는 HiTab 전용 `scripts/fair_filter_eval.py` 만 있었고 `scripts/answer_accuracy_mh.py` 는 검색 문맥을 그대로 리더에 넣는다(조건 retrieved/gold/fulltable/rowexp). MultiHiertt 답변 표(표 5-8)는 모두 **필터 없음** 결과다. |
| ③ 후보마다 셀 ID·점수·순위가 남는가 | **셀 ID·순위가 기록에 없었다.** `mh_arms.py` doc 기록은 단위 텍스트 목록(순위 순)과 `correct`(전체 근거 포함)만 저장했다. Recall/Precision@k 를 계산할 수 없었다. 점수는 여전히 저장하지 않는다(min-max 혼합 점수, 이번에 추가하지 않음). |
| "검색 정확도"의 뜻 | 기존 `correct` = 정답 근거 셀 **전부**가 예산 20셀 문맥에 들어온 문항 비율 = **All-Evidence@예산20**. Recall 이 아니다. 원고 4.3절의 정의 문장은 이와 같으나 이름이 "검색 정확도"였다. |
| 정답 셀이 색인에 있는가 | 색인은 표의 비어 있지 않은 데이터 셀 전부(`live`). 정답 셀이 색인 밖(머리글 칸 17, 표 파싱 실패 4, 좌표 없음 2)인 23문항은 사유별로 제외·공개(`excluded_by_reason`). 분모는 2,885. |
| 임베딩·인코딩 | `BAAI/bge-base-en-v1.5`, 문서 423,473개 모두 512토큰 안(최대 188), 질의만 접두어 "Represent this sentence for searching relevant passages: ", 정규화 코사인, 전수 내적(ANN 아님), BM25 와 min-max 뒤 α=0.7 혼합. dense-only 는 이번에 돌리지 않았다(시간). |
| 리더 입력 절단 | 필터 없음 3 arm 882건에서 한도(40,960) 초과 0. 필터 후도 0 (아래). |
| 평가 표본 | 표 근거만 필요한 2,908문항(본문 근거 필요 4,922 제외) 중 채점 2,885. 답변 표본 882 = 그룹별 상한 300, seed 20260913, **처음 규칙 시절부터 고정**. 머리글 규칙·고유화가 이 표본을 보고 고쳐졌으므로(6.5절) 독립 test 가 아니다. 공식 test 는 정답 근거가 없어 쓸 수 없다. |
| 문서 군집 | 2,885문항 = 문서(표 내용 해시) 1,096개, 882문항 = 582개 문서(최대 한 문서 5문항). |

## 2. 무엇을 수정했는가

| 커밋 | 변경 | 영향 |
|---|---|---|
| 7821b46 | `mh_arms.py`: doc 기록에 `ranked_units`(순위 순 상위 20단위의 셀 좌표 "표-행-열")와 `gold_ids` 추가. 검색 논리 불변 | 재검색 3 arm 의 `correct`·문맥이 기존 기록과 문항별 **불일치 0** (2,885 × 3) |
| 7821b46, f693c7f | `scripts/mh_filter.py` 신설: 질문 + 번호 붙인 줄 → 번호. 범위 밖 번호 버리고 셈, 빈 선택은 빈 문맥(자동 대체 없음), 원문만 전달, 정답 미참조. 출력은 `answer_accuracy_mh.py` 가 읽는 기록 형식 | MultiHiertt 에 필터 단계가 처음 연결됨 |
| a474378 | `retrieval_metrics.py`: Recall/Precision/F1/Hit/All-Evidence@1,5,10,20 (셀 순위 / 단위 순위 / 예산20 전달 집합), macro | 지표 분리 |

버그로 분류할 것은 없었다(기존 수치는 재현됨). 문제는 **지표 이름·분리 부족**과 **필터 단계 부재**였다.

## 3. 결과 (같은 조건)

공통 조건: MultiHiertt train 표 근거 문항, 문서 안 검색, 머리글 v3.3u, bge-base + BM25(α .7), 예산 20셀, 필터 Qwen3-8B 4bit(비생각·greedy·64토큰), 리더 Qwen3-8B 4bit(비생각·cot·384토큰, seed 42, continuous batching 128), 채점 `multihiertt_em.mh_exact_match`(공식 포팅). 세 arm 은 **표현만** 다르다: 본 방법(셀 문장 s3c), 고정 1,000자 청크, TableRAG(Yu) 청킹 이식판.

### 3A. 필터 전 검색 (2,885문항, macro)

(`retrieval_metrics.md` 참조. 아래 표는 실행 후 채움)

### 3B. 필터 후 근거 선택 (882문항)

### 3C. 답변 (882문항)

### 3D. 진단: 정답 셀만 준 조건

## 4. 논문에서 어디까지 주장할 수 있는가

## 5. 완료 / 실행 중 / 미실행

## 6. 재실행 명령

    bash results/fair_mh_20260928/run_retrieval.sh      # 재검색 3 arm (캐시 .cache/rerun_20260926_mh)
    .venv/bin/python results/fair_mh_20260928/retrieval_metrics.py
    bash results/fair_mh_20260928/run_main.sh           # 필터 → 답변 (s3c, chunk), gold 조건, trag_hetero
    .venv/bin/python results/fair_mh_20260928/filter_metrics.py

데이터: bevaya/MultiHiertt(HF 캐시), 정답은 yilunzhao/MultiHiertt@f18473da; 색인 텍스트 sha256 s3c `ccef1b95…`; 질의 id sha256 `ed462b6d…`; 답변 표본 `results/mh_arms/cap300_20260924/cell.jsonl` 의 882 id. 모델: Qwen/Qwen3-8B revision b968826d, 4bit(bitsandbytes NF4); BAAI/bge-base-en-v1.5.
