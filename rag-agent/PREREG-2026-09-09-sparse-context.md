# 사전등록: 검색된 셀을 희소 표로 전달하기 (2026-09-09)

## 동기와 범위
사용자 요청: 리더를 교체하지 않고 본 방법의 retrieved EM을 0.8까지 개선할 수 있는 방법을 구현하고 실험한다.
이 실험은 검색 문장의 어순 변경이나 제목 중복 제거만의 실험이 아니다.
검색은 셀 문장으로 그대로 수행하고, 그 결과에 이미 있는 행/열 주소를 공유하는 셀들을 읽기 단계에서 교차표로 배열한다.
가설: 서로 비슷한 계층 경로를 20개 문장에서 반복해서 읽는 것보다 행과 열을 구분해 비교하면 형제 셀 오독이 줄어든다.
이는 미검증 가설이며 새 기여나 0.8 달성을 미리 주장하지 않는다. 과거 group(제목 공유) 실패와 구별하려면 group 대조가 필요하다.

## 고정
- 기준 커밋 62f57dfb21dba8db48d79c8ef155e418182163a3.
- reader local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit, 같은 local_qwen.py.
- BASE 프롬프트, temperature=0, seed=42, max_new_tokens=64, batch=1.
- 입력: results/retrieval_accuracy/t_s3c_hybrid_records.jsonl의 동결된 20셀. 재검색/재색인/예산 변경 없음.
- 운영 인코더와 검색 정확도는 바뀌지 않는다.
- 판정: 기존 hitab_exact_match_text 그대로. 주 모집단 all, aggregation none, m=1인 기존 991개.
- 전체 채점 가능 1581개를 실행하고 나머지 590개도 별도로 보고한다. gold/m/answer는 실행 분기에 사용하지 않는다.
- 이 test는 이미 반복적으로 분석된 데이터이다. 이번 사전등록은 독립적인 미사용 test를 만들지 않으며, 결과는 그 한계를 명시한다.

## 표현
sparse: 기존 문장에 들어 있는 제목/행 경로/열 경로/값만으로 희소 Markdown 표를 만든다.
title, row, column의 표시 문자열은 원래 템플릿의 조각 그대로이다. 모델에 새 표 제목·외부 셀·raw 문서·gold 정보를 넣지 않는다.
원본 corpus 전체에서 문장 -> 보이는 필드 사전을 만들되, 같은 문장이 서로 다른 필드 분해를 허용하면 원문으로 보존한다.
같은 주소의 복수 셀은 모두 표시한다(값 중복 제거 금지). 각 셀에 원래 순위 번호를 남긴다.
표에 생기는 빈 교차점은 —(not retrieved)이며 0으로 채우지 않는다.
grouped: 제목만 묶고 원래 셀 문장의 나머지는 보존하는 대조.
MT2Net 단위에도 같은 sparse 변환을 적용할 수 있게 준비하되, 원래 없는 표 제목을 절대로 보충하지 않는다.

## 실행 순서와 판정 (새 출력 확인 이전)
1. CPU 전수 검사: 입력 20줄 보존, 필드 multiset 보존, 애매한 문장 fallback, gold/질문을 변환기가 받지 않음.
2. 원래 순서 첫 50개 scored 질의에서 original reader 재현성 확인.
   기존 예측과 판정이 전부 같으면 저장된 baseline을 사용한다.
   다르면 현재 환경의 전체 original baseline을 새로 실행한다. 낡은 baseline과 차이를 새 방법 효과로 부르지 않는다.
3. s3c sparse를 전체 1581개에 실행. 실행 오류는 오답으로 감추지 않고 중단/재개하며 미완료 표시한다.
4. 주지표 delta가 양수이면 grouped와 MT2Net sparse를 전체 1581개에 추가 실행해 기전을 분리한다.
   delta가 0 이하이면 개선 실패로 판정하고 추가 GPU 실험은 하지 않는다(자원 절약 규칙).
   이 gate는 독립적인 확증 검정이 아니므로 후속 대조도 탐색적 결과로 보고한다.
5. 주지표 EM >= 0.8(>=793/991)이 목표 달성. 그보다 낮으면 목표 미달.
   방법 개선 지지 조건: delta >= .02 AND paired exact McNemar p<.05.
   baseline과 group을 모두 비교하면 두 비교에 Holm 보정한다. 95% paired bootstrap CI도 보고한다.
   어느 결과든 남긴다. 같은 test에서 renderer, budget, prompt를 바꾸어 성공할 때까지 재시도하지 않는다.

## 예측
사전 기대값 s3c sparse EM .75, 넓은 예상 범위 [.69,.81]. .8은 도전 목표이며 보장이 아니다.
grouped보다 sparse가 +.02 이상 높을 것으로 예측하되 실패하면 공간적 배열의 이득 주장을 하지 않는다.
다른 방법에도 같은 정도로 이득이 생기면 일반적인 context rendering 효과로 보고하고 본 방법만의 효과로 주장하지 않는다.

## 산출물
실행/검사 코드 scripts/sparse_context_experiment.py, rag_agent/serialization/sparse_context.py,
tests/test_sparse_context.py.
별도 results/sparse_context_v1/에 source/config hashes, preflight, 원본 예측, summary, 판정 저장.
기존 results/retrieval_accuracy 산출물과 TABLES.md는 덮어쓰지 않는다.
