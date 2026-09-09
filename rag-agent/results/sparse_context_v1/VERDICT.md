# 고정 리더·고정 검색 문맥 실험

원본 코드 기준: 62f57dfb21dba8db48d79c8ef155e418182163a3.
리더: Qwen2.5-7B-Instruct, NF4 4bit, seed 42, greedy decoding, max_new_tokens 64.
검색된 셀 20개와 채점기는 동일하다. 검색 단계는 다시 실행하지 않았다.
수치 출처: 이 폴더의 COMPARISON.json, BASELINE_CHECK.json, PREFLIGHT.json과 각 실행 JSONL.

## 실행 검증

- 원본 50문항의 예측·정오가 보관본과 모두 일치: True.
- 이는 50문항 재현 검사이며 전체 기준선을 새로 실행한 것은 아니다.
- 각 검색 arm의 전체 1581문항에서 입력 20셀 보존 검사 통과.
- 모호한 문장은 원문으로 남기고 해당 질의를 제외하지 않는다.
- 사전등록 후 test에서 측정했으나, 이 test 자체는 과거에도 반복 분석되었다. 독립적인 신규 test 성능으로 주장하지 않는다.

## 결과

| 조건 | n | 기준선 EM | 새 EM | 순 증감 | 회수 | 상실 | McNemar p | Δ 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| s3c_sparse | 991 | 0.7164 | 0.7275 | +11 | 80 | 69 | 0.41274 | [-0.0131, +0.0353] |
| s3c_grouped | 991 | 0.7164 | 0.7185 | +2 | 45 | 43 | 0.91519 | [-0.0172, +0.0202] |
| mt2net_sparse | 991 | 0.5631 | 0.5530 | -10 | 79 | 89 | 0.48757 | [-0.0353, +0.0161] |

## 판정

주지표 721/991. 목표 0.8(793/991): 미달.
원래 기준선 대비 Δ≥.02 및 McNemar p<.05 조건: 미충족.
공간적 배열의 고유한 효과는 grouped 대조와, 방법 간 비교는 같은 renderer를 적용한 MT2Net 대조로 판단한다.

## 전체 모집단 (제외 없이 별도 보고)

| 모집단 | n | 기준선 정답 | 새 정답 |
|---|---:|---:|---:|
| lookup_single | 991 | 710 | 721 |
| lookup_multi | 38 | 19 | 18 |
| arithmetic | 216 | 26 | 34 |
| header_answer | 336 | 142 | 145 |
| all_data | 1245 | 755 | 773 |
| all_scored | 1581 | 897 | 918 |

## 제목 공유 대조

sparse−grouped Δ=+0.0091, Holm p=0.82549.
배열 효과 판정: 미지지.

## 같은 표현을 양쪽에 적용한 비교

S3c EM 0.7275, MT2Net 단위 EM 0.5530, 격차 +0.1746.
각 방법의 개선 폭: S3c +0.0111, MT2Net -0.0101.
개선 폭의 차이는 기술 통계다. 별도의 상호작용 검정 없이 본 방법만의 이득이라고 주장하지 않는다.

## 재현

실험 사전등록: ../../PREREG-2026-09-09-sparse-context.md
실행기: ../../scripts/sparse_context_experiment.py
변환기: ../../rag_agent/serialization/sparse_context.py
검사: ../../tests/test_sparse_context.py
기존 결과와 TABLES.md는 수정하지 않았다.
기존의 gold 제목 버그는 이번 retrieved-vs-retrieved 비교에 사용하지 않았다. gold/oracle 재측정은 별도 미완료 작업이다.
