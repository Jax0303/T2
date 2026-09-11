# 평가 결함 수정 후 답변 재측정

기준 커밋: 6d88d4d. 이 작업은 EM 개선 개입이 아니라 수정된 검색 문맥과 답변 평가를 일치시키는 재측정이다.
HiTab test만 사용하며 학습, dev, 프롬프트 선택, 예산 변경은 하지 않는다.

## 고정 조건
- Qwen2.5-7B-Instruct, 기존 local backend NF4 4bit.
- 기존 BASE prompt, temperature 0, max_new_tokens 64, seed 42, batch size 1.
- 입력: results/audit_fix_20260910/{t_rowcol_values,t_tablerag_leaf,t_tablerag_path}_fix_records.jsonl.
- arm당 1,581문항. 주지표 991, 데이터셀 전체 1,245, 헤더답 336을 구분한다.
- 원본 S3c 50문항 예측·채점 일치 확인 후 보관 기준선 사용. 불일치하면 중단하고 새 기준선이 필요하다고 기록한다.
- 문맥 해시와 질문 집합을 검증한다. 긴 입력을 잘라내거나 실패 문항을 제외하지 않는다.
- 문항별 결과를 즉시 저장하며, 동일 설정·입력 해시인 실행만 재개한다.
- 수정된 답변 결과와 수정된 검색 기록을 함께 표 생성기에 연결한다. 완료 전 옛 답변 결과로 대체하지 않는다.

## 판정과 보고
검색 정확도와 EM을 같은 문항 집합에서 계산한다. 기존 결함 있는 arm 수치와의 변화는 버그 수정 효과이며 새로운 방법의 성능 개선이 아니다.
S3c와의 비교는 문항별 paired McNemar 및 기존 표의 Holm·bootstrap 규칙을 유지한다.
값과 데이터 판본을 추적할 수 있도록 입력·코드 해시 및 실행 환경을 저장한다.
gold-only/oracle은 특정 입력 정책의 측정값이다. 모든 고정 리더 정책의 상한으로 해석하지 않는다.
과거 test 분석 이력이 있으므로 사전 고정만으로 독립적인 확증 평가가 되는 것은 아니다.

실행: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/remeasure_audit_answers.py
