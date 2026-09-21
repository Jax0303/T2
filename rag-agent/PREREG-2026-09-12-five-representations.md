# 다섯 표현 비교 보완 — 2026-09-12

> **2026-09-21 지도교수 지시로 MT2Net을 비교대상에서 제외했다.** 이 사전등록은 실행 당시(2026-09-12)
> 원문 그대로 두되, MT2Net arm에 대한 언급을 지웠다 — 그 arm의 원 수치는 `archive/mt2net-2026-09-21/`에
> 있고, 실제 비교 결과(`results/five_representations_20260912/`)는 이 결정에 맞춰 4-arm으로 재생성했다.

목적: 기존 슬라이드의 고정 청킹 표현을 현재 v2 근거 계약으로 보완해 검색 정확도−답변 EM의 방법별 차이를 동일 기준에서 설명한다. 기존 test 결과를 본 상태의 재측정이며 새로운 독립 확증 실험이 아니다.

추가 arm: unit=chunk, chunk_chars=1000, template=s3c인 자체 행 보존·헤더 반복 Markdown 청킹. 전체 원본 시스템 재현으로 명명하지 않는다. 코드가 제공하는 제목·행/열 경로를 감사해서 보고한다.

고정 조건: HiTab test, corpus=split, BGE-base revision a5beb1e3e68b9ab74eb54cfd186867f64f240e1a, alpha=.7, 쿼리 접두어 유지, budget=20의 whole-unit-stop 정책. 큰 청크는 기존 절단 조건을 명시적으로 재현하는 embed-overflow=truncate로 실행하고 초과 건수를 보고한다.

리더: Qwen2.5-7B-Instruct 4bit revision a09a35458c702b33eeacc393d103063234e8bc28, neutral, seed=42, greedy, max_new_tokens=64. 전체 채점 1,581건 생성, 주지표 991건 및 보조 모집단 보고. 기존 결과를 덮어쓰지 않는다.

판정/보고: 파일 hash·문항·선택 셀·검색 판정·리더 조건을 검증하고 다섯 표현의 공통 hit 집합과 전체 주지표를 구분해서 보고한다. retrieval−EM=(hit_wrong−miss_correct)/n. 점수를 보고 다른 프롬프트나 채점 규칙을 선택하지 않는다. RowCol leaf와 전체 경로 진단은 구분한다. 사람 채점은 미실시로 명시한다.

MultiHiertt 전환 조건: 출처와 비교 조건 검증 통과, 구현·정보 전달 차이 공개, 실패 분해의 설명 범위 정리. 기존 모델 학습 없이 test 전용 조건 유지. 데이터에 test 정답이 없으면 임의로 train/dev로 바꾸지 않고 평가 가능 여부를 먼저 보고한다.
