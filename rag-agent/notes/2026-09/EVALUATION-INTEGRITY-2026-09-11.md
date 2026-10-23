# 평가 무결성 v2 — 2026-09-11

기준 main은 `6d88d4db00018fa08bc521676045b38198ae960e`이다. 작업 중 원격에 추가된
`4be96ab`, `6d88d4d`를 먼저 반영했다. main에 이미 들어간 RowCol 교집합 렌더링,
전체 문맥·gold 저장, MULTI 실험과 재측정 결과는 보존한다. 이 변경은 그 위에
기계적으로 검사 가능한 근거 계약과 남은 splitter·결과 생성기 수정을 추가한다.

이 문서와 새 manifest는 이전 문서의 “온전 재현”, “1,000토큰≈2,400자”,
“gold는 수학적 상한”이라는 해석을 대체한다. 원본 결과 JSONL은 고쳐 쓰지 않는다.

## 변경

1. `context_version=2`: 실제 문맥, 문맥 단위별 셀 좌표, 전체 gold 좌표, SHA-256을 함께
   저장한다. 저장 직전과 답변 생성 전에 셀 집합·개수·gold·hit 판정을 재검사한다.
   문맥 저장 옵션은 켜기/끄기이며 문맥을 잘라내는 두 번째 예산이 아니다.
2. RowCol은 교집합과 5개 반환값 API를 유지한다. 공식 코드 대조에서 열 헤더
   전달 누락을 추가로 발견하여 기본 문맥을 열 라벨을 포함한 Markdown으로 고쳤다.
   `--rowcol-context legacy_values`로 이전 행 값 문맥을 명시적으로 재현할 수 있다.
   기본 헤더는 leaf이며 `--rowcol-header path`는 별도 변형이다. 단위마다 셀 좌표를 붙인다. 선택한 행·열의 원문을 합쳐 보내는 구버전으로 돌아가지 않는다.
   `--rowcol-max-pairs`는 기존 200개 제한을 명시적인 인자로 노출한다.
3. Huawei에서 착안한 청크 표현의 문자 모드는 실제
   `RecursiveCharacterTextSplitter`를 쓴다. `langchain-text-splitters==1.1.2`를 고정한다.
   일반 `chunk`는 기존 행 보존·헤더 반복 구현으로 남겨 이름을 정확히 적는다.
4. 토큰 모드는 `--chunk-tokens`, `--chunk-tokenizer`, `--chunk-overlap`으로 명시한다.
   원문 offset을 사용하고 잘린 셀 값에는 완전한 근거 셀 좌표를 부여하지 않는다.
   토큰 예산은 본문 창 기준이며 `File name:` 접두어는 그 밖에 붙는다. 임베딩 길이
   검사는 접두어·특수토큰까지 포함한 실제 입력 전체를 센다.
5. BGE 입력 길이를 질의/문서 양쪽에서 캐시 읽기 **전에** 검사한다. 기본은 초과 시
   오류다. 기존 절단 조건을 재현할 때만 `--embed-overflow truncate`를 명시하고,
   초과 인덱스·개수·비율을 결과에 남긴다. 기본 모델과 예산을 몰래 바꾸지 않는다.
6. 소스·데이터·표·페이지 제목·모델 revision·실행 인자·패키지 버전을 기록한다.
   임베딩 캐시 키에 모델 설정을 포함한다. 기존 출력이나 남은 요약 파일을 덮어쓰지 않는다.
7. 새 리더 기본값 `--prompt neutral`은 표/청크/셀 문장 모두를 설명한다. 예전 `base`,
   `format`, `evidence`는 명시적으로 선택할 수 있다. 입력 토큰+출력 예산을 확인하고,
   gold 조건의 셀 개수를 검색 문맥의 셀 개수로 잘못 기록하지 않는다.
8. 결과 생성기는 `analysis/comparison_manifest.json`에 지정된 파일만 읽는다.
   질의 교집합으로 분모를 축소하거나 없는 파일을 건너뛰지 않는다. EM을 재채점한다.
   새 v2 비교는 문맥 hash·동일 데이터·검색기·리더·프롬프트 조건도 확인한다.
   주지표 991건, 데이터셀 1,245건, 헤더답 336건과 주지표 밖 유형·제외 사유를 보고한다.
9. 기본 manifest는 **과거 결과 감사용**이다. 구 RowCol의 답변을 포함한 비교 행은
   제외한다. 검색 집합 자체의 판정은 main에서 재현됐지만 구 문맥의 EM과 짝지어서는
   안 된다. Google TableRAG 힌트 문서는 셀 근거 비교에서 제외한다.
   `gold_v2` 파일을 명시적으로 선택하며, 구 gold로 돌아가는 fallback은 없다.
10. oracle 합성은 `--gold`, `--retrieved`, `--out`을 반드시 지정한다. v2 메타데이터와
    질의·리더 조건이 맞아야 합성한다. 재생성된 답변이나 성능 상한으로 표시하지 않는다.

## 실행

`rag-agent`에서 실행한다. 기존 결과와 섞이지 않도록 새 디렉터리/태그를 쓴다.

```powershell
python -m pytest tests -q
python scripts/retrieval_accuracy.py --corpus split --unit cell --template s3c --tag s3c_v2 --out-dir results/evaluation_v2
python scripts/retrieval_accuracy.py --corpus split --unit rowcol --row-text values --tag rowcol_v2 --out-dir results/evaluation_v2
python scripts/retrieval_accuracy.py --corpus split --unit trag_hetero --chunk-chars 1000 --chunk-overlap 200 --tag huawei_char_v2 --out-dir results/evaluation_v2
python scripts/answer_accuracy.py --records results/evaluation_v2/s3c_v2_records.jsonl --prompt neutral --out results/evaluation_v2/s3c_v2_answer_retrieved.jsonl
python scripts/answer_accuracy.py --records results/evaluation_v2/s3c_v2_records.jsonl --condition gold --prompt neutral --out results/evaluation_v2/s3c_v2_answer_gold.jsonl
python analysis/compose_oracle.py --gold results/evaluation_v2/s3c_v2_answer_gold.jsonl --retrieved results/evaluation_v2/s3c_v2_answer_retrieved.jsonl --out results/evaluation_v2/s3c_v2_answer_oracle.jsonl
python analysis/accuracy_tables.py --out results/evaluation_v2/historical_audit.md
```

Huawei 문자 실험은 기본 BGE에서 초과 오류가 날 수 있다. 그것이 의도한 동작이다.
기존 512토큰 절단 조건을 측정하려면 해당 실행에 `--embed-overflow truncate`를 명시한다.
긴 인코더로 변경하려면 비교군 전체에 동일 조건을 적용한 별도 manifest로 평가한다.
모델을 고정할 때는 `--embed-revision <commit>`과
`--reader 'local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit&revision=<commit>'`을 쓴다.

새 비교 표는 `analysis/comparison_manifest.v2.example.json`의 결과 파일들이 생성된 후
`--manifest`로 지정한다. 이름만 `_v2`인 과거 파일은 v2 근거 계약을 충족하지 않는다.
기존 `results/audit_fix_20260910/` 역시 과거 형식이므로 새 reader는 재생성을 요구한다.

## 이번 검증의 범위

성능 향상 가설을 test 점수로 선택하지 않는다. 회귀 테스트와 BM25-only 전수 실행은
저장·렌더링·분모 검증용이다. 문자/토큰 splitter를 통과한 전체 test 질의에서
좌표·문맥 계약이 유지되고 gold 절단이 0건인지를 확인한다. 실패 기준은 계약 불일치
1건 이상이다. 1,000토큰/200토큰 창은 논문 설정을 표현할 수 있는지 검사할 뿐,
BGE+LLM 성능 비교나 새 방법 채택 근거로 사용하지 않는다.

새 GPU 답변 EM은 여기서 생성하지 않았다. 이 변경으로 EM이 올랐다고 주장할 수 없다.
Huawei 원래 시스템의 BGE-M3·reranker·SQL·질의 분해·반복 추론과 MixRAG 전체 이식은
별도 실험이다. 현재 청킹 어댑터를 해당 시스템 재현으로 부르지 않는다.

## 원본 코드 대조 후 추가 정정

공식 코드 commit별 대조와 새 회귀 테스트는 `analysis/verify_upstream.py`,
`tests/test_upstream_review_regressions.py`에 있다. 같은 입력을 넣은 Huawei 공식
chunking 함수와 538표의 결과가 모두 같았고 67,664셀의 좌표 누락은 0건이었다.
이는 splitter 검증이며 원본 Excel 렌더러·BGE-M3·SQL·전체 QA 재현은 아니다.
RowCol 열 헤더 누락, 1행짜리 열 문서에 행 라벨이 붙는 오류, 질문이 다른 동일 ID 비교,
다른 표/음수 gold 좌표를 허용하는 검증 결함을 수정했다. 새 RowCol 문맥으로 EM 재생성이 필요하다.
