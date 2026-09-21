# T2 논문·공식 코드 구현 검증 — 2026-09-11

> **2026-09-21 지도교수 지시로 MT2Net을 비교대상에서 제외했다.** MT2Net/MultiHiertt 공식
> 코드 대조 행·서술을 지웠다(검증은 유효했던 기록이며 `archive/mt2net-2026-09-21/`에
> 원문이 있다). Huawei·Google TableRAG·MixRAG 대조는 MT2Net과 무관하므로 그대로 둔다.

현재 비교군은 **동일 검색기·reader 아래 표현 방식을 비교하는 HiTab 적응 실험**으로는 사용할 수 있다. 그러나 Huawei TableRAG, Google TableRAG의 **전체 시스템을 재현한 결과로 제시하면 안 된다**. 공식 코드 대조에서 RowCol 열 헤더 누락을 추가 발견해 수정했다. 기존 EM은 수정된 문맥으로 다시 생성해야 하며, 이번 작업에서 GPU 답변 EM을 새로 측정하지 않았다.

## 원본과 구현의 차이

| 비교군 | 원본에서 확인한 핵심 | T2 판정과 조치 |
|---|---|---|
| Huawei TableRAG (2025) | 질의 분해, 텍스트 검색·재정렬, SQL 실행, 중간 답변의 반복 | 현재 청킹+BGE/BM25+Qwen 실험에는 핵심 추론 단계가 없다. ‘Huawei에서 착안한 문자 청킹’으로 제한해 해석해야 한다. |
| Google TableRAG (2024) | schema/cell 별도 검색, 질의 확장, 전체 표에 접근하는 PyReAct | T2의 혼합 hint 인덱스와 단일 답변 생성은 전체 재현이 아니다. 기존 Google arm은 본 비교에서 제외 유지. |
| Google 논문의 RowCol | 선택한 행·열의 교집합 DataFrame을 열 이름과 함께 Markdown으로 제공 | 교집합 수정 뒤에도 열 이름이 빠져 있었다. 선택된 열의 leaf 헤더를 보존하도록 수정. HiTab 적응이라는 표기 유지. |
| MixRAG | H-RCL, 검색 결합, LLM 재정렬, RECAP | 문헌상의 후보이며 T2에 전체 구현되어 있지 않다. 구현된 비교군으로 집계하지 않는다. |

근거 논문: Yu·Jian·Chen (2025), [TableRAG: A Retrieval Augmented Generation Framework for Heterogeneous Document Reasoning, arXiv:2506.10380v2](https://arxiv.org/html/2506.10380v2); Chen 외 (2024), [TableRAG: Million-Token Table Understanding with Language Models, arXiv:2410.04739](https://arxiv.org/html/2410.04739); [MultiHiertt: Numerical Reasoning over Multi Hierarchical Tabular and Textual Data, arXiv:2206.01347v1](https://arxiv.org/html/2206.01347v1); [Mixture-of-RAG: Integrating Text and Tables with Large Language Models, arXiv:2504.09554v3](https://arxiv.org/html/2504.09554v3).

Google 공식 구현은 schema 검색 문서에 **열 이름**을 넣고 schema 설명은 검색 결과 metadata로 전달한다. T2의 schema 설명 자체를 검색하는 방식과 다르다. 숫자 dtype 추론과 cell corpus 구성도 일치하지 않는다. 원본 RowCol의 top-K·인코딩 예산 및 PyReAct 조건과 T2의 corpus 검색·셀 예산·Qwen 조건 역시 다르다. [고정된 공식 retriever 소스](https://github.com/google-research/google-research/blob/08a8d6736475776f42ffac23b2c13111a28e5795/table_rag/agent/retriever.py), [공식 reader 소스](https://github.com/google-research/google-research/blob/08a8d6736475776f42ffac23b2c13111a28e5795/table_rag/agent/agent.py).

## 추가 발견하고 수정한 결함

1. **RowCol 열 헤더 누락.** 이전 문맥은 `Title | r0|10`처럼 값과 행 라벨만 남았다. 기본 문맥을 `row | Revenue` 헤더와 `r0 | 10` 값을 가진 Markdown으로 변경했다. 교집합 밖의 값은 포함하지 않는다. `--rowcol-context legacy_values`는 과거 조건 재현용이며, 전체 계층 헤더는 `--rowcol-header path`로 명시한다.
2. **1행짜리 열 문서의 행 라벨 혼입.** 공용 직렬화 함수가 열도 행으로 판단했다. 열 인덱스 생성 시 행 라벨 삽입을 끄도록 수정했다.
3. **같은 ID지만 다른 질문인 결과를 결합할 수 있음.** 답변 레코드에 질문을 저장하고 v2 비교 및 oracle 합성에서 질문 일치를 검사한다.
4. **잘못된 gold 좌표 허용.** 다른 표 ID, 음수, bool, 잘못된 좌표 길이를 거부한다. gold 문맥 생성에서도 경계를 검사한다.
5. **답변 단계 제외 판정의 test split 하드코딩.** 검색 metadata의 실제 split을 사용하도록 수정했다.

이전 작업에서 추가한 문맥–좌표–검색 판정 일치 검사, 전체 gold 저장, 입력 길이 감사, 결과 출처·해시 및 비교 분모 검증도 유지했다. 미래 비교 manifest의 RowCol 이름도 새 Markdown 조건에 맞췄다.

## Huawei 공개 코드 자체에 대한 별도 검증

논문은 1,000토큰/200토큰 중첩을 기술하지만 공개 구현은 `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`로 문자 수를 센다. 두 조건을 같은 설정으로 취급하지 않는다. 공식 함수와 T2 함수에 **동일한 Markdown 입력**을 넣었을 때 538표 모두 청크가 일치했다. 67,664개의 비어 있지 않은 원본 셀에서 좌표 누락은 0건이었다. 이는 splitter와 prefix의 대조이며 원본 Excel 렌더러까지 동일하다는 뜻은 아니다. [공식 retriever](https://github.com/yxh-y/TableRAG/blob/f8d798fc60fb44736d7285ca3ca1dbcb7b3e984d/online_inference/tools/retriever.py).

원본 함수의 관련 구문을 추출하여 격리 실행한 결과, Excel 숫자 셀을 문자열 변환 없이 join하는 경로에서 TypeError가 발생했다. 빈 셀은 건너뛰어 렌더링한다. SQL 응답 처리에서는 함수 객체를 subscript한 뒤 예외 처리의 문자열을 3개 변수로 풀면서 ValueError가 발생했다. 이는 확인한 공개 commit의 해당 입력·경로에 관한 결과이며, 논문의 실험 실행본 전체가 실패했다는 주장은 아니다. [Excel 렌더러](https://github.com/yxh-y/TableRAG/blob/f8d798fc60fb44736d7285ca3ca1dbcb7b3e984d/online_inference/utils/tool_utils.py), [SQL 응답 처리](https://github.com/yxh-y/TableRAG/blob/f8d798fc60fb44736d7285ca3ca1dbcb7b3e984d/online_inference/main.py).

Reranker의 기본 초기화는 모형 로딩을 대체한 검사에서 `cuda:4`, `num_gpus=1`로 확인했다. GPU 추론이나 재정렬 점수를 검증한 것은 아니다. 초기 검사 도구의 불완전한 mock이 발생시킨 AttributeError는 원본 결함으로 인정하지 않았고 최종 검사에서 제거했다.

## 실행 증거와 해석

- 회귀 테스트: 131 passed, 2 skipped. 생략은 FAISS 부재와 datasets/MultiHiertt 환경 부재에 해당한다.
- 수정된 RowCol BM25-only 전수 실행: 538표, 12,950 검색 단위, 1,584질의 중 1,581채점·3제외. 모든 레코드의 문맥·좌표·점수 일치 검사 실행.
- 셀 예산 20은 단위 전체를 넣은 뒤 중단하는 정책이다. 실제 평균 전달 셀은 22.9899, 예산 초과는 1,059건이다. 엄격한 20셀 실험이라고 표기하면 안 된다.
- BM25-only 전체 채점 검색 정확도는 0.3023이다. 디버깅 실행이며 BGE 또는 최종 EM 성능으로 해석하지 않는다.
- 모델 가중치를 이용한 전체 원본 파이프라인과 GPU 답변 EM은 실행하지 않았다. 수정으로 점수가 상승했다고 주장할 근거는 없다.

이전 저장 결과의 주지표 991질의에서 검색 성공 906건과 답변 정답 710건의 차이는 **19.78%p**였다. 같은 질의별로 보면 검색 성공·답변 오답 200건, 검색 실패·답변 정답 4건이어서 `(200−4)/991`과 일치한다. 차이 자체가 곧 채점 버그라는 뜻은 아니지만, 문맥 전달 결함과 모델 추론 실패를 분리해야 한다. 이번 RowCol 수정 전 EM은 새 구현의 EM으로 재사용할 수 없다.

‘최신 TableRAG가 왜 낮은가’에 대해 현재 결과가 뒷받침하는 설명은 **원본의 핵심 구성요소가 빠진 적응 비교군을 측정했다**는 것이다. 원본 논문의 시스템보다 우수하다는 결론은 현재 실험으로 지지되지 않는다. 동일 검색기·reader의 표현 비교표와 원본 전체 시스템 비교표를 구분하고, 후자는 별도 재현 실험이 필요하다.

## 고정한 출처와 수정본

| 저장소 | 확인한 commit |
|---|---|
| yxh-y/TableRAG | `f8d798fc60fb44736d7285ca3ca1dbcb7b3e984d` |
| google-research/google-research | `08a8d6736475776f42ffac23b2c13111a28e5795` |
| ChiZhang-bit/Mixture-of-RAG | `f498eb62a68b944bbfc6f107a51466a35add0e97` |

수정은 `work/t2-audit`의 `fix/evaluation-integrity`에서 수행했다. 기준 T2 commit은 `6d88d4db00018fa08bc521676045b38198ae960e`이다. 원래 `C:\Users\ugh\T2`의 작업 변경은 보존했고 commit·push는 하지 않았다.

검사 환경은 Python 3.12.14, pandas 2.2.3, langchain-text-splitters 1.1.2이다. `analysis/verify_upstream.py`는 다운로드한 공식 소스 SHA256을 확인하고 원본 메서드를 실행한다. 재실행은 수정본 `rag-agent`에서 다음과 같다.

```powershell
python -m pytest tests -q -ra
python analysis/verify_upstream.py --sources-dir ../../upstream-verification --huawei-dir ../../huawei-tablerag --out new_upstream_verification.json
```

최종 증거는 같은 outputs 디렉터리의 `T2_원본대조_최종검증.json`, `T2_최종테스트.xml`, `T2_최종검증요약.json`이다. `T2_upstream_verified.patch`는 기준 commit 대비 이전 무결성 수정까지 포함한 누적 패치이며, 이전 패치 위에 중복 적용하지 않는다. 과거 중간 검사 JSON보다 ‘최종검증’ 파일을 우선한다.
