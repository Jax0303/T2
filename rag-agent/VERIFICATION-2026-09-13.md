# 비교군 구현 실사 — 원본 코드와 대조 (2026-09-13)

> **2026-09-21 지도교수 지시로 MT2Net을 비교대상에서 제외했다.** 이 보고서는 원래 MT2Net
> 재현도 함께 대조했다 — 그 대조 행과 §3(a)의 MT2Net 수치를 지웠다(원 수치는
> `archive/mt2net-2026-09-21/`). TableRAG·Huawei 대조는 MT2Net과 무관하므로 그대로 둔다.

`.md` 를 읽고 확인한 것이 아니라 **원본 저장소를 받아 함수를 실행해서** 대조했다.
받은 원본과 커밋:

| 저장소 | 쓰인 곳 | 확인 방법 |
|---|---|---|
| `microsoft/HiTab` (리포에 이미 있음, `data/hitab`) | HiTab 공식 채점기 | `qa/table/utils.py` `hmt_score`/`hmt_equal`/`hmt_process_answer`, `qa/datadump/utils.py` `naive_str_to_float` 를 줄 단위 대조 |
| `google-research/google-research` `table_rag/` | TableRAG (Chen et al., NeurIPS 2024) | `agent/retriever.py` 의 `build_*_corpus` 를 **그대로 실행**해 우리 포트 출력과 문자열 비교 (`analysis/upstream_parity.py`) |
| `yxh-y/TableRAG` | Huawei TableRAG (arXiv 2506.10380, EMNLP 2025) | `online_inference/tools/retriever.py`, `utils/tool_utils.py: excel_to_markdown` 대조 |

---

## 1. 채점기 — 결함 없음

- HiTab 공식 채점기 포팅(`rag_agent/eval/metrics.py`)은 원본과 같다. 추가한 것은
  두 가지이고 둘 다 **관대함이 아니라 모양 맞추기**다: ① `$` 제거(HiTab 은 0건 이동,
  AIT-QA 44건), ② 자유 텍스트 한 줄을 다값 gold 모양으로 쪼개기.
- **자체 검증 통과**: gold 를 그대로 예측으로 넣으면 EM = 1.0000
  (조회 query count=1,132, 산술 query count=449). 채점기가 정답을 떨어뜨리고 있지 않다.
  재현: `PYTHONPATH=. .venv/bin/python -m rag_agent.eval.answer_em results/evaluation_v2/s3c_v2_records.jsonl`
- MultiHiertt 공식 채점기도 같은 방식으로 옮겼다(`rag_agent/eval/multihiertt_em.py`).
  ⚠️ 원본 `str_to_num` 은 `-` 를 지워 부호를 없앤다. 원본의 성질이므로 그대로 두되,
  양쪽에 똑같이 걸린다는 것과 HiTab EM 과 섞지 않는다는 것을 적어 둔다.

## 2. 배달-채점 일치 — 결함 없음

`rag_agent/eval/artifacts.py: validate_retrieval` 가 **모든 레코드에서** 다음을 강제한다:
채점에 쓴 셀 집합 == 선택된 색인 단위가 담은 셀 == 리더에게 실제로 간 문자열.
답변 레그는 시작할 때 전 레코드에 이 검사를 다시 돌린다(`load_evidence`).
→ "검색은 맞혔다고 세는데 리더는 그 셀을 못 봤다" 부류의 결함은 구조적으로 불가능하다.

## 3. 원본과 다른 점 — 찾았고, 셋 중 둘은 이번에 고쳤다

### (a) 비교군을 **자기 논문의 검색 범위 밖**에서 재고 있었다 → 고침

TableRAG 의 `init_retriever(table_id, df)` 는 **표 하나마다** 색인을 세우고 그 안에서만
찾는다. 그런데 표 1 은 전부 코퍼스 전역(538표)에서 쟀다. 같은 계측기로 각 비교군의
**자기 범위**(`--corpus gold`)를 다시 쟀다:

| arm | 코퍼스 전역(538표) | 자기 범위(표 하나) |
|---|---|---|
| 본 방법 `s3c` | .8805 | **.9475** |
| generic chunk (1,000자) | .7704 | .8893 |
| Huawei TableRAG 청크 | .6945 | .7951 |
| RowCol (TableRAG §4.2) | .4870 | **.7457** |
| TableRAG 셀/스키마 (`infer`, leaf) | .0721 | .3669 |
| TableRAG 셀/스키마 (`all_object`, path) | .2385 | **.6787** |

마지막 줄이 (b) 의 수정까지 반영한 **가장 유리한 읽기**다. 예전 표의 `.0721` 은
원본이 이 데이터에서 실제로 타는 분기가 아니었다 — 자기 범위·유리한 읽기로는
**.6787**, 9 배 차이다. 리뷰어가 가장 먼저 짚을 자리였다.

**결론이 바뀐다.** 코퍼스 전역과 표 하나 범위에서 arm 간 격차의 크기 자체가 다르다 —
RowCol 은 .4870→.7457(+.2587), TableRAG(all_object/path)는 .2385→.6787(+.4402)로
범위를 좁히면 격차가 크게 준다. 즉 **범위를 하나로 고정해 낸 격차만으로 "문장 표현이
더 좋다"고 일반화할 수 없다** — 코퍼스 규모에서의 표 식별이 섞여 있을 수 있고, 그것이
`CLAUDE.md` §2·§5 가 이미 말한 것과 같다. 논문 표는 두 범위를 **둘 다** 실어야 한다 —
전역만 실으면 비교군을 자기 설계 밖에서 재고 이겼다고 쓰는 것이 된다.

### (b) TableRAG 의 숫자 열 접기 → 변형을 추가함

우리 포트는 "비어 있지 않은 셀이 전부 숫자로 읽히는 열"을 숫자 열로 보고 min/max
요약 문서 하나로 접는다. 원본은 pandas dtype 으로 가른다(`utils/utils.py: infer_dtype`).
**계층 표를 잎 라벨로 펴면 열 이름이 겹치고**(HiTab 120표에서 932열 중 487열),
그때 원본의 `pd.to_numeric(df[중복이름])` 은 Series 가 아니라 DataFrame 을 받아
예외를 내고 `errors='ignore'` 가 삼킨다 → 그 열은 object 로 남아 **셀마다 색인된다.**
즉 이 데이터에서 원본이 실제로 타는 분기는 우리 포트보다 **비교군에 유리**하다.
`--tablerag-dtype all_object` 로 그 읽기를 추가하고 둘 다 보고한다.
실측: 538표에서 .0721 → .1056(leaf) / .2385(path), 표 하나 범위에서
.3669 → .6344(leaf) / **.6787**(path).

열 분류가 일치하는 비율(중복 이름 없는 프레임 기준, 120표): 96.5% (512열 중 494열).
겹치는 이름까지 포함하면 원본은 더 많은 셀을 개별 색인한다.

### (c) Huawei 렌더러의 사소한 차이 → 기록만 남김

원본 `excel_to_markdown` 은 ① 첫 행 뒤에 `| --- |` 구분행을 넣고 ② `None` 셀을
건너뛰며 ③ `" | "` 로 시작·끝낸다. 우리 렌더러는 구분행을 넣지 않고 빈 칸을
유지한다(정렬 보존 — 비교군에 **유리한** 쪽). 청크 경계가 몇 글자 밀리는 수준이고
방향은 비교군에 불리하지 않다. 재실행하지 않고 차이를 적어 둔다.

## 4. 아직 남은 판단 — 숨기지 않는다

- 어느 arm 도 그 논문의 **시스템**이 아니다. 검색기·예산·리더·채점기를 고정하고
  **색인 단위만** 갈아 끼운 비교다. TableRAG 의 ReAct 코드 실행, Huawei 의 NL2SQL·
  리랭커는 모두 빠져 있다.
- 청킹 arm 은 인코더 한계(512토큰)로 문서의 11.8%가 잘린다(`--embed-overflow truncate`).
  1,000자 청크 + 512토큰 인코더 조합의 성질이고, 요약에 감사 수치로 기록된다.
