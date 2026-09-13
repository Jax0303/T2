# 사전등록 — MultiHiertt 산술 리더 설정 고정 → 본 방법 대 MT2Net 답변 EM (2026-09-13)

## 유지하는 것

- `REPORT-2026-09-13.md` 의 수치와 기존 사전등록 3개(`PREREG-2026-09-13-*.md`)는 고치지 않는다.
  기존 결과 파일은 덮어쓰지 않고, 새 실행은 새 경로에 쓴다.
- 라벨 규칙을 L2 로 바꿔 다시 돌리지 않는다(`PREREG-2026-09-13-table-label.md`).
- 공정 비교 기준 조건은 헤더 v2, 라벨 없음이다.

## 1. 파일럿 — validation, 탐색용 (보고용 수치가 아니다)

- 질의: validation 산술(gold 해석됨, 제외 아님)에서 seed 20260913 으로 60건. REPORT §5 와 같은 추출 규칙.
- 문맥: gold 셀만, 헤더 v2, **라벨 none**. 본 실행의 gold 레그와 같은 렌더링이다. §5 파일럿은 L1
  렌더링이었으므로 Qwen2.5-7B 도 none 으로 다시 잰다.
- 후보 4개 = 리더 {`Qwen/Qwen2.5-7B-Instruct` 4bit, `Qwen/Qwen3-8B` 4bit 비생각 모드} ×
  프롬프트 {`direct`(현행 neutral, 64토큰), `cot`(풀이 허용, 마지막 줄 'Final answer:', 384토큰)}.
  greedy, batch 1. 명령은 `results/mh_arms/run_reader_pilot.sh`.

### 선택 규칙 (파일럿 결과를 보기 전에 고정)

1. **시간 가드**: 본 실행 예상 시간 = 3 레그 × 1,047 × 파일럿 질의당 초(`sec_per_query`).
   30시간을 넘는 후보는 제외한다.
2. 남은 후보 중 gold 문맥 산술 EM 이 가장 높은 후보를 채택한다.
3. 1위와의 차이가 2건(3.3pt) 이하인 후보가 있으면 그중 `sec_per_query` 가 가장 작은 후보를 채택한다.

선택은 본 실행의 설정을 고정하려는 것이지 리더 우열의 주장이 아니다(n=60 은 후보 간 검정력이 낮다).

## 2. 본 실행 — train, 같은 1,047 질의·같은 문맥

선택된 리더·프롬프트·출력 토큰 하나로 **1,047건 전부**(조회 포함)를 돌린다. 질의 유형을 보고
프롬프트를 바꾸지 않는다 — 배포된 리더는 유형 라벨을 모른다.

| 레그 | 검색 레코드 | 문맥 기준 (`--same-contexts-as`) |
|---|---|---|
| 본 방법 | `mh_cell_hv2_records.jsonl` | `mh_cell_hv2_answer_doc.jsonl` |
| MT2Net 원문 | `mh_mt2net_records.jsonl` | `mh_mt2net_answer_doc.jsonl` |
| gold 셀만 | `mh_cell_hv2_records.jsonl` (`--condition gold --label-rule none`) | `mh_GOLD_hv2_doc.jsonl` |

- `--same-contexts-as` 는 질의 id 집합과 질의마다 `context_sha256` 이 기준 실행과 같은지 리더를
  싣기 전에 확인하고, 하나라도 다르면 멈춘다. 실행 전 점검에서 세 레그 모두 통과했다(1,047건).
- 다른 비교군(chunk, Huawei, RowCol, TableRAG) 재실행은 이 결과를 본 뒤 사용자가 정한다.

### 분석

- **주**: 본 방법 대 MT2Net 답변 EM, 짝지은 정확 이항 검정 — 전체 / 다중 조회 / 다중 산술, Holm 보정
  (REPORT §3 과 같은 세 칸).
- **부**: 층별 EM, gold 문맥 EM, 같은 arm 의 새 설정 대 기존 답만 설정(짝지음, 층별).
- **기술**: `analysis/mh_em_split.py` 로 검색 성공 여부별 EM 불일치. '둘 다 검색 성공' 칸의 차이는
  "검색 성공 여부로 설명되지 않는다"까지만 쓴다 — 그 칸에서도 함께 들어간 셀과 문장 표기는 색인
  표현이 정하므로 검색 표현과 무관하다고 해석하지 않는다.

## 예측

1. 파일럿: 두 리더 모두 `cot` EM > `direct` EM (§5 방향 재현).
2. 파일럿: Qwen3-8B `cot` EM ≥ Qwen2.5-7B `cot` EM.
3. 본 실행(`cot` 채택 시): 산술 471건 EM 이 두 arm 모두 기존 답만 설정보다 오른다(짝지음 p<.05).
4. 본 실행: 본 방법 대 MT2Net 전체 EM 차이는 유의하지 않다(Holm p≥.05). 검색이 비겼기 때문이다
   (doc 105:116, REPORT §2).
5. 본 실행: EM 불일치 중 '둘 다 검색 성공' 칸의 비율이 85% 이상이다(기존 125/133).
6. `cot` 채택 시 조회 층 EM 은 떨어질 수 있다(형식, 다중 값). 떨어지면 그대로 싣는다.

## 알려진 한계

- Qwen3-8B 의 `generation_config.json` 기본값은 샘플링(temperature 0.6, top_p 0.95, top_k 20)이다.
  결정성과 기존 레그와의 동일 조건을 위해 greedy 를 쓴다.
- batch 1 고정. 배치 생성은 greedy 출력을 바꿀 수 있어 기존 레그·파일럿과 조건이 달라지므로 쓰지 않는다.
- `cot` 는 'Final answer:' 표시가 없으면 출력 전체로 채점한다(관대한 추출 없음). 누락 건수를 요약에 싣는다.
