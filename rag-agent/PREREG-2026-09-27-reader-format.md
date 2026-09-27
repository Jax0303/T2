# 사전등록: 리더 입력 형식 — 셀 문장 20셀 대 행 확장 (2026-09-27)

상태: **실행 전 커밋.** 실행 중에는 코드를 고치지 않는다. 바뀐 것은 전부 §9 이탈 기록에 적는다.

## 1. 가설의 출처

test 882건 교집합 결과(본 방법 처음 규칙 대 고정 청크, 38:76, p=.00048)를 본 뒤 설정한 가설.
(두 조건이 모두 검색에 성공한 628문항에서 본 방법 280, 고정 청크 318 — `results/answer_decomp_20260927/intersection.json`.)

## 2. 가설과 판정

- **주 가설:** 리더 입력을 셀 문장 20줄 대신 "검색된 20셀이 속한 행 전체 + 열 머리글 경로"로 주면 답변 정확도가 오른다.
- **판정은 MultiHiertt dev 332건(§5) 결과로만 한다.** 행 확장 대 셀 문장, 정확 McNemar 양측. 행 확장의 맞힘이 더 많고 p < .05 이면 지지,
  그 밖은 지지하지 않음. HiTab dev 는 같은 방식으로 보고하되 판정에 쓰지 않는다.
- 본 방법(검색)은 바뀌지 않는다. 행 확장은 리더 입력 형식 조건이며 논문의 기여로 서술하지 않는다.

## 3. 조건

| 조건 | 리더 입력 |
|---|---|
| (1) 셀 문장 (`cell`) | 기존과 같다. s3c 검색 상위 20셀의 셀 문장을 순위대로 한 줄씩. |
| (2) 행 확장 (`rowexp`) | 같은 20셀이 속한 행의 비어 있지 않은 데이터 셀 전부를 표로 그린 것(`scripts/retrieval_accuracy.py: row_expand_context`). |

- 행 확장 형식: 표마다 `| row | <열 머리글 전체 경로> … |` 머리 줄, 구분 줄, 행마다 `| <행 머리글 전체 경로> | 값 … |`.
  열은 고른 행들에 값이 있는 열 전부. 표 순서·행 순서는 표 안의 위치(검색 순위 아님). HiTab 은 표마다 `# <표 제목>` 줄이 앞에 붙고,
  제목이 없는 MultiHiertt 는 제목 줄이 없다. 경로는 색인과 같은 머리글 경로다(MultiHiertt 는 최종 규칙 v3.3u 의 경로 그대로 —
  고유화가 붙인 `row N`·`column N`·`Table k` 도 들어간다).
- 셀 좌표: HiTab 은 검색 레코드 옆 `*_type_accuracy.jsonl` 의 `retrieved_cell_ids`, MultiHiertt 는 레코드의 셀 문장을 같은 코드로
  다시 만든 셀 문장과 맞춰 찾는다(문서 안에서 셀 문장이 유일하다는 v3.3u 성질, 하나라도 겹치면 코드가 멈춘다).
- 검색 결과는 두 조건이 같다: s3c, 하이브리드 α=.7, 예산 20셀, HiTab 은 질문의 표 안, MultiHiertt 는 문서 안·최종 머리글 규칙 v3.3u·라벨 없음.
- 실행 전 확인(리더 없이 문맥만 생성): 좌표 대응 실패 0. 행 확장의 전달 셀 평균은 MultiHiertt dev(911건 기준) 44.22, test 44.40, HiTab dev 61.67, test 59.64.
  (셀 문장 조건은 20.)

## 4. 리더 (기존 실험과 같다)

| | HiTab | MultiHiertt |
|---|---|---|
| 모델 | Qwen2.5-7B-Instruct | Qwen3-8B, 생각 모드 끔 |
| 양자화 | bitsandbytes 4bit NF4(이중 양자화, bf16 연산) | 같음 |
| 프롬프트 | `neutral` | `cot` |
| 최대 출력 토큰 | 64 | 384 |
| 디코딩 | temperature 0 | temperature 0 |
| 생성 방식 | 한 건씩 (`fair_filter_eval.py --no-filter`) | 128건 묶음 continuous batching (`answer_accuracy_mh.py --batch-size 128`) |
| 채점 | `hitab_exact_match_text` | `mh_exact_match`, 정답은 공식 릴리스 |

사용자 입력은 두 조건 모두 `Context:` 줄 다음에 문맥 줄, 빈 줄, `Question: …`, `Answer:` 이다(기존과 같다).

## 5. 데이터

**dev (두 조건 모두 실행)**
- MultiHiertt dev(validation): 검색 레코드 `results/dev_alpha_20260926/mh_dev/mh_dev_a1.0_records.jsonl`(sha256 47f715f9…, 라벨 섞기 α=1.0 =
  라벨 없는 s3c 셀 벡터, 머리글 v3.3u, `--keep-hybrid`). 이 레코드의 채점 문항 911건 중 579건은 표 근거와 함께 본문 문장 근거도 필요한 문항이다.
  **주 판정 집합 = 332건:** 911건에 test 모집단과 같은 규칙(본문 문장 근거가 필요한 문항 제외 = `mh_arms.load_population(keep_hybrid=False)`)을
  적용한 것. 911 = 579(본문 근거 필요) + 332. 표 근거만 필요한 validation 문항 338건 중 정답 셀 규칙으로 6건이 빠진 수다.
  id 목록 `results/reader_format_20260927/mh_dev_pop332.jsonl`(파일 sha256 bc7baad0959199a847dbf146cc39b8caa0840c8f74e0d028fed43ab5472a87d0).
  그룹: 조회 셀 1개 20, 조회 셀 2개+ 48, 산술 셀 1개 10, 산술 셀 2개+ 254. **911건은 실행하지 않는다.**
- HiTab dev 300건: dev 단일 셀 조회 1,062건(`results/dev_alpha_20260926/hitab_dev/hitab_dev_gold_prefix_type_accuracy.jsonl`)에서
  `random.Random(20260927).sample(정렬한 id, 300)` 후 정렬 → `results/reader_format_20260927/hitab_dev_pop300.json`
  (id 목록 sha256 ddce1059…). 검색 레코드 `hitab_dev_gold_prefix_records.jsonl`(sha256 dfcd75d0…, s3c, 질문의 표 안).
  HiTab dev 를 쓰는 것은 `CLAUDE.md` §3.3 의 두 번째 예외다(사용자 지시).

**test (GATE 2 뒤 사용자 지시가 있을 때만 실행, 각 1회)**
- HiTab 300(`results/ksweep_population_300.json`, 기존 s3c 답변과 같은 300건): 행 확장 1회.
  검색 레코드 `results/retrieval_accuracy/t_s3c_gold_labelabl_records.jsonl`(sha256 998048c5…, 기존 s3c 답변이 읽은 것).
- MultiHiertt 882(`results/mh_arms/cap300_20260924/cell_uniq.jsonl` 과 같은 id): 행 확장 1회. 검색 레코드
  `results/mh_arms/mh_train_cell_hv3.3u_none_doc_records.jsonl`(sha256 7394c2b1…, 기존 본 방법 답변이 읽은 것).
- MultiHiertt 882: 고정 청크(최종 머리글 규칙 검색 기록) 답변 1회. `results/rerun_20260926/mh/mh_train_chunk_records.jsonl`(sha256 ea256205…).
- 기존 결과를 그대로 쓰는 조건: HiTab 셀 문장(`results/s3c_answer_hitab300_20260926/rows.jsonl`), HiTab 표 전체
  (`results/fulltable_20260924/hitab_rows.jsonl`), HiTab 고정 청크(`results/fair_filter_20260921/rows.jsonl` arm `chunk`, 무필터),
  MultiHiertt 셀 문장(`cap300_20260924/cell_uniq.jsonl`), MultiHiertt 표 전체(`cap300_20260924/fulltable.jsonl`, 두 머리글 규칙에서 같은 입력 —
  `results/thesis_fix_20260927/context_rule_check.json`).

## 6. 비교와 보고

- **dev:** 행 확장 대 셀 문장(MultiHiertt 332건 = 주 판정, HiTab 300건 = 보고만). 보정 없음.
- **test:** 데이터셋마다 네 비교를 한 묶음으로 Holm.
  행 확장 대 셀 문장 / 행 확장 대 표 전체 / 행 확장 대 고정 청크 / 셀 문장 대 고정 청크.
  고정 청크는 MultiHiertt 에서는 최종 머리글 규칙 검색 기록으로 새로 돌린 답변, HiTab 에서는 기존 답변(HiTab 에는 머리글 규칙 구분이 없다).
- 보고 항목(조건마다): 문항 수, 맞힘, 전달 셀 평균, 리더 입력 토큰 평균. 쌍마다: b:c(앞 조건만 맞힘 : 뒤 조건만 맞힘), p(정확 이항, 양측),
  test 는 Holm p. MultiHiertt 4그룹(조회 셀 1개 / 조회 셀 2개+ / 산술 셀 1개 / 산술 셀 2개+)별 값은 **탐색적 결과**로 표기.
- 집계: `results/reader_format_20260927/analyze.py dev|test` (실행이 끝난 뒤에만 돌린다).

## 7. 로그 규칙

실행 중 로그에 정확도·맞힘 수를 쓰지 않는다(진행 문항 수만). `answer_accuracy_mh.py --quiet-accuracy`, `fair_filter_eval.py` 는 원래 진행 수만 쓴다.
정확도는 결과 파일(`*.jsonl`, `*.json`)에만 남고 GATE 보고 때 `analyze.py` 로 처음 센다.

## 8. 코드 변경 (이 커밋)

- `scripts/retrieval_accuracy.py`: `row_expand_context(cells, tabs, page_titles)` 추가(기존 `subtable_context` 를 행 전체 셀로 호출). 기존 경로 변화 없음.
- `scripts/answer_accuracy_mh.py`: `--condition rowexp`(셀 문장 → 셀 좌표 → 행 확장, 레코드가 s3c 셀·같은 머리글·라벨 규칙이 아니면 멈춤),
  `--quiet-accuracy`. 기본 동작 변화 없음.
- `scripts/fair_filter_eval.py`: `--pop-file`, `--records`(arm 하나), `--context cell|rowexp`, 행에 `context_format`·`cells_delivered` 필드. 기본 동작 변화 없음.
- `results/reader_format_20260927/`: `hitab_dev_pop300.json`, `mh_dev_pop332.jsonl`(실행 전 수정 커밋), `analyze.py`, `run_dev.sh`, `run_test.sh`.
- dev 검색 입력(그동안 미커밋): `results/dev_alpha_20260926/hitab_dev/hitab_dev_gold_prefix.*`, `results/dev_alpha_20260926/mh_dev/mh_dev_a1.0.*`.

## 명령

    mkdir -p results/reader_format_20260927/dev && setsid nohup bash results/reader_format_20260927/run_dev.sh  > results/reader_format_20260927/dev/run.log 2>&1 &   # GATE 2
    python3 results/reader_format_20260927/analyze.py dev
    mkdir -p results/reader_format_20260927/test && setsid nohup bash results/reader_format_20260927/run_test.sh > results/reader_format_20260927/test/run.log 2>&1 &  # GATE 3 (지시 후)
    python3 results/reader_format_20260927/analyze.py test

## 9. 이탈 기록

- 실행 전 수정(2026-09-27): 판정 집합 911 → 332, 사유: test 모집단 정의와 일치. `run_dev.sh` 의 MultiHiertt 두 실행에
  `--same-queries-as results/reader_format_20260927/mh_dev_pop332.jsonl` 을 더했다. 첫 사전등록 커밋 c869675 뒤, dev 실행 전.

## 결과

(실행 후 기록)
