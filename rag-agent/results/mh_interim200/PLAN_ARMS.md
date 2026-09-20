# interim200 비교군 네 개 — 답변 EM 탐색 비교 (2026-09-14) — 탐색용, 사전등록 아님

- 전체 1,047문항 실행은 보류한다. `PREREG-2026-09-13-reader-qwen3.md` 는 바꾸지 않고, 이 결과는 그 판정에 쓰지 않는다.
- 질의: interim200 고정 id(`ids_200.json`, ids_sha256 `52696e6b…`, 커밋 `784e244`). 새로 뽑지 않는다.
- 기존 결과 파일은 덮어쓰지 않는다.

## 비교군 (REPORT-2026-09-13 §3 v2 조건 — 헤더 v2, 라벨 없음)

| 이름 | 검색 레코드 (`results/mh_arms/`) | 문맥 기준 = 기존 답변 레그 | 색인 단위 (`run_v2base.sh`) |
|---|---|---|---|
| chunk | `mh_chunk_hv2_records.jsonl` | `mh_chunk_hv2_answer_doc.jsonl` | 고정 청크 1,000자 |
| Huawei 기반 | `mh_huawei_hv2_records.jsonl` | `mh_huawei_hv2_answer_doc.jsonl` | trag_hetero, 1,000자 · 겹침 200 |
| RowCol | `mh_rowcol_hv2_records.jsonl` | `mh_rowcol_hv2_answer_doc.jsonl` | 행·열 단위, row-text values |
| TableRAG 기반 (유리한 설정) | `mh_tablerag_allobj_path_hv2_records.jsonl` | `mh_tablerag_allobj_path_hv2_answer_doc.jsonl` | dtype all_object, colmode path |

함께 비교: 본 방법·MT2Net 은 interim200 에서 이미 생성한 `mh_{cell_hv2,mt2net}_answer_doc_qwen3_8b_cot_200.jsonl`
(2026-09-14 00:33 완료)을 재사용한다.

## 생성

- 문맥: 기존 검색 레코드의 doc 문맥 그대로. `make_refs_arms.py` 가 기존 답변 레그의 context_sha256 을
  `ref_contexts_<arm>_200.jsonl` 로 뽑고, `--same-contexts-as` 가 리더를 싣기 전에 질의마다 일치를 확인한다.
- 설정: 본 방법·MT2Net 200건과 같다 — `local:Qwen/Qwen3-8B?quantization=4bit`(revision `b968826…`),
  `--prompt cot --max-tokens 384`, greedy, seed 42, batch 1, `--header-rule v2`.
- 문항별 즉시 저장·재개(`66b39bf`): 모든 레그를 `--resume` 으로 돌린다. 중단되면 `run_arms200.sh` 를 다시
  돌리면 남은 문항만 생성하고, 완료된 레그는 건너뛴다. 로그는 이어 붙인다.
- 코드 차이: 본 방법·MT2Net 200건은 저장·재개 추가 전 코드(`784e244`), 비교군은 추가 후(`66b39bf` 이후)다.
  리더 호출(프롬프트, `complete` 인자)은 같고 달라진 것은 결과를 쓰는 방식이다. 요약에 커밋과 코드 해시를 싣는다.
- gold 레그 없음.

## 분석 (`analyze_arms.py`, 생성 전에 커밋)

- 검사: 여섯 파일 모두 200 id 를 담고 질의마다 문맥 해시가 기준과 같으며, 요약의 records_sha256 이 파일과
  같고, 생성 설정·revision 이 본 방법과 같다. 하나라도 어긋나면 멈춘다.
- 칸: 전체(200), 다중 조회(70), 다중 산술(76). 단일 조회·단일 산술은 전체에만 들어간다.
- 기술 통계: arm × 칸 EM, 검색 정확도, 'Final answer:' 표시 누락 수. arm 별 EM 에는 신뢰구간을 붙이지 않는다.
- 추론 통계 (탐색용): 본 방법 대 다섯 arm × 세 칸 = **15개 비교를 한 묶음**으로 본다(REPORT §3 과 같은 구성).
  - 정확 McNemar p 에 **Holm 보정**.
  - EM 차이(본 방법 − 상대)의 짝지음 부트스트랩 CI 는 **Bonferroni 동시 수준 1 − 0.05/15**, 20,000회,
    층 안 복원추출, seed 20260913.
- 해석: 탐색용이다. query count=200 은 작은 차이를 가를 검정력이 낮고, 전체 칸은 다중 조회·다중 산술 칸을 포함해 독립이 아니다
  (Holm 은 의존 구조와 무관하게 성립한다).

## 경위 — 첫 실행 실패와 재실행 (2026-09-14)

- 00:45 첫 실행(`b90287c`)은 네 레그 모두 문맥 해시 확인 뒤 리더를 싣고 **생성 전에** 멈췄다.
  `answer_accuracy_mh.run_config` 가 상대 경로 `__file__`('_ops.py', torch 가 등록)을 작업 폴더 기준으로 풀어
  해시하려다 FileNotFoundError. 가짜 리더 테스트는 torch 를 대체해 이 경우를 잡지 못했다.
- 생성된 행·`.run.json`·요약은 없다(실행 조건을 쓰기 전 단계). 레그 로그에 traceback 이 남아 있고 재실행 로그는 이어 붙인다.
- 수정: 절대 경로이고 실제 파일인 모듈만 해시한다. 테스트에 상대 경로 `__file__` 모듈을 넣어 회귀를 막고,
  실제 Qwen3-8B 로 chunk 2문항을 scratchpad 에 돌려 `.run.json`(코드 파일 33개)·문항별 행·요약이 써지는 것을 확인했다.
- 분석 계획·비교군·표본·생성 설정은 바꾸지 않았다. 수정 커밋 뒤 `run_arms200.sh` 를 그대로 다시 돌린다.
