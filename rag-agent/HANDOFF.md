# 인수인계 — 2026-08-31

브랜치: `fix/encoder-provenance` (로컬) → 원격은 `phase4-audit-local`.
원격 `fix/encoder-provenance`에는 다른 머신의 커밋 22개가 있고 **병합하지 않았다**
(충돌 8건, `corpus_dump_vs_cell.py` / `freeze_populations.py` 포함).

## 지금 서 있는 곳

Phase 4(리더 EM) 완료, 셀 문장 정합성 감사 완료, Phase 5(산술 리더 교체)는
**Task B까지 완료 / Task C 미실행**.

### 진행 중이던 것 — Phase 5 Task C

사용자 지시 원문:
- 리더 `Qwen/Qwen2.5-Coder-7B-Instruct` rev `c03e6d358207e414f1eca0bb1891e29f1db0e242`,
  4-bit NF4, temp=0, seed=42, **max_new_tokens=128**
- 대상 `hitab_arith` 60, `rhb_num` 54 (조회 pool 건드리지 말 것)
- 조건 `P1_fixed_512`, `P4_path_cell(완전 재구성)`, `gold_cell` × `direct`/`extract`
  = 6조건 × 114 = 684회
- extract: `{"values": [...], "op": "..."}`, 지원 연산자 6종
  (`sum, diff, div, average, range, opposite`), 목록 밖은 파싱 실패로 기록,
  임의 매핑 금지, 파싱 실패율 조건별 보고
- 대조군은 Phase 4의 `Qwen2.5-7B-Instruct` 값 **인용** (재실행 금지)
- 출력 `results/phase5/`, phase4 덮어쓰기 금지

스크립트 `analysis/phase5_taskC.py`는 **작성 직전에 중단됐다** (사용자가 정합성
확인을 먼저 요청). 다시 쓸 때 필요한 컨텍스트 출처:

| 조건 / pool | 컨텍스트 출처 |
|---|---|
| P1 / hitab_arith | `results/phase4/retrieval_294/P1_fixed_512_hitab_arith.jsonl`의 `topk[].used` |
| P1 / rhb_num | 같은 디렉터리 `P1_fixed_512_rhb_num.jsonl` |
| P4 / rhb_num | 같은 디렉터리 `P4_path_cell_rhb_num.jsonl` (RHB의 P4는 원래 재구성 기반) |
| P4 / hitab_arith | 재구축 필요 — `rec_rp + rec_cp` 색인. `analysis/taskd_recon.py`의 빌드 블록 참조 (ranks만 저장했고 텍스트는 저장 안 함) |
| gold_cell / 양쪽 | `build_chunks(C, "P4_path_cell", bud, by, "S3c")`의 gold 셀 문장 |

greedy fill은 Qwen2.5 토크나이저 기준 B_reader=4096이고 Coder가 같은 토크나이저를
쓰므로 컨텍스트를 그대로 승계할 수 있다. Task B 실측: 파싱 실패 3/10(전부
허용 목록 밖 연산자 `percent`/`percentage`/`same`), 생성 토큰 max 60 → 128이면 충분.

## 반드시 지켜야 할 규약 (사용자가 매 단계 반복한 것)

1. 숫자를 추정하거나 생성하지 마라. 측정되지 않은 값은 `MEASURED: NO`로 표기한다.
2. 일부 구간만 골라 보고하지 마라. 모든 조건의 모든 수치를 표에 넣는다.
3. 결론·해석·논문용 문장을 쓰지 마라. 수치와 관측 사실만 출력한다.
4. GATE에서 반드시 멈추고 승인을 기다린다.
5. 기존 실험 결과 파일을 덮어쓰지 마라. 새 디렉터리에 저장한다.
6. 실패하거나 건너뛴 항목을 숨기지 말고 전부 기록한다.
7. 설명은 짧게. 결과만.

## 채점 규칙 (고정)

`analysis/phase4_summary.py: em()`. 통화/퍼센트 기호 제거 → 천단위 콤마 제거 →
공백 축약·소문자화 → 순수 소수의 후행 0 제거 → **R1**(수치 정답에 한해
상대오차 <0.01). R2/R3/R4 미채택, 부호 정규화 없음, 다중 gold는 전부 일치 요구.
`gold_parts`는 `[`/`(`로 시작하는 리터럴만 컨테이너로 파싱한다 (`BUGFIX_LOG.md`).

## 알려진 함정

- **Recall 분모가 두 가지다.** 보고된 값은 전부 **셀 단위**(분모 = gold 셀 수).
  산술 pool은 쿼리 단위와 다르다 (`results/audit/recall_denominators.json`에 둘 다).
  `results/audit/rec_rp/summary.json`의 recall만 쿼리 단위로 계산됐다.
- **`np.argsort`는 비안정 정렬**이다. Phase 3의 저장된 rank와 대조하려면 그 run의
  kmax로 다시 랭킹해야 한다 (`analysis/phase4_retrieve.py`의 `rank3`).
- 인라인 스크립트는 `.venv/bin/python`으로 실행한다.
- `git` 명령을 리포 루트에서 돌리면 작업 디렉터리가 바뀐다. `rag-agent/`로 돌아올 것.
- HiTab 정렬 실패 116표(21.48%)의 쿼리 309건은 **동결 모집단에 애초에 없다**.
  분모에서 빠진 것이지 오답으로 세어진 게 아니다.

## 핵심 수치 (자세한 건 아래 파일들)

- `results/phase4/FINAL.md` — pool × 조건 EM, McNemar + Holm, gold_cell 상한, 채점 규칙 전문
- `results/FINAL_TABLE.md` — dataset × 정책, Recall/EM, (O)/(R) 오라클 표기
- `results/audit/` — 셀 문장 감사, 행 경로 실패 분류, 정렬 실패 영향, 재구성 색인 결과
- `PREREGISTER.md` 개정 1~6, `BUGFIX_LOG.md`

## 아직 안 한 것

- Phase 5 Task C (위)
- `results/audit/manual_check_{rhb,mh}.xlsx` 채점 — 사람이 할 일.
  RealHiTBench/MultiHiertt는 원본 헤더 트리가 없어 자동 검증이 불가능하다.
- AIT-QA 재구성판 (원본 격자·HTML 부재로 실행 불가, `results/FINAL_TABLE.md` 각주 1)
- hitab_arith의 P1-guessed 외 나머지 재구성 조합, unaligned 115건 (배치 불가)
