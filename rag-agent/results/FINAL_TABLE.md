# 데이터셋 × 정책 최종 표 — Recall / EM, 오라클 의존 표기

각 칸은 `Recall / EM`. Recall 분모는 gold 셀 수, EM 분모는 쿼리 수.
**(O)** = 데이터셋 원본 주석(오라클) 사용, **(R)** = 파이프라인 재구성.

공통 조건: 하이브리드 검색 α=0.7 (`α·minmax(dense) + (1−α)·minmax(bm25)`,
`BAAI/bge-small-en-v1.5`), greedy fill B_reader=4096 Qwen 토큰,
리더 `Qwen/Qwen2.5-7B-Instruct` rev `a09a35458c702b33eeacc393d103063234e8bc28`
4-bit NF4, temperature=0, seed=42, max_new_tokens=32.
채점은 Phase 4 규칙(R1 상대오차 <0.01 포함, `gold_parts` 버그 수정 후).

## 오라클 판정

| dataset | P1/P3 텍스트 | P2/P4 헤더 경로 | 근거 |
|---|---|---|---|
| hitab | raw published grid + 헤더 행 수만 `pt["n_r"]` (부분 O) | `pt["gold_rp"]` / `pt["gold_cp"]` = `tables/hmt/` 헤더 트리 (O) | `corpus_dump_vs_cell.py:233, 242` |
| aitqa | `row_header`/`column_header`로 조립 (O) | `row_header` / `column_header` (O) | `corpus_dump_vs_cell.py:295-299, 308` |
| realhitbench | HTML 파싱 격자 (R) | `reconstruct_*_paths` + `guess_n_header_*` (R) | `corpus_dump_vs_cell.py:436-441, 451, 468` |
| multihiertt | HTML 파싱 격자 (R) | `reconstruct_*_paths` + `guess_n_header_*` (R) | `baseline_comparison_multihiertt.py:130-135`, `corpus_dump_vs_cell.py:535, 548` |

## 본표

| dataset (n) | 조건 | P1_fixed_512 | P4_path_cell |
|---|---|---|---|
| hitab_lookup (189) | 오라클 | 0.7989 / 0.4444 **(O)** | 0.9312 / 0.6720 **(O)** |
| hitab_lookup (189) | 행만 재구성 | — | 0.9259 / 0.6614 **(부분 R)** |
| hitab_lookup (189) | 완전 재구성 | 0.7989 / 0.4339 **(R)** | 0.9259 / 0.6455 **(R)** |
| hitab_arith (60) | 오라클 | 0.8315 / 0.0333 **(O)** | 0.7921 / 0.0167 **(O)** |
| hitab_arith (60) | 행만 재구성 | — | 0.8090 / 0.0167 **(부분 R)** |
| hitab_arith (60) | 완전 재구성 | 0.8315 / 0.0500 **(R)** | 0.8090 / 0.0333 **(R)** |
| aitqa (60) | 오라클 | 0.7333 / 0.4333 **(O)** | 0.8333 / 0.4167 **(O)** |
| aitqa (60) | 완전 재구성 | 미측정 [1] | 미측정 [1] |
| rhb_fact (60) | (재구성만 존재) | 0.6000 / 0.3167 **(R)** | 0.5692 / 0.3500 **(R)** |
| rhb_num (54) | (재구성만 존재) | 0.4219 / 0.1667 **(R)** | 0.4062 / 0.1296 **(R)** |

hitab의 gold 셀 수: lookup 189 (쿼리당 1), arith 178 (쿼리 60).
aitqa 60 / rhb_fact 65 / rhb_num 64.

### 각주

[1] **AIT-QA 재구성판 미측정.** `data/aitqa/`에는 `aitqa_questions.jsonl`,
`aitqa_tables.jsonl`, `generated_titles.json`만 있고 HTML도 원본 격자도 없다.
표 레코드의 키는 `['column_header', 'data', 'id', 'row_header']`이며 `data`는
데이터 영역만 담는다. `reconstruct_row_paths` / `reconstruct_col_paths`는
헤더 밴드가 상단 `n_header_rows`행·좌측 `n_header_cols`열에 놓인 격자를
입력으로 요구하는데, 그 격자가 배포되지 않았다. 주석을 다시 행·열로 펼쳐
격자를 만들면 재구성기의 입력이 재구성 대상 주석 자체가 되어 순환이므로
실행하지 않았다.

## 재구성 조건의 구성

- **행만 재구성** — `pt["gold_rp"]` → `pt["rec_rp"]`. 열 경로·값·제목·템플릿 유지.
- **완전 재구성 (P4)** — 행·열 모두 `rec_*`.
  `rec_*`는 `reconstruct_{row,col}_paths(raw["texts"], nhr, nhc)`,
  `nhr = min(rows_c)`, `nhc = min(cols_c)` (`point3_reconstruction_cost.py:101-102`).
- **완전 재구성 (P1)** — 헤더 경계를 `guess_n_header_cols` / `guess_n_header_rows`로
  두 번 교대 추정해 `markdown_table(raw, nhr_guess)`로 markdown을 만들고
  데이터 영역을 `(len(texts)−nhr, width−nhc)`로 잡는다.
  경계 추정이 오라클과 일치한 표: **326 / 424**.

## 부속 검정 — McNemar, P1 vs P4-recon (hitab_lookup 189, paired)

| P1 기준 | P1 EM | P4-recon EM | b (P4만) | c (P1만) | Δ | 95% CI | p |
|---|---:|---:|---:|---:|---:|---|---:|
| P1 오라클 | 0.4444 | 0.6455 | 60 | 22 | +0.2011 | [+0.1111, +0.2910] | 3.23e-05 |
| P1 완전 재구성 | 0.4339 | 0.6455 | 59 | 19 | +0.2116 | [+0.1270, +0.2963] | 6.42e-06 |

paired bootstrap B=10000 seed=42. McNemar 정확검정 `binomtest(b, b+c, 0.5)`,
연속성 보정 없음, 다중비교 보정 없음.

## 산출물

- `analysis/taskd_recon.py` → `results/audit/taskd/` (6개 jsonl + `summary.json`)
- `analysis/rec_rp_index.py` → `results/audit/rec_rp/` (행만 재구성, lookup)
- Phase 4 오라클 수치: `results/phase4/reader_records.jsonl`,
  `results/phase4/taskB.json`, `results/phase4/taskB_retrieval_ext129.json`
- 출처 추적 근거: `results/audit/cell_sentence_audit.json`,
  `results/audit/row_path_failure.json`
