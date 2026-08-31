# HPC RUN_LOG

Every command run, its wall time, and every error / skip. Nothing omitted.

## Phase 1 — 2026-08-30

### 1. selftest
```
./.venv/bin/python analysis/header_path_coverage.py --selftest
```
`selftest ok` — <1s. Checks: markdown char-span math recovers "10"/"20" from a
2-column table, P3 chunk contains both header elements, P2_row text shape,
P4 S2 text shape.

### 2. FIRST ATTEMPT — FAILED (import)
```
python3 analysis/header_path_coverage.py --selftest
```
`ModuleNotFoundError: No module named 'numpy'` — system python3 is not the
project env. All later runs use `./.venv/bin/python`.

### 3. SECOND ATTEMPT — FAILED (frozen population not derivable)
```
PYTHONPATH=. ./.venv/bin/python analysis/header_path_coverage.py \
    --dataset hitab --policy P4_path_cell,P1_fixed_512 --max-queries 50 --verify 3
```
```
RuntimeError: population 'hitab_dev_arith': 39/214 frozen queries are not
derivable by the current code (first: ['7003c112e8d92977d4ed3f0c6dbd320d',
'aa092095f237cde2b45e89b0b2672f9b', '5a498dc4804f5e25a5e3fc7e9a6ad187']).
```
Pre-existing, not caused by this work: the working tree carries uncommitted
edits to `scripts/corpus_dump_vs_cell.py` (+104/-13) and
`scripts/freeze_populations.py` (+24) at session start.

Derivation probe of the other HiTab freezes (all under the same working tree):

| population | result |
|---|---|
| `hitab_dev_arith` | **FAIL** 39/214 not derivable |
| `hitab_dev_lookup_all` | OK, 830 queries |
| `hitab_dev_lookup_single` | OK, 100 queries |
| `hitab_dev_corpus_arith` | OK, 175 queries |

Phase 1 therefore ran on `hitab_dev_lookup_all` (first 50 queries). The
`hitab_dev_arith` population is **NOT MEASURED** until the freeze is repaired
or re-frozen by a human decision.

### 4. Phase 1 trial run — OK
```
PYTHONPATH=. ./.venv/bin/python analysis/header_path_coverage.py \
    --dataset hitab --population hitab_dev_lookup_all \
    --policy P4_path_cell,P1_fixed_512 --max-queries 50 --verify 3
```
wall 10.3s total (corpus load 1s; each policy <1s; the rest is tokenizer load).
corpus 424 tables / 58,759 cells; population 50 queries.

Outputs:
- `results/hpc/hitab_P4_path_cell_records.csv`, `..._summary.json`
- `results/hpc/hitab_P1_fixed_512_records.csv`, `..._summary.json`

## Known deviations / open items

- **P2_row is NOT the repo's existing `row` arm.** `scripts/baseline_comparison_llm.py:row_chunks`
  renders the **leaf** column header (`cp[-1]`); the task spec says
  "행 + 최상위 열 헤더만", so `P2_row` here renders `cp[0]`. The two are
  different measurements. Not yet run.
- Fixed-window chunks (`P1_*`) cut the markdown at token boundaries, which can
  cut mid-cell. A gold cell counts as being in a chunk only when its whole
  field lies inside the window.
- Gold cells absent from the indexed cell set (RealHiTBench drops blank cells)
  are counted separately as `gold_cell_not_in_corpus_cellset`, not as
  NOT_IN_ANY_CHUNK.
- Phase 2 (4 datasets x 6 policies) and Phase 3: **not run.**

## Phase 2 — 2026-08-30

```
for ds in "hitab --population hitab_dev_lookup_all" multihiertt aitqa realhitbench; do
  PYTHONPATH=. ./.venv/bin/python analysis/header_path_coverage.py --dataset $ds --policy all
done
```
전 24칸 성공. 실패·건너뛴 항목 없음. 코퍼스 로드 hitab 1s / multihiertt 3s /
aitqa 0s / realhitbench 6s, 정책당 측정 0~1s.

산출물: `results/hpc/{dataset}_{policy}_records.csv` 24개,
`{dataset}_{policy}_summary.json` 24개, `results/hpc/summary_table.md`.

정합성: `P4_path_cell` HPC_full = 1.0 (4/4). 통과.

### 이 실행에서 드러난 기존 코드의 성질 (수정하지 않음)

`aitqa_corpus`의 `md[tid]`가 행 헤더를 쓰지 않아 `P3_whole_table`이 AIT-QA에서
0.2040이다. 같은 마크다운을 기존 `dump`/`goldtable` arm이 리더에게 넘긴다.
보고만 하고 고치지 않았다.

## 안 한 것

- Phase 3: 실행 안 함. 정책별 검색 파이프라인이 필요한데, 고정 길이 청크는
  기존 임베딩 캐시(`.cache/corpus_dump_vs_cell`, 셀 문장 기준)에 없어
  정책마다 코퍼스 전체를 새로 인코딩해야 한다.
- `hitab_dev_arith` 모집단: 동결 파생 실패로 미측정 (Phase 1 로그 참조).

## 2026-08-31 — AIT-QA 행 헤더 직렬화 버그 수정 및 재측정

### 수정

`scripts/corpus_dump_vs_cell.py:290-291` (`aitqa_corpus`). 이전에는
`head = [" / ".join(cp_of(j)) ...]` 로 **열 헤더만** 쓰고 본문은 `data`의 값만 써서
`row_header`가 `md[tid]`에 전혀 들어가지 않았다. `rp_of(i)`를 선행 열로 붙이도록 고침:

```python
n_h = max((len(rp_of(i)) for i in range(n_r)), default=0)
head = [""] * n_h + [" / ".join(cp_of(j)) for j in range(n_c)]
md[tid] = (["| " + " | ".join(head) + " |", "|" + "---|" * (n_h + n_c)]
           + ["| " + " | ".join(((rp_of(i) + [""] * n_h)[:n_h]
                                 + [str(x) for x in r])) + " |"
              for i, r in enumerate(data)])
```

검증: 113개 표 전부 `len(md)-2 == n_r`, 모든 행의 필드 수가 헤더 행과 동일
(span-math 위반 0). 데이터 행 ragged 0/1315. `--selftest` 통과.

### Task C 정정

2026-08-30에 "영향 파일 28개"라고 보고한 것은 **틀렸다.** 결과 json의 `arms` 키는
arm 카탈로그(설명 사전)이지 실행 기록이 아니다. 실제 실행 기록은 `arms_run`/`summary`다.
`_records.jsonl` 첫 줄로 교차확인한 결과 `dump`/`cell2dump`를 실제로 돌린 AIT-QA 실행은
**9개**다. 나머지 파일은 `cell`/`flat` 등만 돌려 `md[tid]`를 건드리지 않는다.

### 재실행 1 — HPC (AIT-QA 6정책)

기존 12개 파일을 `results/hpc/pre_rowheader_fix/`로 옮긴 뒤 (덮어쓰기 아님):

```
PYTHONPATH=. ./.venv/bin/python analysis/header_path_coverage.py --dataset aitqa --policy all
```
6정책 전부 성공, 정책당 0s.

| policy | 이전 HPC_full | 이후 | 이전 partial | 이후 |
|---|---|---|---|---|
| `P1_fixed_256` | 0.1849 | 0.6778 | 0.5318 | 0.8093 |
| `P1_fixed_512` | 0.2040 | 0.8780 | 0.6139 | 0.9368 |
| `P1_fixed_1024` | 0.2040 | 0.9712 | 0.6457 | 0.9849 |
| `P2_row` | 0.4191 | 0.4191 | 0.8228 | 0.8228 |
| `P3_whole_table` | 0.2040 | 1.0000 | 0.6457 | 1.0000 |
| `P4_path_cell` | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

`P1_fixed_256`의 NOT_IN_ANY_CHUNK가 2 → 1로 바뀌어 분모가 449 → 450이 됐다.
`P2_row`/`P4`는 `md[tid]`를 쓰지 않아 불변 (정합성 확인).

### 재실행 2 — `dump`/`cell2dump` arm

전부 `results/aitqa_rowheader_fix/`에 저장 (기존 파일 무수정).

reader 없는 5개 (S3c, hybrid, 현행 기준 §7):
```
PYTHONPATH=. ./.venv/bin/python scripts/corpus_dump_vs_cell.py \
  --dataset aitqa --data-dir data/aitqa --population aitqa_answer_matched \
  --cell-scheme S3c --budget {512,1024,2048,4096} --retriever hybrid \
  --alpha {0.7,0.5} --seed 42 --out results/aitqa_rowheader_fix/<name>.json
```
실행당 약 45s. 5/5 성공.

2026-08-19~20 판 4개 (S2). run.json에 `cell_scheme`/`alpha`가 없어
CLI 기본값(S2, ALPHA[hybrid]=0.5, ALPHA[dense]=1.0)으로 재구성했고,
`--arms dump,cell2dump,cell,cascade,row,flat`로 검색 레그만 돌렸다. 4/4 성공.

**리더 레그(EM)는 재실행하지 않았다.** 이 4개는 전부 Qwen 7B 리더를 썼고
2026-08-26 이전이라 CLAUDE.md §8이 이미 채점기 `$` 버그로 오염 처리한 구간이다.
→ 이 4개의 EM은 **MEASURED: NO**.

### 재측정 결과 — 전부 하락

| 파일 | arm | osc 이전 | 이후 | Δ | n_tables 이전 | 이후 |
|---|---|---|---|---|---|---|
| `hyb_aitqa_s3c_512_a7` | dump | 0.3415 | 0.2772 | −0.0643 | 2.92 | 2.28 |
| `hyb_aitqa_s3c_512_a7` | cell2dump | 0.4545 | 0.3858 | −0.0687 | 2.83 | 2.29 |
| `hyb_aitqa_s3c_512_a50` | dump | 0.3392 | 0.2905 | −0.0487 | 2.96 | 2.28 |
| `hyb_aitqa_s3c_512_a50` | cell2dump | 0.4501 | 0.3858 | −0.0643 | 2.87 | 2.34 |
| `hyb_aitqa_s3c_1024_a7` | dump | 0.4922 | 0.4479 | −0.0443 | 4.78 | 3.94 |
| `hyb_aitqa_s3c_1024_a7` | cell2dump | 0.6053 | 0.5366 | −0.0687 | 4.63 | 3.89 |
| `hyb_aitqa_s3c_2048_a7` | dump | 0.6120 | 0.5455 | −0.0665 | 8.25 | 6.18 |
| `hyb_aitqa_s3c_2048_a7` | cell2dump | 0.7118 | 0.6541 | −0.0577 | 7.96 | 6.09 |
| `hyb_aitqa_s3c_4096_a7` | dump | 0.7384 | 0.6741 | −0.0643 | 14.31 | 10.32 |
| `hyb_aitqa_s3c_4096_a7` | cell2dump | 0.8315 | 0.7672 | −0.0643 | 13.97 | 10.43 |
| `..._h2h_aitqa_dense_256` | dump | 0.1885 | 0.1330 | −0.0555 | 1.93 | 1.41 |
| `..._h2h_aitqa_dense_256` | cell2dump | 0.2328 | 0.1685 | −0.0643 | 1.81 | 1.36 |
| `..._h2h_aitqa_dense_512` | dump | 0.2439 | 0.2151 | −0.0288 | 2.90 | 2.26 |
| `..._h2h_aitqa_dense_512` | cell2dump | 0.3858 | 0.3392 | −0.0466 | 2.75 | 2.13 |
| `..._hyb_aitqa_512` | dump | 0.3392 | 0.2905 | −0.0487 | 2.96 | 2.28 |
| `..._hyb_aitqa_512` | cell2dump | 0.4523 | 0.3858 | −0.0665 | 2.87 | 2.34 |
| `..._hyb_aitqa_256` | dump | 0.2209 | 0.1574 | −0.0635 | — | — |
| `..._hyb_aitqa_256` | cell2dump | 0.2699 | 0.1929 | −0.0770 | — | — |

`..._hyb_aitqa_256`은 요약 json이 없어 이전 값을 `_records.jsonl`에서 재계산했다.
`n_tables`는 그 파일에 없어 **MEASURED: NO**.

`goldtable` arm은 osc가 1.0으로 불변이고 `tokens`만 310.39 → 452.66 (+142.27)으로
바뀐다. 나머지 arm(`cell`/`cascade`/`cellrow`/`row`/`flat`/`group`/`capped`/`goldcell`)은
전 파일에서 **전 지표 완전 일치** — `md[tid]`를 안 쓴다는 것의 정합성 확인이다.

## Phase 3 — 2026-08-31

계측기 `analysis/stratified_recall.py` 신규 작성. 리더 호출 없음.

```
PYTHONPATH=. ./.venv/bin/python analysis/stratified_recall.py --dataset aitqa --policy all
PYTHONPATH=. ./.venv/bin/python analysis/stratified_recall.py --dataset hitab \
    --data-dir data/hitab --population hitab_dev_lookup_all --policy all
PYTHONPATH=. ./.venv/bin/python analysis/stratified_recall.py --dataset multihiertt --policy all
PYTHONPATH=. ./.venv/bin/python analysis/stratified_recall.py --dataset realhitbench \
    --data-dir data/realhitbench --policy all
PYTHONPATH=. ./.venv/bin/python analysis/stratified_recall.py --dataset realhitbench \
    --data-dir data/realhitbench --policy all --rhb-drop-multimatch
```

25/25 조건 성공. 실패·건너뛴 조건 없음. 전체 약 18분.
색인 구축: 청크 118~143,377개, 조건당 3~22s. `P4_path_cell` 인코딩은
`.cache/corpus_dump_vs_cell`의 기존 캐시에 전부 적중 (텍스트가 동일하므로).
고정 크기 청크와 `P2_row`는 새로 인코딩해 같은 캐시에 기록.

정합성: 층 구분(`full`/`partial`/`NOT_IN_ANY_CHUNK`)을 Phase 2 CSV와 셀 단위로
대조. hitab 830/830, multihiertt 737/737, aitqa 451/451, realhitbench 287/287,
전부 **불일치 0**. `realhitbench_nomulti`는 모집단이 달라 대조 생략.

### 사양과 다르게 한 것 (숨기지 않고 기록)

사양은 "두 군 차이의 paired bootstrap"이었으나 `full`/`partial`은 서로 다른 셀의
**서로소 집합**이라 짝지음이 정의되지 않는다. two-sample percentile bootstrap으로
바꿨다 (두 군 독립 복원추출, 차이의 2.5/97.5 분위수). B=10000, seed=42는 사양 그대로.
p는 (군 × 적중) 2×2 Fisher 정확검정 양측값이고 **보정하지 않았다** (총 40개 검정).

### n<30으로 해석 불가 표기된 칸

- `P4_path_cell` 4/4 데이터셋: HPC_full = 1.0이라 `partial` 군 n=0
- hitab `P1_fixed_1024` partial n=25
- multihiertt `P1_fixed_512` partial n=22, `P1_fixed_1024` partial n=4
- aitqa `P1_fixed_1024` partial n=13
- realhitbench_nomulti `P4_path_cell` partial n=0

`P3_whole_table`은 HPC_full = 1.0 (4/4)이라 사양대로 제외했다.

산출물: `results/hpc/stratified_recall.md`,
`results/hpc/stratified/{dataset}_{policy}_stratified.json` 25개,
`results/hpc/stratified/{dataset}_{policy}_cells.csv` 25개.

## Phase 3b — 2026-08-31

계측기 `analysis/token_equiv_recall.py` 신규 작성. 리더 호출 없음.

```
PYTHONPATH=. ./.venv/bin/python analysis/token_equiv_recall.py --dataset aitqa --policy all
PYTHONPATH=. ./.venv/bin/python analysis/token_equiv_recall.py --dataset hitab \
    --data-dir data/hitab --population hitab_dev_lookup_all --policy all
PYTHONPATH=. ./.venv/bin/python analysis/token_equiv_recall.py --dataset multihiertt --policy all
PYTHONPATH=. ./.venv/bin/python analysis/token_equiv_recall.py --dataset realhitbench \
    --data-dir data/realhitbench --policy all
```

20/20 조건 성공. 실패·건너뛴 조건 없음. 약 10분.

평균 청크 토큰은 코퍼스 전체 청크에 대해 `Budget(BAAI/bge-small-en-v1.5)`로 실측했다.
등가 k = `floor(B / 평균청크토큰)`, 최소 1. 실비용 = k × 평균청크토큰을 같이 실어
세 예산 모두 실제로 등가인지 표에서 확인 가능하게 했다 (오차 최대 −5.0%,
realhitbench `P1_fixed_1024` B=10240에서 9,740 토큰).

Phase 3와 달리 층화하지 않고 전체 gold 셀을 분모로 삼았다. `NOT_IN_ANY_CHUNK`
셀도 분모에 넣었고 정의상 0으로 센다. 그 개수를 표에 병기했다.

### RealHiTBench 표당 셀 수 (질문에 대한 답)

셀 1만개 이상인 표 **0개**. 최대 2,156셀 (`business-table15`, 87x28 격자의 88.5%).
143,377 = 색인 셀이 있는 표 532개 × 평균 269.5셀. 중앙값 179, p99 1,500.
gold 셀을 가진 표 179개 중 1만 셀 이상은 0개다. 비정상 표로 인한 팽창은 없다.

산출물: `results/hpc/token_equiv_recall.md`,
`results/hpc/token_equiv/{dataset}_{policy}_tokeq.json` 20개,
`results/hpc/token_equiv/{dataset}_{policy}_cells.csv` 20개.
