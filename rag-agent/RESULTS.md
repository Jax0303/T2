# 현행 수치 — 2026-08-31 기준

이 파일은 리포에 **현재 남아 있는** `results/` 406개 파일에서 인용 가능한 수치만 싣는다.
2026-08-30~31에 측정된 것들이다. 그 이전 단계 산출물 799개는 커밋 `54ed06e`에서
리포지토리에서 제거했고 히스토리에는 그대로 있다:

```
git checkout 753fa2e -- rag-agent/results          # 전량 복원
git checkout 753fa2e -- rag-agent/results/<경로>   # 개별 복원
```

제거된 파일을 인용하던 문장은 `README.md`, `RESEARCH_STRUCTURE.md` 안에 남아 있고
각 문서 머리에 그 사실을 적어 두었다. 이 파일에는 그 수치를 옮겨 오지 않았다 —
재측정하지 않은 값을 새 기준 문서에 싣지 않는다.

**표기.** `MEASURED: NO` = 측정하지 않음. 값은 전부 출처 파일에서 그대로 옮긴 것이고
이 파일에서 계산하거나 반올림한 값은 없다.

---

## 1. 리더 EM — pool × 조건

출처 `results/phase4/FINAL.md`, 원본 레코드 `results/phase4/reader_records.jsonl` (1156행).
리더 `Qwen/Qwen2.5-7B-Instruct` rev `a09a3545…` 4-bit NF4, temperature=0, seed=42,
max_new_tokens=32, B_reader=4096 Qwen 토큰 (greedy fill).
채점은 `analysis/phase4_summary.py: em()` (R1 상대오차 <0.01 포함, `gold_parts` 버그 수정 후).

| pool | dataset | P1_fixed_512 | P4_path_cell | gold_cell |
|---|---|---|---|---|
| hitab_lookup | hitab | 0.4444 (n=189) | 0.6720 (n=189) | 0.9206 (n=189) |
| hitab_arith | hitab | 0.0333 (n=60) | 0.0167 (n=60) | 0.2903 (n=31) |
| aitqa | aitqa | 0.4333 (n=60) | 0.4167 (n=60) | 0.8000 (n=30) |
| rhb_fact | realhitbench | 0.3167 (n=60) | 0.3500 (n=60) | 0.7000 (n=30) |
| rhb_num | realhitbench | 0.1667 (n=54) | 0.1296 (n=54) | 0.4000 (n=30) |

McNemar(P4 vs P1, paired 정확검정) + Holm(m=5, alpha=0.05): `hitab_lookup`만 유의
(delta +0.2275, 95% CI [+0.1376, +0.3175], p=3.28e-06). 나머지 4 pool은 기각 실패이고
b+c가 작아 검정력이 부족하다. 전체 표·필요 n·검정력은 `FINAL.md` §2, §5.

gold_cell 조건 전체 EM 0.7742 (n=310) = 현재 리더 상한.

## 2. 조회셀 EM 갭 분해 (hitab_lookup 189, 오늘)

출처 `results/lookup_gap/gap.md`, `gap.json`. 계측기 `analysis/lookup_gap.py`.
Phase 4 레코드 재채점만 했고 모델을 다시 돌리지 않았다. 189건 전부 gold 셀 1개다.

| policy | n | EM | gold 셀 컨텍스트 포함률 | 오답 | A 조회 실패 | B 컨텍스트엔 있음 | C gold_cell에서도 오답 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1_fixed_512 | 189 | 0.4444 | 0.7989 | 105 | 37 | 58 | 10 |
| P4_path_cell | 189 | 0.6720 | 0.9312 | 62 | 12 | 38 | 12 |
| gold_cell | 189 | 0.9206 | 1.0000 | 15 | 0 | 0 | 15 |

B = 정답 셀이 컨텍스트에 있었고 gold_cell 조건에서는 맞힌 건. C = gold_cell 조건에서도 틀린 건.

### 2.1 B 버킷 오답값의 출처

출처 `results/lookup_gap/distractor.md`, `distractor.json`. 계측기 `analysis/lookup_distractor.py`.
예측값을 같은 컨텍스트 안 각 청크가 주장하는 값과 대조했다.

| origin | P4_path_cell (n=38) |
|---|---:|
| 같은 행, 다른 열 | 13 |
| 같은 열, 다른 행 | 9 |
| 다른 표 | 5 |
| 같은 표 다른 위치 | 1 |
| 컨텍스트에 그 값 없음 | 10 |

gold_rank == 0인데 오답: 15/38.
gold 셀 문장이 gold 답을 담고 있지 않은 건: 3 (전부 부호 차이 — gold `[41]` ↔ 셀 `-41.0`,
gold `[2.8]` ↔ `-2.8`, gold `[7]` ↔ `-7.0`. 채점 규칙에 부호 정규화가 없다).

P1_fixed_512 58건: 예측값이 컨텍스트 안에 있음 52 / 없음 6.
P1 청크는 512토큰 블록이라 행·열 귀속은 계산하지 않았다 (N/A).

### 2.2 틀린 헤더의 축, 그리고 컨텍스트에 없는 값

출처 `results/lookup_gap/b_detail.md`, `b_detail.json`. 계측기 `analysis/lookup_b_detail.py`.
셀 문장을 `(표 제목, 행 경로, 열 경로, 값)`으로 파싱해 gold 셀 문장과 대조했다.

이웃 셀에서 값을 가져온 28건:

| 틀린 헤더 | n |
|---|---:|
| 열 헤더만 | 12 |
| 행 헤더만 | 11 |
| 양쪽 | 3 |
| gold 셀 문장 자체 | 1 |
| 주소 중복 (행·열 경로가 gold와 완전히 같은 다른 셀) | 1 |

축별 경로 차이 (`equal` = 그 축은 gold와 동일, `sibling` = 부모까지 같고 마지막 마디만 다름):

| 축 | equal | sibling | prefix | other |
|---|---:|---:|---:|---:|
| 행 | 14 | 6 | 1 | 7 |
| 열 | 13 | 8 | 0 | 7 |

`gold 셀 문장 자체` 1건은 §2.1의 부호 불일치 건이다 (gold `[41]`, 색인된 셀 값 `-41.0`,
gold_rank 0). `주소 중복` 1건은 같은 표 안에서 행·열 경로가 gold와 완전히 같은 다른 셀이
rank 45에 있었고 gold 셀은 rank 71이었다.

컨텍스트에 없는 값 10건:

| gold | pred | pred/gold | gold의 배율 | 질문 본문의 수 | 컨텍스트 헤더·제목의 수 |
|---|---|---:|---|---|---:|
| [17718556.0] | 17718.556 | 0.001 | /1000 | 아니오 | 0 |
| [51.5] | 32.7 | 0.634951 | 아니오 | 아니오 | 0 |
| [2.8] | 16.3 | 5.821429 | 아니오 | 아니오 | 0 |
| [13.4] | 17.5 | 1.305970 | 아니오 | 아니오 | 0 |
| [7] | 2.5% | 0.357143 | 아니오 | 아니오 | 0 |
| [13.5] | 46.1 | 3.414815 | 아니오 | 아니오 | 0 |
| [25.0] | 3.0 | 0.120000 | 아니오 | 아니오 | 0 |
| [70.0] | 33 | 0.471429 | 아니오 | 아니오 | 0 |
| [101123.0] | 96500 | 0.954283 | 아니오 | 아니오 | 0 |
| [12632.0] | 325.719 | 0.025785 | 아니오 | 아니오 | 14 |

배율(×/÷ 10·100·1000, 부호)로 설명되는 건 1/10. 질문 본문에 있던 수를 그대로 답한 건 0/10.
컨텍스트의 헤더·제목에 그 수가 있던 건 1/10. 나머지 8건은 컨텍스트 어디에도 없는 값이다.

### 2.3 검색 정확도 (Recall 아님) — 순위 기준

출처 `results/lookup_gap/retrieval.md`, `retrieval.json`.
계측기 `analysis/lookup_retrieval.py`. 후보 목록은 저장된 top-200,
분모는 쿼리 189건. 830건 코퍼스 전체 랭킹은 §7.

| 지표 | P4_path_cell | P1_fixed_512 |
|---|---:|---:|
| R@1 | 0.6085 | 0.5714 |
| R@5 | 0.7831 | 0.7619 |
| R@10 | 0.8466 | 0.8307 |
| R@50 | 0.9153 | 0.9259 |
| R@200 | 0.9471 | 0.9788 |
| MRR | 0.6918 | 0.6536 |
| gold 발견 시 순위 중앙값 | 0 | 0 |
| 컨텍스트에 들어간 청크 수 평균 | 70.58 | 7.56 |

순위(위 표)와 예산 절단은 다른 축이다. greedy fill 컨텍스트 포함률(= §1~§3에서
보고한 Recall)은 P4 0.9312 / P1 0.7989다. P1은 후보 순위로는 P4보다 높은 R@200을
내지만 컨텍스트에 7.56청크만 들어가 절단에서 잃는다.

gold 셀이 1위가 아니었던 P4 64건, gold 위의 셀 1067개 (그중 878개가 컨텍스트에 진입):

| 관계 | 셀 수 | 비율 |
|---|---:|---:|
| 다른 표 | 710 | 0.6654 |
| 같은 열 경로 | 128 | 0.1200 |
| 형제 (부모까지 같음, 다른 위치) | 122 | 0.1143 |
| 같은 행 경로 | 67 | 0.0628 |
| 파싱 불가 | 29 | 0.0272 |
| 같은 표 그 외 | 11 | 0.0103 |

P1은 청크가 512토큰 블록이라 헤더 관계를 정의할 수 없다 (관계 표 없음).

### 2.4 830건 전체 — gold 셀을 앞지른 셀

출처 `results/lookup_gap/above830.md`, `rank830/*_above.jsonl`.
`analysis/cell_rank_dump.py --dump-above 20`으로 재랭킹했고, 같이 나온 ranks 파일은
`results/rank/`의 커밋본과 **바이트 동일**하다 (재현 확인).
예산 없음, 코퍼스 전체 랭킹, S3c, hybrid α=0.7.

gold 셀이 1위가 아닌 쿼리 **402 / 830**. gold 위 셀 총 54,418개
(쿼리당 중앙값 4, rank 2000에서 절단된 쿼리 15건).

| 관계 | 셀 수 | 비율 | 1위 셀이 이 관계인 쿼리 수 | 비율 |
|---|---:|---:|---:|---:|
| other_table | 50,748 | 0.9326 | 128 | 0.3184 |
| sibling_elsewhere | 1,246 | 0.0229 | 25 | 0.0622 |
| same_table_far | 1,012 | 0.0186 | 11 | 0.0274 |
| same_col | 995 | 0.0183 | 123 | 0.3060 |
| same_row | 408 | 0.0075 | 112 | 0.2786 |
| same_address | 9 | 0.0002 | 3 | 0.0075 |

셀 단위로 세면 93.3%가 다른 표지만(§7 STAGE1_BOARD §B와 같은 방향), **gold 대신 1위에 온
셀만 보면 68.2%가 gold와 같은 표**다 (같은 열 경로 30.6 + 같은 행 경로 27.9 + 형제 6.2
+ 그 외 3.5). gold 표 안의 셀이 하나라도 gold를 앞지른 쿼리는 362/402 (0.9005).
gold_rank는 2위 안 113건, 10위 안 267건, 200위 밖 41건이다.

### 2.5 실패 케이스 원장 (수동 확인용)

`results/lookup_gap/failures.xlsx`. 계측기 `analysis/lookup_failures_xlsx.py`.

| 시트 | 행 | 내용 |
|---|---:|---|
| `reader_failures` | 182 | 리더가 틀린 전 건 (P1 105 / P4 62 / gold_cell 15). 버킷, 값 출처, 틀린 헤더 축, gold 셀 문장, 값을 가져온 청크 |
| `retrieval_189` | 74 | Phase 4 189건 중 gold가 1위가 아닌 쿼리 (top-200 후보 기준) |
| `above_gold_189` | 3,067 | 위 74건에서 gold보다 위에 있던 청크 전부, 컨텍스트 진입 여부 포함 |
| `retrieval_830` | 402 | 830건 전체에서 gold가 1위가 아닌 쿼리, 관계별 셀 수 |
| `above_gold_830` | 3,285 | 각 쿼리의 gold 위 상위 20셀 (행·열 경로, 값, 관계) |

### 2.6 다중 셀 조회 — 모집단 동결과 검색 정확도

모집단 `populations/hitab_dev_multicell_lookup.txt` **n=94**, 2026-09-01 동결
(`scripts/freeze_populations.py: hitab_dev_multicell_lookup`).
필터: `len(gold_operands)>=2 and aggregation not in ARITH and 헤더 트리 빌드 and 전 피연산자 격자 안`.
`hitab_dev_lookup_all`(830, m=1) 및 `hitab_dev_corpus_arith`(175, 산술)와 서로소.

두 유형은 `results/lookup_gap/multicell_pop.md` 참조 (A: 답 원소 1개 60건 /
B: 답이 목록 34건). 검색만 측정했고 리더는 태우지 않았다
(`results/lookup_gap/multicell_retrieval.md`, `rank_multicell/`).
setEM@k = gold 셀 **전부**가 상위 k 안. setEM@1은 정의상 0이다.

| 층 | n | setEM@5 | setEM@10 | setEM@50 | setEM@200 | Recall@10 | 표 R@1 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A. 답 원소 1개 | 60 | .2000 | .2667 | .5000 | .7167 | .4451 | .5667 | .4836 |
| B. 답이 목록 | 34 | .6176 | .6176 | .8235 | .9412 | .6838 | .7647 | .6306 |
| 다중 셀 전체 | 94 | .3511 | .3936 | .6170 | .7979 | .5315 | .6383 | .5368 |
| (대조) m=1, 830건 | 830 | — | .8373 | .9120 | — | .8373 | .8458 | .6299 |

오라클 표 게이팅(gold 표 안에서만 랭킹, 94건) setEM: @5 .4681 / @10 .6277 / @20 .7234 / @50 .8830.
setEM@10 실패 57건에서 gold를 앞지른 셀 중 같은 표 출신은 4.1%,
gold 위 서로 다른 표 수 중앙값 11.

`MEASURED: NO` — 다중 셀 조회의 리더 EM(어느 정책으로도 미실행), 유형 A의 조건 셀만
따로 본 순위.

### 2.7 다중 셀 조회 — 색인 단위 (셀 / 행 / 표)

출처 `results/lookup_gap/multicell_units.md`. 계측기 `analysis/multicell_unit_bench.py`.
같은 코퍼스를 세 단위로 랭킹하고 같은 규칙으로 채점했다 (상위 k 청크가 **합쳐서**
gold 셀 전부를 덮으면 1). 리더 없음.

**gold 셀의 위치: 94/94가 한 표 안, 68/94가 한 행 안, 20/94가 한 열 안**
(행 수 중앙값 1, 열 수 중앙값 2). 셀 색인만 집합을 여러 청크로 맞춰야 한다.

| 단위 | 청크 수 | setEM@1 | setEM@5 | setEM@10 | setEM@20 | setEM@50 |
|---|---:|---:|---:|---:|---:|---:|
| cell (S3c) | 58,759 | .0000 | .3511 | .3936 | .4681 | .6170 |
| row | 7,196 | .1915 | .4362 | .5213 | .6489 | .7872 |
| table | 424 | .6277 | .8298 | .8936 | .9362 | .9787 |

k는 같은 비용이 아니다. **리더가 받는** 평균 토큰으로 등가 k를 잡으면
(표는 색인 텍스트 81.72토큰으로 랭킹하지만 격자 전체 569.69토큰이 컨텍스트에 들어간다):

| 단위 | 리더가 받는 평균 토큰 | 512 | 1024 | 2048 | 4096 |
|---|---:|---|---|---|---|
| cell | 38.19 | .4362 (k=13) | .5106 (k=26) | .6170 (k=53) | .7340 (k=107) |
| row | 53.87 | .5106 (k=9) | .6383 (k=19) | .7234 (k=38) | .8191 (k=76) |
| table | 569.69 | .6277 (k=1) | .6277 (k=1) | .7553 (k=3) | **.8723 (k=7)** |

같은 예산에서 셀 색인이 세 단위 중 가장 낮다. 행은 네 예산 모두에서 셀보다 높고,
표는 512와 4096에서 가장 높다. 이 비교는 **다중 셀 조회 94건에 한정**된다 —
m=1 모집단에서 구조적 장치가 S3c 셀에 진 결과(`CLAUDE.md` §5, §6)와 다른 모집단이다.

`MEASURED: NO` — 세 단위의 리더 EM (검색만 쟀다), m=1 모집단에서의 같은 표.


## 3. Recall 분모 — 셀 단위와 쿼리 단위

출처 `results/audit/recall_denominators.json`. 보고된 Recall은 전부 **셀 단위**다.

| arm | 셀 적중/셀 수 | recall_cell | 쿼리 적중/쿼리 수 | recall_query |
|---|---|---:|---|---:|
| phase4 · P1_fixed_512 · hitab_lookup | 151/189 | 0.7989 | 151/189 | 0.7989 |
| phase4 · P1_fixed_512 · hitab_arith | 148/178 | 0.8315 | 43/60 | 0.7167 |
| phase4 · P1_fixed_512 · aitqa | 44/60 | 0.7333 | 44/60 | 0.7333 |
| phase4 · P1_fixed_512 · rhb_fact | 39/65 | 0.6000 | 35/60 | 0.5833 |
| phase4 · P1_fixed_512 · rhb_num | 27/64 | 0.4219 | 21/54 | 0.3889 |
| phase4 · P4_path_cell · hitab_lookup | 176/189 | 0.9312 | 176/189 | 0.9312 |
| phase4 · P4_path_cell · hitab_arith | 141/178 | 0.7921 | 41/60 | 0.6833 |
| phase4 · P4_path_cell · aitqa | 50/60 | 0.8333 | 50/60 | 0.8333 |
| phase4 · P4_path_cell · rhb_fact | 37/65 | 0.5692 | 37/60 | 0.6167 |
| phase4 · P4_path_cell · rhb_num | 26/64 | 0.4062 | 20/54 | 0.3704 |
| taskd · P4_recon_row · hitab_lookup | 175/189 | 0.9259 | 175/189 | 0.9259 |
| taskd · P4_recon_row · hitab_arith | 144/178 | 0.8090 | 43/60 | 0.7167 |
| taskd · P4_recon · hitab_lookup | 175/189 | 0.9259 | 175/189 | 0.9259 |
| taskd · P4_recon · hitab_arith | 144/178 | 0.8090 | 43/60 | 0.7167 |
| taskd · P1_guessed_boundary · hitab_lookup | 151/189 | 0.7989 | 151/189 | 0.7989 |
| taskd · P1_guessed_boundary · hitab_arith | 148/178 | 0.8315 | 43/60 | 0.7167 |
| rec_rp · P4_recon_row · hitab_lookup | 저장 안 됨 | — | 175/189 | 0.9259 |
| unaligned · P1_fixed_512 · hitab_lookup_80 | 71/80 | 0.8875 | 71/80 | 0.8875 |

`rec_rp`만 쿼리 단위 all() 플래그로 저장돼 셀 단위 재계산이 불가능하다.

## 4. 재구성 조건 (오라클 주석 없이)

출처 `results/FINAL_TABLE.md`, `results/audit/taskd/summary.json`, `results/audit/rec_rp/summary.json`.
각 칸 `Recall / EM`. (O) = 원본 주석, (R) = 파이프라인 재구성.

| dataset (n) | 조건 | P1_fixed_512 | P4_path_cell |
|---|---|---|---|
| hitab_lookup (189) | 오라클 | 0.7989 / 0.4444 (O) | 0.9312 / 0.6720 (O) |
| hitab_lookup (189) | 행만 재구성 | — | 0.9259 / 0.6614 (부분 R) |
| hitab_lookup (189) | 완전 재구성 | 0.7989 / 0.4339 (R) | 0.9259 / 0.6455 (R) |
| hitab_arith (60) | 오라클 | 0.8315 / 0.0333 (O) | 0.7921 / 0.0167 (O) |
| hitab_arith (60) | 행만 재구성 | — | 0.8090 / 0.0167 (부분 R) |
| hitab_arith (60) | 완전 재구성 | 0.8315 / 0.0500 (R) | 0.8090 / 0.0333 (R) |
| aitqa (60) | 오라클 | 0.7333 / 0.4333 (O) | 0.8333 / 0.4167 (O) |
| aitqa (60) | 완전 재구성 | 미측정 [1] | 미측정 [1] |
| rhb_fact (60) | (재구성만 존재) | 0.6000 / 0.3167 (R) | 0.5692 / 0.3500 (R) |
| rhb_num (54) | (재구성만 존재) | 0.4219 / 0.1667 (R) | 0.4062 / 0.1296 (R) |

[1] AIT-QA는 원본 격자·HTML이 배포되지 않아 재구성기의 입력을 만들 수 없다.
    상세는 `results/FINAL_TABLE.md` 각주 1.

P1 헤더 경계 추정이 오라클과 일치한 표: **326 / 424**.
McNemar(P1 vs P4-recon, hitab_lookup 189): P1 오라클 기준 delta +0.2011 (p=3.23e-05),
P1 완전 재구성 기준 delta +0.2116 (p=6.42e-06). 상세 `FINAL_TABLE.md`.

## 5. 셀 문장·헤더 경로 감사

- **정렬 (`results/audit/hitab_align_recheck.json`)** — dev 540표 중 정렬 성공 424,
  실패 116 (0.2148). 열 경로 복원 정확도 0.9749 (n=3464), 행 경로 0.8202 (n=7196).
- **정렬 실패의 영향 (`unaligned_impact.json`)** — 실패 116표에 걸린 dev 쿼리 309건.
  그중 단일 피연산자 195, 산술 적격 39. 동결 모집단
  (`hitab_dev_lookup_all` 830, `hitab_dev_corpus_arith` 175)에 포함된 건 **0**.
  분모에서 빠진 것이지 오답으로 세어진 것이 아니다.
- **정렬 실패 표의 P1 (`unaligned_p1.json`)** — 116표 전부 markdown 생성 성공,
  청크 169개 추가. 채점 가능 80건에서 Recall 0.8875 (71/80), 컨텍스트 청크 평균 7.61.
  P4는 검증된 헤더 경로가 없어 실행하지 않았다.
- **셀 문장 값 대조 (`cell_sentence_audit.json`)** — hitab 58,759셀.
  markdown 대비 불일치 19,653 (0.3345), 원본 값 대비 18,411 (0.3133).
  최다 분류는 `numeric_equal_text_differs` (각각 16,333 / 16,493),
  다음이 천단위 구분자 927. 헤더 트리 대비 불일치는 1,336 (0.0227, 전부 열 경로).
- **행 경로 실패 분류 (`row_path_failure.json`)** — 7,196 경로 중 일치 5,902 (0.8202).
  불일치 1,294 = 조상 누락 1,201 (불일치의 0.9281) / 조상 과잉 80 / 기타 12 / 서로소 1.
- **행 경로 실패 × 검색 (`row_path_vs_retrieval.json`, P4/hitab_lookup 189)** —
  일치 층 n=137 Recall 0.9124 EM 0.6277, 조상 누락 층 n=37 Recall 1.0 EM 0.8108,
  기타 불일치 n=15 Recall 0.9333 EM 0.7333.
  일치−누락 차이: Recall −0.0876 [−0.1387, −0.0438], EM −0.1831 [−0.3277, −0.0282]
  (two-sample bootstrap B=10000 seed=42). 배포된 P4 코퍼스는 **gold** 행 경로를 색인하므로
  이 층은 색인된 문장이 아니라 그 행의 성질로 쓴 것이다 (파일 내 note).

## 6. Phase 2 — 청킹 정책별 헤더 경로 손실 (HPC)

출처 `results/hpc/summary_table.md` (전체 표는 그 파일). LLM·임베딩 없음, 문자열 매칭.
각 칸 HPC_full: hitab / multihiertt / aitqa / realhitbench.

| policy | hitab | multihiertt | aitqa | realhitbench |
|---|---:|---:|---:|---:|
| `P1_fixed_256` | 0.5328 | 0.8096 | 0.6778 | 0.5248 |
| `P1_fixed_512` | 0.8099 | 0.9701 | 0.8780 | 0.7527 |
| `P1_fixed_1024` | 0.9699 | 0.9946 | 0.9712 | 0.8526 |
| `P2_row` | 0.2241 | 0.5305 | 0.4191 | 0.6167 |
| `P3_whole_table` | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `P4_path_cell` | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

P4의 1.0은 측정 결과가 아니라 구현 정합성 확인이다 (설계상 헤더 경로를 포함한다).
2026-08-31 정정: AIT-QA 6칸은 `md[tid]`가 행 헤더를 쓰지 않던 버그를 고쳐 재측정했다.
수정 전 값은 `results/hpc/pre_rowheader_fix/`에 보존.

- Phase 3 층화 Recall@10/@50: `results/hpc/stratified_recall.md` (40개 검정, 다중비교 보정 없음).
- Phase 3b 토큰 등가 Recall: `results/hpc/token_equiv_recall.md`.

## 7. 1단계 보드 — 예산 없는 셀 검색

출처 `results/rank/STAGE1_BOARD.md` (2026-08-30), 계측기 `analysis/cell_rank_dump.py`.
S3c, hybrid α=0.7, `BAAI/bge-small-en-v1.5`, 리더 호출·토큰 예산 없음.

| 모집단 | n | R@1 | R@10 | R@50 | setEM@10 | setEM@50 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| hitab_dev_lookup_all | 830 | .5157 | .8373 | .9120 | .8373 | .9120 | .6299 |
| hitab_dev_corpus_arith | 175 | .1956 | .6282 | .7552 | .4743 | .6514 | .4605 |
| multihiertt | 399 | .1542 | .5387 | .7406 | .4561 | .6792 | .3691 |
| aitqa | 451 | .2550 | .5322 | .7118 | .5322 | .7118 | .3553 |
| realhitbench | 231 | .1674 | .3610 | .4784 | .3506 | .4675 | .2482 |

setEM@10 실패 건에서 gold를 앞지른 셀 중 같은 표 출신 비율: .057 / .087 / .025 / .026 / .097.
개입 2건(표 사전확률 혼합, top-T 표 좁히기)은 둘 다 기각 — 최대 이득 +.0229, 전부 n.s.
표·오라클 게이팅 상한은 `STAGE1_BOARD.md` C~F.

RealHiTBench gold 셀 231건 중 61건(26.4%), AIT-QA 451건 중 34건(7.5%)이 답 문자열 매칭으로
복원된 오염 gold다 (`CLAUDE.md` §8). 두 코퍼스 수치는 하한이다.

## 8. 검색축 — 하이브리드 α sweep (사전등록)

출처 `results/retrieval_to_90_verdict.json`, 사전등록 `PREREG-2026-08-30-*`,
예측 커밋 `5c9bec4`. S3c, 예산 512, LLM 없음. 값은 OSC / gold_table_any / 컨텍스트 표 수.

| dataset | α=0.5 | α=0.7 | α=1.0 (dense) |
|---|---|---|---|
| hitab (830) | — | .8458 / .9253 / 2.32 | .7892 / .9181 / 2.56 |
| multihiertt (399) | .6366 / .8521 / 10.78 | .6115 / .8622 / 12.11 | .4962 / .7995 / 15.24 |
| aitqa (451) | .6497 / .7827 / 9.4 | .6519 / .7938 / 9.7 | — |
| realhitbench (231) | .4026 / .8009 / 4.14 | .4026 / .8225 / 4.29 | .3550 / .7879 / 4.62 |

판정: R1 PASS (어느 코퍼스도 OSC .9 미달, 최대 .8458), R2 **FAIL**
(α=0.5 > α=0.7 예측이 두 곳 동률·한 곳 경계 p=.0755; 반대 분기도 성립 안 함),
R3 PASS, R4 PASS. 사전등록 밖 관측: hybrid가 dense를 4/4 코퍼스에서 이긴다
(RHB +.048 p=.0127, MH +.115 p<1e-4).

## 9. 예산 사다리 (사전등록)

출처 `results/budget_1024_verdict.json` (예측 `bf9484b`),
`results/budget_ladder_verdict.json` (예측 `43825e0`). hybrid α=0.7, S3c, LLM 없음.
값은 OSC / gold_table_any / 컨텍스트 표 수.

| dataset | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| hitab | .8458 / .9253 / 2.32 | .8819 / .9458 / 2.88 | .9133 / .9590 / 4.00 | .9313 / .9711 / 5.80 |
| aitqa | .6519 / .7938 / 9.7 | .7317 / .8537 / 14.1 | .8137 / .9246 / 21.66 | .8825 / .9645 / 33.56 |
| multihiertt | .6115 / .8622 / 12.11 | .7118 / .9023 / 19.1 | .7669 / .9248 / 30.96 | .8371 / .9599 / 51.5 |
| realhitbench | .4026 / .8225 / 4.29 | .4632 / .8745 / 5.60 | .5152 / .8874 / 7.42 | .5844 / .9048 / 10.48 |

512→1024 paired OSC (McNemar): hitab .846→.882 (32:2, p<1e-4),
multihiertt .612→.712 (40:0), aitqa .652→.732 (36:0), realhitbench .403→.463 (14:0, p=1e-4).
B2 **FAIL** — MultiHiertt 이득 +.1003이 사전등록 상한 +.08을 넘었다.
사다리 판정 L1~L5 전부 PASS (HiTab만 2048에서 .9133으로 .9를 넘고, 4096에서도 .9를 넘는
코퍼스는 HiTab뿐, RealHiTBench는 4096에서도 .5844).

## 10. Phase 5 — 산술 리더 교체

출처 `results/phase5/`. 대상 pool은 `hitab_arith` / `rhb_num`.

- **VRAM (`vram.json`)** — `Qwen2.5-7B-Instruct` 4-bit NF4 weights 5302.4 MiB, peak 5425.5 MiB.
- **Task A (`taskA.json`)** — `Qwen2.5-Math-7B-Instruct` (rev `ef9926d7…`, weights 5302.3 MiB)
  스모크 3건 **3건 모두 max_new_tokens=32에서 잘림**, EOS 도달 0.
  `Qwen2.5-Coder-7B-Instruct` (rev `c03e6d35…`, weights 5312.3 MiB) 스모크 3건 잘림 0.
- **Task B (`taskB.json`)** — Coder, max_new_tokens=128 시험 n=10:
  파싱 실패 3 (0.30, 전부 허용 목록 6종 밖 연산자 `percent`/`percentage`/`same`),
  생성 토큰 min 20 / max 60 / 평균 30.4, 캡 도달 0.
  집계 연산자 분포 — 모집단 175: div 80, sum 36, opposite 31, diff 17, average 6, range 5.
  표본 60: div 29, opposite 10, sum 8, diff 7, range 3, average 3.
- **Task C — 미실행.** 6조건 × 114 쿼리 = 684회. 사양과 컨텍스트 출처는 `HANDOFF.md`.

## 11. 아직 측정하지 않은 것

- Phase 5 Task C (위 §10).
- `results/audit/manual_check_{rhb,mh}.xlsx` 채점 — 사람이 할 일.
  RealHiTBench/MultiHiertt는 원본 헤더 트리가 없어 자동 검증 불가.
- AIT-QA 재구성판 (§4 각주 1).
- hitab_arith의 P1-guessed 외 나머지 재구성 조합, 정렬 실패 115건.
- 다중 셀 조회 94건의 리더 EM (§2.6).

## 12. 참조

- 채점 규칙 전문 — `results/phase4/FINAL.md` §6
- 사전등록 — `PREREGISTER.md` 개정 1~6, `PREREG-2026-08-30-*.md`
- 버그 수정 이력 — `BUGFIX_LOG.md`
- 인수인계 — `HANDOFF.md`
