# 사전 기록 — 리더 사다리: 단일 셀 → 다중 셀 → 산술 (2026-09-05)

**실행 전 작성.** 개입이 아니라 측정이고 arm 을 고르지 않는다 (arm 은 p0 α=0.8
`--title-mode page`, 2026-09-02 확정). 그래도 이 저장소 규칙대로 예측을 숫자로 먼저 적는다.

## 사용자 지시 (2026-09-05)

1. 표 검색과 셀 검색을 **같은 차원**에서 본다 — 표를 맞혔을 때 그 표 안에서 셀을
   맞히는가. 지표는 `analysis/stage1_board.py` 의 `셀|표@k` (오라클 표 게이팅).
2. 그 다음 로컬 리더로 **단일 셀 조회 → 다중 셀 조회 → 산술** 순으로 답변을 잰다.
   산술은 gold 셀이 **하나라도** 빠지면 오답으로 본다 (= `gold_in_ctx < m` 이면
   `EM|안됨` 열, 정의상 정답이 나오면 안 된다).

⚠️ `CLAUDE.md` §7 "산술 모집단에 로컬 7B를 쓰지 말 것"과 충돌한다. 사용자가 이번에
명시적으로 로컬 리더로 산술까지 재라고 했으므로 그대로 잰다. 다만 **`gold` 조건(정답
셀만 주입)의 EM 을 반드시 같이 보고**해서 검색 실패와 리더 실패를 가른다.

## 실행

`analysis/retrieved_answer_em.py`, 리더 `Qwen/Qwen2.5-7B-Instruct` 4-bit NF4 temp=0
seed=42 max_new_tokens=32, 채점 `phase4_summary.em`. 조건 `gold` / `top1` / `top3` /
`top10` / `orc10` (산술은 `top20` 추가 — m 최대 12 라 top10 이 정의상 못 담는 질의가 있다).

| 레그 | 모집단 | n | 출력 |
|---|---|---:|---|
| 단일 셀 (dev) | `hitab_dev_lookup_all` | 830 | 이미 있음 `results/answer_ret/p0_dev_all.jsonl` |
| **단일 셀 (test)** | `hitab_test_lookup_all` | 769 | `results/stage3/single_test.jsonl` (이월 항목) |
| 다중 셀 (dev/test) | `hitab_{dev,test}_lookup_multi` | 33 / 31 | `results/stage3/multi_{dev,test}.jsonl` |
| 산술 (dev/test) | `hitab_{dev,test}_corpus_arith` | 175 / 183 | `results/stage3/arith_{dev,test}.jsonl` |

## 예측 (실행 전)

근거: dev 단일 셀 `gold` .9024 / `top10` .7205 (`VERDICT_ORC.md`); 다중 셀 gold 상한
dev .7879 / test .9032 (`results/multicell/CEILING.md`); 산술 gold_cell .2903 (n=31,
Phase 4, 4096 토큰 greedy fill); 산술 all-covered@10 (p0) dev .5314.

| 레그 | 조건 | 예측 EM | 구간 |
|---|---|---:|---|
| 단일 test | gold | .89 | .87–.91 |
| 단일 test | top10 | .70 | .67–.73 |
| 단일 test | top3 | .69 | .66–.72 |
| 다중 dev/test | gold | .79 / .90 | (재실행, 같은 프롬프트면 동일해야 함) |
| 다중 dev/test | top10 | .55 | .45–.65 |
| 산술 dev | gold | .35 | .25–.45 (셀만 넣으면 Phase 4 보다 오른다) |
| 산술 dev | top10 | .15 | .10–.22 |
| 산술 dev | top20 | .17 | .10–.25 |
| 산술 dev | orc10 | .25 | .18–.32 |

기각 조건 없음 (측정). 결과가 구간 밖이면 그대로 적는다.
