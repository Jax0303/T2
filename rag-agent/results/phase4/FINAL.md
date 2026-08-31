# Phase 4 FINAL

행 1156. `Qwen/Qwen2.5-7B-Instruct` (rev a09a35458c702b33eeacc393d103063234e8bc28) 4-bit NF4, temperature=0, seed=42, max_new_tokens=32, B_reader=4096 Qwen 토큰 (greedy fill).

채점 코드는 `BUGFIX_LOG.md`의 `gold_parts()` 수정을 반영한 상태다. 규칙 변경이 아니라 사전등록 규칙(천단위 콤마 제거)이 실행되지 않던 구현 버그의 수정이다.

## 0. 수정 전후 EM (pool x 조건)

| pool | 조건 | n | 수정 전 EM | 수정 후 EM | 변경 |
|---|---|---:|---:|---:|---:|
| hitab_lookup | P1_fixed_512 | 189 | 0.4444 | 0.4444 | 0 |
| hitab_lookup | P4_path_cell | 189 | 0.6720 | 0.6720 | 0 |
| hitab_lookup | gold_cell | 189 | 0.9206 | 0.9206 | 0 |
| hitab_arith | P1_fixed_512 | 60 | 0.0333 | 0.0333 | 0 |
| hitab_arith | P4_path_cell | 60 | 0.0167 | 0.0167 | 0 |
| hitab_arith | gold_cell | 31 | 0.2903 | 0.2903 | 0 |
| aitqa | P1_fixed_512 | 60 | 0.4167 | 0.4333 | +1 |
| aitqa | P4_path_cell | 60 | 0.3667 | 0.4167 | +3 |
| aitqa | gold_cell | 30 | 0.7000 | 0.8000 | +3 |
| rhb_fact | P1_fixed_512 | 60 | 0.3167 | 0.3167 | 0 |
| rhb_fact | P4_path_cell | 60 | 0.3500 | 0.3500 | 0 |
| rhb_fact | gold_cell | 30 | 0.7000 | 0.7000 | 0 |
| rhb_num | P1_fixed_512 | 54 | 0.1667 | 0.1667 | 0 |
| rhb_num | P4_path_cell | 54 | 0.1296 | 0.1296 | 0 |
| rhb_num | gold_cell | 30 | 0.4000 | 0.4000 | 0 |

변경 행 수: **0→1 7행, 1→0 0행** (전 1156행 중).

| pool | 조건 | query_id | pred_parsed | gold_answer |
|---|---|---|---|---|
| aitqa | P1_fixed_512 | q-401 | `6180` | `6,180` |
| aitqa | P4_path_cell | q-118 | `4311` | `4,311` |
| aitqa | P4_path_cell | q-146 | `1179` | `1,179` |
| aitqa | P4_path_cell | q-196 | `13200` | `13,200` |
| aitqa | gold_cell | q-118 | `4311` | `4,311` |
| aitqa | gold_cell | q-267 | `3904` | `3,904` |
| aitqa | gold_cell | q-396 | `4113` | `4,113` |

## 1. pool x 조건별 EM

| pool | dataset | P1_fixed_512 | P4_path_cell | gold_cell |
|---|---|---|---|---|
| hitab_lookup | hitab | 0.4444 (n=189) | 0.6720 (n=189) | 0.9206 (n=189) |
| hitab_arith | hitab | 0.0333 (n=60) | 0.0167 (n=60) | 0.2903 (n=31) |
| aitqa | aitqa | 0.4333 (n=60) | 0.4167 (n=60) | 0.8000 (n=30) |
| rhb_fact | realhitbench | 0.3167 (n=60) | 0.3500 (n=60) | 0.7000 (n=30) |
| rhb_num | realhitbench | 0.1667 (n=54) | 0.1296 (n=54) | 0.4000 (n=30) |

`hitab_lookup`만 n=189로 확장했다(PREREGISTER 개정 5). 나머지 pool은 필요 n이 pool 크기를 넘어 확장하지 않았다. `gold_cell`은 개정 4로 pool당 30건으로 축소했고, `hitab_arith`의 31은 재배치 이전 실행분 1건이 포함된 결과다.

## 2. McNemar 정확검정 (P4_path_cell vs P1_fixed_512, paired) + Holm-Bonferroni

| pool | n | P1 EM | P4 EM | b (P4만) | c (P1만) | delta | 95% CI | 원 p | Holm 순위 | 임계값 | Holm 판정 | 필요 n | 검정력 |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| hitab_lookup | 189 | 0.4444 | 0.6720 | 64 | 21 | +0.2275 | [+0.1376, +0.3175] | 3.28e-06 | 1 | 0.010000 | 기각 (유의) | 66 | 충족 |
| hitab_arith | 60 | 0.0333 | 0.0167 | 1 | 2 | -0.0167 | [-0.0833, +0.0333] | 1.0000 | 4 | 0.025000 | 기각 실패 | 1411 | **부족** (b+c<10, 추정 불안정) |
| aitqa | 60 | 0.4333 | 0.4167 | 9 | 10 | -0.0167 | [-0.1500, +0.1333] | 1.0000 | 5 | 0.050000 | 기각 실패 | 8946 | **부족** |
| rhb_fact | 60 | 0.3167 | 0.3500 | 11 | 9 | +0.0333 | [-0.1167, +0.1833] | 0.8238 | 3 | 0.016667 | 기각 실패 | 2353 | **부족** |
| rhb_num | 54 | 0.1667 | 0.1296 | 3 | 5 | -0.0370 | [-0.1296, +0.0741] | 0.7266 | 2 | 0.012500 | 기각 실패 | 846 | **부족** (b+c<10, 추정 불안정) |

McNemar는 정확검정 `binomtest(b, b+c, 0.5)`, 연속성 보정 없음. delta의 CI는 paired bootstrap B=10000, seed=42. Holm-Bonferroni는 m=5, alpha=0.05, 순위 k의 임계값 alpha/(m-k+1); 순차 절차이므로 한 번 기각에 실패하면 이후 순위는 모두 기각 실패다. 필요 n은 Connor(1987) 정규근사, 관측 pi_d·pi_diff 고정, alpha=0.05 양측, power=0.80.

## 3. hitab_lookup 부분집합 안정성

| 부분집합 | n | P1 EM | P4 EM | delta | b | c | Recall P1 | Recall P4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 기존 60 | 60 | 0.4667 | 0.6000 | +0.1333 | 17 | 9 | 0.8667 | 0.9167 |
| 추가 129 | 129 | 0.4341 | 0.7054 | +0.2713 | 47 | 12 | 0.7674 | 0.9380 |

두 부분집합 delta 차이 (추가 129 − 기존 60): **+0.1380**, SE=0.1002, 95% CI [-0.0584, +0.3344] (독립 두 표본, 쿼리별 차이값 d=P4−P1의 정규근사). Recall은 gold 셀 수를 분모로 한 greedy fill 컨텍스트 내 포함률이다.

## 4. gold_cell 조건 = 리더 상한

| pool | EM | n |
|---|---:|---:|
| hitab_lookup | 0.9206 | 189 |
| hitab_arith | 0.2903 | 31 |
| aitqa | 0.8000 | 30 |
| rhb_fact | 0.7000 | 30 |
| rhb_num | 0.4000 | 30 |
| **전체** | 0.7742 | 310 |

## 5. 검정력 부족 pool

| pool | 현재 n | 필요 n | pool 전체 크기 | 불일치쌍 b+c |
|---|---:|---:|---:|---:|
| hitab_arith | 60 | 1411 | 175 | 3 |
| aitqa | 60 | 8946 | 451 | 19 |
| rhb_fact | 60 | 2353 | 164 | 20 |
| rhb_num | 54 | 846 | 54 | 8 |

위 pool은 현재 n이 필요 n에 못 미친다. 필요 n이 pool 전체 크기를 넘는 경우 현재 설계로는 충족할 수 없다. b+c<10인 pool은 pi_d·pi_diff 추정 자체가 불안정하며 필요 n 값도 그만큼 신뢰할 수 없다.

## 6. 채점 규칙 전문

Exact Match. 정규화는 통화/퍼센트 기호 제거 → 천단위 콤마 제거 → 공백 축약·소문자화 → 순수 소수의 후행 0 제거. 그 위에 PREREGISTER 개정 5의 R1(수치 정답에 한해 상대오차 <0.01 허용)만 적용한다. R2/R3/R4 미채택, 부호 정규화 없음. 다중 gold(원소 2개 이상)는 전부 일치를 요구한다.

```python
_CUR = re.compile(r"[$€£¥%]")
_THOU = re.compile(r"(?<=\d),(?=\d)")


def norm_em(s):
    s = _CUR.sub("", str(s))
    s = _THOU.sub("", s)
    s = " ".join(s.split()).strip().lower()
    if re.fullmatch(r"-?\d+\.\d+", s):
        s = s.rstrip("0").rstrip(".")
    return s


def gold_parts(g):
    """-> list of gold strings. A length-1 list is unwrapped to its element; a
    length>=2 list stays a list and is judged as a set (see em()).

    Only a bracketed literal is parsed as a container: literal_eval reads a
    bare '1,179' as the tuple (1, 179), which sent thousands-separated golds
    down the multi-gold path and past norm_em entirely (see BUGFIX_LOG.md)."""
    if str(g).strip()[:1] in "[(":
        try:
            v = ast.literal_eval(g)
            if isinstance(v, (list, tuple)):
                return [str(x) for x in v]
        except (ValueError, SyntaxError):
            pass
    return [g]


_NUM = re.compile(r"-?\d+(?:\.\d+)?")
REL_TOL = 0.01                     # PREREGISTER rev5, rule R1


def _one_num(s):
    m = _NUM.findall(str(s))
    return float(m[0]) if len(m) == 1 else None


def _same(p, g, rel):
    """One predicted value against one gold value."""
    if norm_em(p) == norm_em(g):
        return True
    if not rel:
        return False
    pv, gv = _one_num(p), _one_num(g)
    # R1: numeric answers only, relative tolerance, never an absolute-value or
    # percent-rescaling escape (rev5 rejects R2/R4).
    return (pv is not None and gv is not None and gv != 0
            and abs(pv - gv) / abs(gv) < REL_TOL)


def em(pred, gold, rel=True):
    """Single gold: match after normalisation, plus R1 relative tolerance when
    rel=True. Multi gold (list length >= 2): ALL elements must match, none
    extra -- the prediction is split on commas and paired greedily,
    order-insensitive. No partial credit. No sign normalisation."""
    gs = gold_parts(gold)
    if len(gs) == 1:
        return int(_same(pred, gs[0], rel))
    ps = [x for x in str(pred).split(",")]
    if len(ps) != len(gs):
        return 0
    left = list(ps)
    for g in gs:
        hit = next((x for x in left if _same(x, g, rel)), None)
        if hit is None:
            return 0
        left.remove(hit)
    return int(not left)
```

`gold_parts()`의 괄호 검사는 `BUGFIX_LOG.md`의 수정 사항이다.

## 7. 참조

- `BUGFIX_LOG.md` — `gold_parts()` 콤마 오파싱 수정
- `PREREGISTER.md` — 개정 1~5
- `results/phase4/reader_records.jsonl` (1156행), `reader_records.xlsx`
- 수정 전 수치: `results/phase4/final_summary.md`, `phase4b.md`, `phase4c.md`, `phase4d_taskA.md`, `phase4e.md`
