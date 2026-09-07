# 사전 등록 — 리더 프롬프트: 셀에 적힌 값을 변형하지 않는다 (G)

등록 2026-09-04. **실행 전 작성. 결과를 보고 이 문서를 고치지 않는다.**
상위 문서: `HANDOFF-2026-09-04.md` 「다음 세션 시작점」 G (3·4차 연속 이월).

## 왜 이 실험인가

`cond=gold`(정답 셀 문장을 **강제 주입**)에서 dev 830 EM 이 **.9024**, 오답 81 건이
남는다. 주입률이 1.000 이므로 **검색 탓이 아니다.** 그 81 건을 분류한 원장이
`results/audit/gold_reader_failures.xlsx`(`analysis/gold_reader_failures_xlsx.py`,
자동·규칙 기반)이고, 분류는 이렇다:

| 분류 | n |
|---|---:|
| G 문맥에 없는 값 (지어냄/계산) | 38 |
| **B 스케일 차이** | **13** |
| **A 부호 반대** | **12** |
| **C 여집합** | **12** |
| 그 밖 (D/E/F/H/I) | 6 |

**A·B·C 37 건에는 공통 성질이 하나 있다 — 정답이 주입된 셀에 이미 적혀 있다.**
37 건 중 **24 건은 gold 가 셀 값과 글자 그대로 같고**, **12 건은 부호만 다르다**
(gold 가 크기, 셀이 음수). 나머지 1 건은 gold 라벨 쪽이 깨졌다(`1.091 (0.340`).
즉 **36/37 은 리더가 베끼기만 하면 되는데 변형해서 틀렸다.**

⚠️ **이것은 "규칙이 없어서"가 아니다.** `answerer._RATIO_RULE` 은 이미
"NEVER multiply or divide by 100" 과 "If the answer is simply printed in a cell,
report it exactly as printed" 을 담고 있다. **긴 문단 맨 끝에 묻혀 있고, 계산 규칙
뒤에 온다.** 개입은 새 지식을 주는 것이 아니라 **이미 있는 규칙을 맨 앞으로 올리고
실패한 세 갈래를 이름으로 지목하는 것**이다.

### 세 갈래의 실제 모양 (원장에서 그대로)

```
A  Q "...how many percentage points did gdp ... receded in 2013?"
   셀 -1.2      gold 1.2       pred -1.2      ← 리더가 셀을 정확히 베꼈다
B  Q "how many thousand dollars were for r&d?"
   셀 39781.8   gold 39781.8   pred 39.7818   ← 1000 으로 나눔
B  Q "what was the percentage of workers in class 3?"
   셀 0.165     gold 0.165     pred 16.5%     ← 100 을 곱함 (_RATIO_RULE 이 이미 금지)
C  Q "what was the percentage of social workers in toronto?"
   셀 14.9      gold 14.9      pred 85.1      ← 100 에서 뺌
```

**A 는 리더 잘못이 아니다.** 12/12 전부 질문이 방향어(receded / decrease / decline /
fall / reduce / revised down)를 이미 담고 있고, HiTab 의 gold 는 그때 **크기만** 쓴다.
리더는 프롬프트가 시킨 대로("exactly as printed") 부호까지 베꼈고 0 점을 맞았다.
**규약 불일치이지 오독이 아니다** — 그렇게 쓸 것.

## 기준선 (dev, n=830, 전부 m=1)

인코더 `models/bge-base-cell-ft-p0`, α=0.8, `--title-mode page` (2026-09-02 확정 arm).
리더 `Qwen/Qwen2.5-7B-Instruct` 4-bit NF4, temp=0, seed=42, max_new_tokens=32.
채점 `analysis/phase4_summary.py: em()`.

| 조건 | EM | gold 주입률 | EM\|주입됨 | 출처 |
|---|---:|---:|---:|---|
| `gold` | **.9024** | 1.000 | .9024 | `gold_reader_failures.xlsx` |
| `top10` | **.7205** | .9446 | .7564 | `RESULTS.md` §12 |
| `top1` | .6807 | .7193 | .9196 | §12 |

## 개입

**검색을 한 글자도 바꾸지 않는다.** 프롬프트의 시스템 메시지만 바꾼다.
`rag_agent/generate/answerer.py` 의 `_DIRECT_SYS` 맨 앞에 아래 `_READ_RULE` 을 넣고,
`_RATIO_RULE`·`_ANSWER_FORM_RULE` 은 **문구도 순서도 그대로 둔다**(계산 질의가 그것에
매달려 있다). `_CODEGEN_SYS` 는 건드리지 않는다.

```python
_READ_RULE = (
    "Copy rule, applied first. If the answer is printed in one of the rows, report "
    "that number EXACTLY as printed and stop -- do not convert it. Do NOT divide by "
    "1,000 when the question says \"thousand\". Do NOT move the decimal point between "
    "0.165 and 16.5 when the question says \"percent\". Do NOT subtract the value from "
    "100. If the question already names the direction of the change (decrease, "
    "decline, fall, drop, reduce, recede, revised down), give the size WITHOUT a "
    "minus sign."
)
```

기존 프롬프트를 `v1`, 위를 얹은 것을 `v2` 라 부른다. **`v1` 을 지우지 않는다** —
이 저장소의 과거 답변 수치가 전부 `v1` 으로 나왔고, 지우면 짝지을 수 없게 된다.
계측기에 `--prompt {v1,v2}` 를 더하고 기본값은 `v1` 이다.

### 조건 (dev 830, 리더 호출 1,660 회)

| 조건 | 주입 | 프롬프트 | 무엇을 재나 |
|---|---|---|---|
| `gold_v2` | 정답 셀 강제 | v2 | **기전.** 37 건이 실제로 사는가 |
| `top10_v2` | 검색 상위 10 | v2 | **배치 수치.** 현행 확정 arm 위에서의 순이득 |

`v1` 쪽은 다시 돌리지 않는다 — `gold` .9024 와 `top10` .7205 가 같은 시드·같은
디코딩으로 이미 커밋돼 있고, R① 에서 재실행 불일치 **0/830** 으로 결정성이 확인됐다.

## 예측 (숫자, 실행 전 고정)

| 항목 | 기준 | **예측 (구간)** |
|---|---:|---|
| A 부호 12 건 중 회복 | 0 | **10 (8–12)** |
| B 스케일 12 건 중 회복 (gold 파손 1 건 제외) | 0 | **8 (5–12)** |
| C 여집합 11 건 중 회복 (역방향 1 건 제외) | 0 | **6 (3–11)** |
| G 문맥에 없는 값 38 건 중 회복 | 0 | **2 (0–6)** |
| **`gold_v2` EM** | .9024 | **.930 (.915–.950)** |
| **`top10_v2` EM** | .7205 | **.748 (.730–.770)** |
| 현재 정답이 깨지는 수 (회귀) | — | **≤ 15** |

**미리 적어두는 산술**: `gold` 에서 24 건이 살면 .9024 → .9313 이고, 그 이득은
`top10` 에 **주입률 .9446 을 곱해** 들어오므로 .7205 + .0273 = **.748** 이다.
**점추정은 .75 에 못 미친다.** 사용자가 목표로 말한 .75 는 이 개입의 **구간 상단**이지
점추정이 아니다 — 실행 후에 ".75 를 달성했다"로 쓰려면 실측이 .750 이상이어야 한다.

**A 를 가장 높게 예측하는 근거**: 12/12 가 예외 없이 같은 모양이고, 질문에 방향어가
있는지는 문자열로 판정 가능하다. B 를 그다음으로 두는 근거: `_RATIO_RULE` 이 이미
÷100 을 금지했는데도 8 건이 났다 — **문구를 앞으로 옮기는 것만으로는 안 될 수 있다.**
C 를 가장 낮게 두는 근거: 여집합은 질문 이해의 문제라 금지 문구가 안 닿을 수 있다.

## 부작용 상한 — 실행 전에 측정했다

부호 규칙이 지금 맞고 있는 답을 깨는지 dev 830 전수로 확인했다
(`results/audit/gold_reader_failures.xlsx` 시트 `전체_830`):

- **gold 가 음수인 질의는 830 건 중 4 건뿐이다.** 이 데이터셋은 사실상 부호를 안 쓴다.
- 그 4 건 중 질문에 방향어가 있는 것은 **1 건**이고, 그것은 **이미 오답**이다.
- 나머지 3 건(현재 정답)은 방향어가 없어 규칙이 **발동하지 않는다.**

```
python3 - <<'PY'
import pandas as pd, re
x = pd.read_excel('results/audit/gold_reader_failures.xlsx', sheet_name='전체_830')
neg = x['정답(gold)'].astype(str).str.strip().str.strip('[]()').str.startswith('-')
dirw = x['질문'].astype(str).str.contains(r'decreas|declin|fall|fell|reduc|reced|drop|down|less|lower|loss', case=False)
print(neg.sum(), (neg & dirw).sum(), x[neg & dirw]['정답여부'].tolist())   # 4 1 [0]
PY
```

→ **부호 규칙으로 깨질 수 있는 현재 정답은 0 건이다.** 배율·여집합 규칙은 이런 사전
상한이 없고, 그래서 위 「회귀 ≤ 15」가 그 자리를 대신한다.

## 반드시 성립해야 하는 것 (아니면 버그)

- `gold_v2` 의 gold 주입률 **1.000**, `top10_v2` 의 주입률 **.9446 정확히 일치**.
  검색을 안 건드렸으므로 달라지면 버그다.
- `top10_v2` 의 주입 셀 10 줄, 프롬프트 토큰이 `top10`(631) 대비 `_READ_RULE` 길이만큼만
  증가. 문맥이 잘리면(`context_truncated`) 그 질의는 따로 표시한다.
- 빈 예측(생성 캡 도달) 수가 `v1` 대비 늘지 않는다. `_READ_RULE` 이 길어져 32 토큰
  캡을 압박하면 이 실험은 프롬프트가 아니라 캡을 잰 것이 된다.

## 주검정

**`top10_v2` vs `top10`, n=830, `em()`, 페어드 exact McNemar.**
유의하지 않으면 **프롬프트를 바꾸지 않는다.**

부검정 (판정 아님, 기록용): `gold_v2` vs `gold` · A/B/C/G 분류별 회복 수 ·
회귀 건수와 그 분류.

## 기각 조건과 그 다음

- **주검정 유의 + 회귀 ≤ 15** → `v2` 를 기본 프롬프트로 채택하고 **test 1 회**.
  그 test 가 이 저장소의 **첫 답변 EM test 측정**이 된다(`hitab_test_lookup_all` 답변
  레그는 4 차까지 미실행).
- **주검정 유의 + 회귀 > 15** → 순이득이 아니다. **채택하지 않고** 어느 규칙이 깨뜨렸는지
  분류해서 규칙을 갈라 다시 등록한다.
- **주검정 유의하지 않은데 `gold_v2` 는 올랐다** → 이득이 distractor 에 먹힌 것이다.
  그때는 `top3`(사전등록 T)와 묶어야 하고, 이 문서로는 채택하지 않는다.
- **`gold_v2` 도 안 오르고 A 가 8 건 미만 회복** → 규칙 문구가 7B 에게 안 닿는다.
  프롬프트 축을 닫고 **채점 규약 쪽**(gold 가 크기만 쓰는 관례를 채점기에 반영할지)을
  별도로 등록해 다툰다. ⚠️ **그 경우에도 `em()` 을 고치지 말 것** — 과거 수치와 짝을
  못 짓는다. `em_lenient` 처럼 옆에 두고 진단으로만 쓴다.

## 이 실험이 판정하지 않는 것

- **G 38 건 (문맥에 없는 값).** 지어냈거나 계산해버린 것이라 베끼기 규칙이 안 닿는다.
  81 건 중 가장 큰 덩어리이고 **여기 남는다.**
- **`top3`.** 별도 사전등록(핸드오프 T)의 대상이다. 여기서 같이 돌리면 "프롬프트가
  벌었다"와 "10→3 으로 깎아서 벌었다"가 섞인다.
- **산술 pool.** `hitab_arith` 는 EM .0167 이고 베끼기 규칙이 오히려 해로울 수 있다.
  `_CODEGEN_SYS` 를 안 건드리는 이유가 그것이다. 산술에서의 효과는 미측정으로 남긴다.
- **test split.** dev 에서 먼저 판정하고, 채택될 때만 test 1 회.
- **다른 데이터셋.** AIT-QA·RealHiTBench 는 gold 규약이 다르다(RHB 는 퍼센트 스케일을
  강제하는 `_RHB_RATIO_RULE` 을 이미 따로 쓴다). HiTab 밖으로 확장하지 않는다.
