# 사전 등록 — 어순(leaf-first)의 몫 (S2r) (2026-08-25 실행 예정)

**결과를 보기 전에 커밋한다.** `PREREG-2026-08-24-hitab-test.md`의 후속이고,
CLAUDE.md §"문장 형태는 무의미"가 흔들린다 — 재확인 필요가 지목한 실험이다.

## 왜 하는가 — 마지막 남은 교란 하나

우리 색인 단위와 MT2Net 문장은 **두 가지**가 다르다(대소문자는 죽었다 — 인코더가
`do_lower_case=True`라 애초에 못 본다):

1. **어순** — MT2Net은 잎 라벨을 앞에 둔다(leaf-first). 우리 경로는 root-first다.
2. **문장 골격** — `"Among X, the value of Y is Z."`의 틀 단어. 제목이 없으면 정보 0.

**2는 이미 걷어냈다** — S3c(`PREREG-2026-08-23-compact-untitled.md`)가 제목 없을 때
S2 경로 문자열로 떨어진다. 그런데 남은 증거가 서로 어긋난다:

| 코퍼스 | 대조 | 결과 |
|---|---|---|
| HiTab 제목 X, n=100, 512 | MT2Net .540 대 S2 .430 | **15:4, p=.0192 — 저쪽이 이김** |
| AIT-QA(제목 0%), n=451, 512 | MT2Net .239 대 S3c(=S2) .244 | 35:33, p=.904 — 동률 |

**어느 쪽이 맞는지 미해결이다.** 그런데 두 대조 모두 어순과 골격이 **아직 섞여 있다** —
MT2Net은 `"For a, b is v"` 골격을 가지고 S2는 `"a > b: v"`라 구분자까지 다르다.

## 무엇을 바꿨는가 (실행 전에 고정, 커밋됨)

`--cell-scheme S2r`: S2의 **각 축을 leaf-first로 뒤집기만** 한다.
`scripts/corpus_dump_vs_cell.py`, 3줄. `cell_text`를 그대로 재사용한다.

```
S2   1992/1993 > current dollars > per capita cost > dollars: 202
S2r  1992/1993 > dollars > per capita cost > current dollars: 202
MT   For 1992/1993, dollars, per capita cost, current dollars is 202
```

### 이 대조만이 용량 교란이 없다 — 전수 검증 (실행 전, LLM 없이)

| 코퍼스 | 셀 | S2 tok/cell | **S2r tok/cell** | **토큰 multiset 동일** | 문자열까지 동일 |
|---|---|---|---|---|---|
| HiTab dev | 58,759 | 15.91 | **15.91** | **58,759 / 58,759 (100%)** | 914 (1.6%) |
| AIT-QA | 5,320 | 19.36 | **19.36** | **5,320 / 5,320 (100%)** | 1,374 (25.8%) |

`BAAI/bge-small-en-v1.5` 토크나이저, 셀 전수. **토큰이 한 개도 늘거나 줄지 않는다.**
→ **여기서 나오는 차이는 용량으로 설명될 수 없다.** 이 저장소의 다른 모든 템플릿
대조는 길이가 같이 움직여서 "예산에 몇 셀이 들어가느냐"로 반박이 가능했는데,
이 대조는 그 반박이 원리상 닫혀 있다. `tests/test_templates.py`가 성질을 고정한다.

⚠️ **문자열까지 같은 셀은 뒤집을 것이 없다** — 경로가 1단이면 S2r ≡ S2다.
HiTab은 1.6%뿐이라 무해하나 **AIT-QA는 25.8%**라 효과가 그만큼 희석된다.
AIT-QA에서 0이 나와도 "어순이 무의미"의 강한 증거가 아니다. **HiTab이 주 무대다.**

## 예측

### W1 (주) — HiTab에서 어순은 0이다

`--cell-scheme S2r` 대 `S2`, HiTab dev n=830, 512.

**예측: |S2r − S2| ≤ .03, McNemar 유의하지 않음(p > .05).**

근거는 RESEARCH_STRUCTURE §4.1 주장 4다 — 순열 6회로 **순서 효과가 검출되지 않았다.**
leaf-first는 그 순열들 중 특정한 하나이므로, 주장 4가 참이면 여기서도 0이어야 한다.
방향은 **0 또는 아주 약한 양(leaf-first가 조금 나음)**으로 본다: 잎 라벨이 질문 어휘와
가장 겹치고 인코더가 앞쪽 토큰에 민감할 수 있다.

### W2 — 그러면 p=.0192는 골격·구분자의 몫이다

W1이 참이면 MT2Net이 S2를 이긴 +.110은 어순이 아니라 남은 차이에서 온다.
**S2r 대 MT2Net을 같은 런에서 재고, 그 차이가 +.08 이상 남을 것으로 예측한다.**

### W3 (통제) — flat arm 0:0

세 런(S2 / S2r / mt2net)의 `flat`이 **불일치 0:0**이어야 한다.
아니면 셀 템플릿 말고 다른 것이 움직인 것이므로 W1·W2를 해석하기 전에 원인부터 찾는다.

### W4 — AIT-QA는 희석 때문에 판정력이 낮다

**|S2r − S2| ≤ .03 예측하되, 이 코퍼스의 0은 W1의 확인이 아니라 참고로만 쓴다.**
셀의 25.8%가 뒤집을 것이 없기 때문이다.

## 무엇이 예측을 깨는가

| 관측 | 함의 |
|---|---|
| **W1이 유의하게 양 (leaf-first가 이김)** | 주장 4("순서의 기여는 검출되지 않는다")가 **답변 EM에서 깨진다.** OSC에서 안 보이던 것이 EM에서 보이는 것이므로 큰 발견이고, 색인 단위를 leaf-first로 바꾸는 것을 검토한다. RESEARCH_STRUCTURE §4.1과 §5 표 3을 다시 쓴다 |
| **W1이 유의하게 음 (root-first가 이김)** | 우리 순서가 우연이 아니라 선택이었다고 쓸 수 있게 된다. 단 사후 정당화로 읽히지 않게 **이 사전 등록을 함께 인용**한다 |
| W2가 남지 않음 (S2r ≈ MT2Net) | 어순이 전부였다는 뜻이므로 W1과 모순이다. 두 짝을 다시 본다 |
| **W3이 깨짐** | 런 폐기. 리더·환경부터 맞춘다 |

## 고정한 설계 (실행 전)

| 항목 | 값 |
|---|---|
| 코퍼스 | HiTab **dev** (주), AIT-QA (참고) |
| 모집단 | `hitab_dev_lookup_all` n=830 / AIT-QA 기본 n=451 |
| 예산 | 512 |
| arms | `flat,cell` |
| 검색기 | dense (`BAAI/bge-small-en-v1.5`) |
| 리더 | `local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit&dtype=float16` — **세 런 전부 동일** |
| 채점 | `hitab_exact_match_text` |
| 검정 | exact McNemar, `scripts/paired_em_between_runs.py` |
| 보정 | **W1·W2 2건에 Holm** (W3는 통제, W4는 참고) |

**test split은 쓰지 않는다.** 이건 설계 질문이므로 dev에서 답한다.

## 실행

공통: `--retriever dense --budget 512 --arms flat,cell
--reader 'local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit&dtype=float16'`

```
--dataset hitab --population hitab_dev_lookup_all --cell-scheme S2   --out results/wo_hitab_s2_512.json
--dataset hitab --population hitab_dev_lookup_all --cell-scheme S2r  --out results/wo_hitab_s2r_512.json
--dataset hitab --population hitab_dev_lookup_all --cell-scheme mt2net --out results/wo_hitab_mt2net_512.json
--dataset aitqa --cell-scheme S2  --out results/wo_aitqa_s2_512.json
--dataset aitqa --cell-scheme S2r --out results/wo_aitqa_s2r_512.json
```

**MT2Net을 다시 도는 이유**: 기존 `h2h830_mt2net_512.json`은 리더가 bfloat16 기본값이라
오늘의 4bit 런과 짝지으면 리더 교란이 섞인다. 2026-08-24에 그 검사가 실제로 걸렸다.
