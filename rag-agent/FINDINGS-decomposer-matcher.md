# 분해기(decomposer)로 답변 정확도를 올리는 길 — 근거 정리

작성 2026-08-20. 이 문서의 모든 수치는 **커밋된 결과 파일**에서 나왔고 출처를 함께 적는다.
출처 없는 수치는 쓰지 않는다(저장소 규칙).

관련 문서: [`README.md`](README.md) · [`NEXT.md`](NEXT.md) · [`STATUS-2026-08-06.md`](STATUS-2026-08-06.md)

---

## 왜 분해기인가 — 병목의 위치는 이미 측정돼 있다

답변 정확도를 올리는 레버는 **검색**이 아니다. 저장소가 세 번 독립적으로 측정한 사실:

> 커버리지가 올라가도 답변으로 전환되지 않는다. 커버리지 +5.5 → 답 +1.0,
> MultiHiertt는 **+13.7 → ±0.0**. (아티팩트 "What the failure bought")

병목은 **무엇을 찾을지 정하는 단계 = 질의 분해**다.
`osc_given_decomp = 1.00` — 분해가 맞으면 검색은 다 찾아온다
(`results/e6_scope_treatments.json`, `results/e2_osc_enum*.json`).
따라서 분해 정확도 1점은 곧 완전성 1점이고, 유능한 리더에선 답 1점이다.

---

## 사실 1 — 현재 분해기는 **안 하느니만 못하다** (HiTab)

`results/operand_rag/hitab/summary.json` → `operand_recall`, S2(header-path) 직렬화, k=5:

| 설정 | operand_recall@5 | 의미 |
|---|---|---|
| `s2_headerpath \| plain`   | **0.9053** | 분해 안 함 — 질문 전체로 검색 |
| `s2_headerpath \| operand` | **0.8300** | 현재 분해기(fuzzy)로 분해 후 검색 |
| `s2_headerpath \| oracle`  | **0.9742** | 분해가 완벽할 때의 천장 |

**현재 분해기는 plain보다 recall을 0.905 → 0.830으로 떨어뜨린다.** NEXT.md가 말한
"분해하지 않는 베이스라인에 진다"가 이 숫자다. 진짜 목표는 자기 fuzzy 기준선이 아니라
**plain(0.905)을 넘어 oracle(0.974)로 가는 것** — 위로 약 7pt의 여지가 있다.

---

## 사실 2 — 그 여지를 여는 레버는 **매처(matcher)**다

분해 천장 `header_path_match_accuracy`를 매처별로 측정한 값
(`results/operand_rag/<bench>/summary.json` → `ceiling`, seed 42):

| bench | split | n_op | fuzzy | embedding | hybrid | 최선 |
|---|---|---|---|---|---|---|
| hitab   | dev        | 241 | .3029 | **.4855** | .4090 | embedding |
| finqa   | validation | 238 | .4943 | **.5716** | .5437 | embedding |
| wikisql | validation | 300 | .7413 | .7112 | **.7874** | hybrid |

- **계층형(HiTab/FinQA)** 에서는 embedding이 fuzzy 대비 **+18.3pp / +7.7pp**로 압도.
- **평면형(WikiSQL)** 은 헤더가 짧은 정확 문자열이라 어휘 신호가 살아, hybrid가 최선이고
  embedding 단독은 오히려 fuzzy 아래.

즉 단일 승자는 없다. **매처는 코퍼스의 성질**이며, 현재 파이프라인 기본값
(`resolve_against_table`의 `_fuzzy_score`)은 계층형 두 곳에서 최악의 매처를 쓰고 있다.
이것이 사실 1의 분해기가 net-negative인 직접 원인이다.

README §3의 "예산 맞춤 대조를 통과한 유일한 개입 = 의미 헤더-경로 resolver"가 바로 이
embedding 매처다(train n=1,006, k=5/10/20 각각 +.049/.038/.040, p<.015,
`results/resolver_osc_matched_train.json`).

---

## 중요 — 두 단계 중 어디가 임베딩인가

파이프라인에는 임베딩이 들어갈 자리가 둘이다. 헷갈리지 말 것:

- **검색(retrieval) 단계** — 셀을 문장(S3)으로 바꿔 **dense 임베딩 + BM25로 검색**.
  `HybridIndex`. **이미 임베딩이다.** "문장 생성 → 임베딩 → 검색"이 여기다. 안 건드렸다.
- **분해(decomposition) 단계** — 검색 *전에* "이 질문이 표의 어떤 헤더경로(셀)를 필요로 하나"를
  정하는 단계. `resolve_against_table` / `_rank_paths`. **여기만 fuzzy(어휘 겹침)로 남아 있었다.**
  이 문서의 수정 대상이 바로 이 단계다.

즉 이번 변경은 새 방법이 아니라, **파이프라인에서 유일하게 임베딩이 아니던 분해 단계를
임베딩으로 바꿔 방법 전체를 일관되게** 만든 것이다.

## 이번에 반영한 것 (코드) — 기본값을 고침

1. **분해 단계 기본값을 임베딩으로 전환.**
   `OperandTargetedRetriever(embed_resolver=...)`의 기본값을 `None`(=정책)으로 바꿨고,
   `None`이면 `best_matcher(bench)=="embedding"`(bench 미지정 시 계층형 가정으로 **True**).
   → 이제 **방법의 기본 분해기 = 임베딩**. `embed_resolver=False`는 어휘 베이스라인 고정용.
   (`rag_agent/retrieve/operand_retrieval.py`)

2. **커밋된 수치가 조용히 바뀌지 않도록 모든 어휘 베이스라인을 명시 고정.**
   `answer_accuracy_resolver.py`(base), `resolver_osc_matched.py`(lex),
   `pipeline_osc_asdescribed.py`, 단위테스트 2개에 `embed_resolver=False`를 박았다.
   나머지 스크립트는 이미 명시적으로 `embed_resolver=True`였다. → **기존 결과 파일 전부 그대로.**
   `pipeline_osc_asdescribed.py`는 "AS DESCRIBED" 방법이지만 커밋된 결과
   (`results/pipeline_osc_asdescribed*.json`, `gate_k*.json`)를 지키려 고정해 뒀다.
   **임베딩 방법으로 재기준선을 잡으려면 그 `embed_resolver=False`를 지우고 재실행하면 된다.**

3. **매처 정책을 근거에 못박음.**
   `rag_agent/query/matcher_policy.py`(`best_matcher`+`BENCH_CEILING`, 출처 주석),
   `tests/test_decomposer_matcher_policy.py`가 정책 == `results/operand_rag/*/summary.json`
   argmax 임을, 그리고 **분해 기본값이 임베딩임**을 검증. 순수 stdlib.

## 검증(실행) — 답변 LLM은 로컬

`scripts/answer_accuracy_resolver.py`가 그대로 검증 하네스다. base(어휘) vs treat(임베딩)를
같은 질의·같은 k·같은 리더로 짝지어 잰다. **답변 리더 기본값을 로컬 Qwen2.5-7B로 바꿨다**
(Groq 일일 한도 없이 돌아감):

```
.venv/bin/python scripts/answer_accuracy_resolver.py --mode direct
```

이때 출력의 **`[osc]` 줄**(리더 호출 전, LLM 불필요)이 분해기 수정의 핵심 증거다 —
어휘 vs 임베딩 분해기의 **operand-set 완전성**을 바로 보여준다.

## 표 검색을 먼저 하는 실제 파이프라인 (cascade)

기존 답변 레그는 전부 **정답 표를 공짜로 깔고** 시작했다(within-doc, 표 검색 = 오라클).
실제 시스템처럼 **쿼리 → 표 검색 → 셀 검색 → 답**으로 닫는 레그를 추가했다:

- `rag_agent/retrieve/cascade.py` — `TableIndex`(표 하나 = 그 표의 S3 셀 문장 전체를 한
  문서로 임베딩) + `cascade_retrieve`(top-M 표를 고르고 그 안에서만 셀 검색, 셀은 `chunk_id`로
  병합). 셀 인덱스와 **같은 인코더**를 써 임베딩 공간을 일치시킨다.
- `scripts/answer_accuracy_cascade.py` — 코퍼스 = 스플릿의 모든 표(진짜 distractor 포함).
  `--top-m`으로 표를 몇 개 남길지, `--oracle-table`로 1단계를 건너뛰어 within-doc 천장을 같은
  하네스에서 비교. 리더 기본값은 로컬 Qwen.
- 보고: **`table_hit@M`**(1단계 정확도) · **`osc`**(정답 표 셀로만 채점) · `em`/`num`(답).

**정직한 예상:** 1단계 표 검색은 이 저장소의 알려진 병목이다 — MultiHiertt에서
`gold_table_in_top1 ≈ 27~45%`(`NEXT.md §3`, `results/operand_collision_multihiertt_n300.json`).
따라서 cascade의 최종 EM은 오라클-표 within-doc 숫자보다 **낮게** 나온다. 그 낙차가 곧
"표를 먼저 찾는 비용"이고, `--oracle-table`과의 차이로 정확히 측정된다.

## 정직한 한계 — 로컬 리더의 산술 천장

이 레그의 모집단은 **다중 피연산자(집계형)** 질문이고, 4-bit 7B 로컬 리더는 산술을 거의 못 한다
(아티팩트: arm 무관 .02–.06). 따라서 **답변 EM은 두 arm 모두 낮게** 나오고, 그것은 분해기가
아니라 리더를 재는 것이다(STATUS §1.6: 주입도 gpt-oss-120b에서만 전환, llama-3.1-8b에서 0).
그래서:

- **분해기 수정의 효과**는 `[osc]`(완전성, LLM-free)로 읽는다 — 로컬로 충분.
- **답변 EM 숫자**가 필요하면 계산되는 리더로 바꾼다(`--solver-model openai/gpt-oss-120b`).

"이 변경이 답변 정확도를 올린다"는 계산 가능한 리더에서 전환이 확인되기 전까지 인용 금지.
