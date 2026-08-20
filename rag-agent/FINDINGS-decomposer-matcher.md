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

## 이번에 반영한 것 (코드)

매처 선택을 **근거에 못박은 정책**으로 만들었다. 커밋된 천장의 argmax와 코드가 어긋나면
테스트가 깨진다 — 새 실행으로 숫자를 갱신하지 않는 한 정책을 바꿀 수 없다.

- `rag_agent/query/matcher_policy.py` — `best_matcher(bench)` + `BENCH_CEILING`
  (위 표를 그대로, 출처 주석과 함께). torch/임베더 의존 없음.
- `rag_agent/query/operand_decomposer.py` — `best_matcher` 재수출.
- `tests/test_decomposer_matcher_policy.py` — 정책 == `results/operand_rag/*/summary.json`
  argmax 임을 검증(순수 stdlib, venv 없이 실행됨).

**의도적으로 기본값은 뒤집지 않았다.** `header_path_resolver.resolve_against_table`는
"embedding을 기본으로 켜면 `results/`의 기존 수치가 조용히 비교 불가가 된다"고 명시했고,
파이프라인 기본을 바꾸는 것은 논문 주장 범위가 걸린 결정이라(NEXT.md §3) 지도교수 확인
대상이다. 정책은 **한 줄로 채택 가능하게** 배선만 해 뒀다.

## 채택하려면 (연구 결정)

계층형 벤치의 분해 경로에 임베딩 매처를 기본으로 건다:

```python
from rag_agent.query.matcher_policy import best_matcher
# decompose_operands(...) / resolve_intent(...) 호출부에 encoder=EmbedResolver(enc) 를
# best_matcher(bench)=="embedding"/"hybrid" 일 때 전달.
```

## 아직 열려 있는 것 (정직하게)

천장(ceiling) 이득이 **recall → 답변 정확도**로 전환되는지는 자동이 아니다. 저장소의 반복된
교훈이다(총합행 주입은 gpt-oss-120b에서만 전환, llama-3.1-8b에서 0 — STATUS §1.6).
전환 확인은 `scripts/answer_accuracy_resolver.py`를 **gpt-oss-120b(effort=low)** 로 돌려야
하고, 이는 현재 Groq 일일 한도로 막혀 있다(아티팩트 "The leg that is still open").
이 문서는 그 실행 없이 **매처 선택이 근거대로 되어 있음**까지만 못박는다.
전환 숫자가 나오기 전까지 "이 변경이 답변 정확도를 올린다"고 인용 금지.
