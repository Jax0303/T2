# 사전 등록 — 두 번째 로컬 리더 (Qwen3-8B) (2026-08-25)

**결과를 보기 전에 커밋한다.** 오늘 세 번째 사전 등록이다.

## 왜 하는가 — 리뷰어의 다음 질문

`PREREG-2026-08-24-hitab-test.md`가 "dev에서만 되는 거 아니냐"를 닫았다.
바로 다음에 오는 것은 **"약한 리더 하나에서만 되는 거 아니냐"**다.
지금까지의 모든 답변 EM은 `Qwen2.5-7B-Instruct` 4-bit 하나에서 나왔다.

2026년 표 QA 논문들이 실제로 쓰는 로컬 리더를 조사했다(본문 확인):

| 논문 | 로컬 리더 |
|---|---|
| HD-RAG (arXiv 2504.09554) | Qwen2.5-7B-Instruct, Qwen2.5-32B, Llama-3.1-8B, Mistral-Nemo |
| RAG over Tables (arXiv 2504.01346) | Qwen2.5-7B-Instruct, Llama-3.1-8B/70B, Llama-3.2-3B, Phi-3.5-mini |
| ASTRA (arXiv 2604.08999) | **Qwen3-8B** (로컬 답변 선택기) |

→ 우리 Qwen2.5-7B는 **표준**이고, 2026년 논문은 **Qwen3-8B**로 옮겨가고 있다.
같은 8GB 카드에 4-bit로 올라가므로 **돈이 들지 않는다.**

⚠️ **양자화는 우리가 남들과 다른 유일한 지점이다.** 위 세 논문 중 리더를 4-bit로
양자화한 곳은 없다(A100 bf16). 8GB 제약이므로 바꿀 수 없고, **두 arm에 동일하게
적용되므로 대비를 편향시키지 않는다** — 논문에 그렇게 명시한다. 이 런은 그 문장을
한 모델 더 넓히는 것이지, 양자화 자체를 시험하는 것이 아니다.

## 무엇을 바꿨는가 (실행 전에 고정, 커밋됨)

`rag_agent/llm/local_qwen.py`: `apply_chat_template(..., enable_thinking=False)`.
Qwen3 템플릿은 기본이 사고 모드라 `<think>...</think>`가 답변 문자열에 섞이고
`max_new_tokens`를 먹는다. **Qwen2.5-7B-Instruct에서는 프롬프트가 바이트 동일**임을
확인했다(그 템플릿이 이 플래그를 무시한다) — **기존 결과는 전부 그대로 재현된다.**

## 예측

HiTab dev `hitab_dev_lookup_all` n=830, 512, dense, `--cell-scheme {S3c, mt2net}`,
리더 `local:Qwen/Qwen3-8B?quantization=4bit&dtype=float16`.

Qwen2.5-7B 4-bit 기준선 (`results/s3c_hitab_512.json`, `wo_hitab_mt2net_512.json`):
**S3c .604 대 MT2Net .516, +.088.**

### V1 (주) — 부호와 크기가 유지된다

**S3c − MT2Net 차이가 +.04 ~ +.14, 부호 양, McNemar p < .01.**

근거: 우리 이득의 메커니즘은 **제목이 문맥의 표 수를 줄여 겨냥을 좁히는 것**
(2.4~2.9표 대 6.1표)이고, 이건 리더의 성질이 아니라 **문맥의 성질**이다.
리더를 바꿔도 문맥은 같으므로 이득이 남아야 한다.

### V2 — 절대값은 오른다

Qwen3-8B가 더 새롭고 크므로 **두 arm 다 Qwen2.5-7B보다 높을 것.**
**S3c가 .60 이상**으로 예측한다. 단 크기는 예측하지 않는다.

### V3 (통제) — flat arm이 두 런에서 0:0

`flat`은 `--cell-scheme`을 읽지 않는다. 같은 리더·같은 예산이므로 **0:0**이어야 한다.
아니면 리더가 비결정적이거나 다른 것이 움직인 것이므로 V1을 해석하기 전에 원인을 찾는다.

### V4 — OSC는 두 런에서 완전히 같다

검색은 리더와 무관하다. **Qwen2.5 런의 OSC와 소수 셋째 자리까지 같아야 한다**
(S3c .789 / MT2Net .813 부근). 다르면 코퍼스나 색인이 바뀐 것이다.

## 무엇이 예측을 깨는가

| 관측 | 함의 |
|---|---|
| **V1의 부호가 음** | 제목의 이득이 리더 특수였다. **논문의 중심 주장이 리더 조건부가 된다** — 숨기지 않고 그렇게 쓴다 |
| V1이 양이지만 유의하지 않음 | n=830은 Qwen2.5에서 145:73을 낸 표본이다. 여기서 무승부면 **검정력 부족으로 변명할 수 없다** |
| **V3이 깨짐** | 사고 모드가 안 꺼졌거나 생성이 비결정적이다. 런 폐기 후 원인부터 |
| **V4가 깨짐** | 검색 쪽이 움직였다는 뜻이므로 리더 실험이 아니다. 런 폐기 |
| V2가 깨짐(절대값이 내려감) | 예측 실패로 적는다. 8B가 7B보다 이 과제에서 나쁠 수 있고, 그것도 결과다 |

## 실행

```
--dataset hitab --split dev --population hitab_dev_lookup_all --retriever dense
--budget 512 --arms flat,cell
--reader 'local:Qwen/Qwen3-8B?quantization=4bit&dtype=float16'
--cell-scheme S3c    --out results/q3_hitab_s3c_512.json
--cell-scheme mt2net --out results/q3_hitab_mt2net_512.json
```
