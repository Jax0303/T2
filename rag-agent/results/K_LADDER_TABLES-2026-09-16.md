# K별 검색·답변 정확도 (2026-09-16)

모집단: HiTab test 단일 셀 991문항 (`mode=all, m=1, aggregation=none`).
검색 레코드: `results/evaluation_v2/s3c_v2_records.jsonl` (s3c 셀 단위, hybrid α=.7, corpus=split).
리더: `local:Qwen/Qwen3-8B?quantization=4bit`, prompt=neutral, max_tokens=64, 채점 `hitab_exact_match_text`.
결과 파일: `results/k_ladder_qwen3_8b_20260916/s3c_v2_primary_qwen3_8b_{k1,k5,k10,k20,gold}.json`.

## 표 1. 검색 정확도 (top-k 고정)

| K | ESM | Recall | Precision | F1 |
|---|---|---|---|---|
| 1 | .5752 | .5752 | .5752 | .5752 |
| 5 | 0 | .7982 | .1596 | .2661 |
| 10 | 0 | .8698 | .0870 | .1582 |
| 20 | 0 | .9142 | .0457 | .0871 |

단일 gold 셀 + 고정 k 구조상 ESM>0 은 k=1 에서만 가능 (§본문 설명 참조). Precision=Recall/k, F1=2·Recall/(k+1).

## 표 2. 답변 정확도 (Qwen3-8B 4bit)

| 조건 | 답변 정확도 | 검색 성공 시 | 검색 실패 시 |
|---|---|---|---|
| k=1 | .5923 | .9877 (query count=570) | .0570 (query count=421) |
| k=5 | .6398 | .7901 (query count=791) | .0450 (query count=200) |
| k=10 | .6478 | .7390 (query count=862) | .0388 (query count=129) |
| k=20 | .6509 | .7086 (query count=906) | .0353 (query count=85) |
| gold (정답 셀만) | **.9839** | — | — |

gold 조건의 "검색 성공/실패" 열은 정의상 없음 — gold 문맥은 항상 정답 셀만 담으므로 분리할 미검색 사례가 없다.

## 해석

- k 를 늘리면 검색 Recall 은 오르지만(.58→.91) 문맥에 정답 외 셀(distractor)이 섞여 리더의
  "검색 성공 시" 정확도는 떨어진다(.99→.71). 두 효과가 상쇄돼 전체 답변 정확도는 완만하게만
  오른다(.59→.65).
- gold(정답 셀만, distractor 0) 조건은 .9839 — 리더 자체의 상한은 높다. k=20 답변 정확도 .6509 와의
  간극(.33)은 검색 실패(8.6%)가 아니라 대부분 distractor 오염이 설명한다.
- k=1 은 예외적으로 검색 성공 시 답변 정확도가 .99 로 거의 리더 상한과 같다 — 문맥이 1줄이라
  다른 셀과 헷갈릴 여지가 없기 때문.

## 참고: Qwen3.5-9B 시도 (폐기)

애초 Qwen3.5-9B(Apache-2.0, 2026년 8GB 최고 성능 후보)로 시작했으나 이 GPU(RTX 3060 Ti 8GB)에
맞지 않았다: 어휘 248k 의 임베딩+lm_head 만 4bit 로 줄지 않는 bf16 약 4GB 를 차지해 k=1 에서 이미
VRAM 여유가 없었고, k=5 부터 극심한 저하(28초/문항) 또는 CUDA OOM 이 재현됐다. Qwen3-8B(어휘
152k)로 교체해 k=20 까지 안정적으로 돌았다. k1 리그 도중 WSL GPU 패스스루 자체의 드라이버 오류
(`dxgk ioctl failed`, Windows 호스트 GPU 메모리 경합, 이 저장소 코드와 무관)로 한 번 죽어 750/991 을
날린 뒤, `scripts/answer_accuracy.py` 에 문항별 즉시 저장 + `--resume` 을 추가하고 처음부터 다시
돌려 위 표를 얻었다.
