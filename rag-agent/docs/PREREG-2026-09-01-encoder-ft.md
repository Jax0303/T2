# 사전등록 — 인코더 파인튜닝 (cross-table hard negative)

작성 2026-09-01, **통제군 보드 실행이 끝나기 전에 예측을 박는다** (규약 8).
개입 실행 전. 이 파일은 실행 후 수정하지 않는다.

## 왜 이 개입인가

지금 검색 점수의 70%가 `BAAI/bge-small-en-v1.5` 기성품에서 나온다 (hybrid α=0.7).
이 저장소의 모든 검색 수치가 **우리 데이터로 한 번도 학습시키지 않은 인코더** 위에
쌓여 있다. 기각된 개입들(soft/hard 표 prior, cascade, S2t/S2h 유일 라벨, 생성 제목,
capped, 리랭킹, BGE 쿼리 접두사, adaptive 꼬리 절단)은 **전부 인코더를 그대로 두고
입력 문장이나 후보 집합만 바꿨다**. 파인튜닝은 코사인 자체를 바꾸는 유일한 개입이다.

## 무엇을 바꾸는가

`scripts/finetune_cell_encoder.py`의 hard negative 구성. 기존은 **같은 표 형제 셀만**
4개(2026-08-25 진단: "형제와 1~2등을 못 가른다"). 여기에 **현재 색인에서 채굴한
고득점 타표 셀**을 넣는다: `--neg-per-query 4 --neg-cross 2` (형제 2 + 타표 2).

근거는 `results/rank/STAGE1_BOARD.md` §B — setEM@10 실패에서 gold를 앞지른 셀의
**94.3%가 다른 표 셀**이다. 두 진단은 모순이 아니라 구간이 다르다: 형제는 gold가
2~3위일 때, 타표 셀은 gold가 10위 밖으로 밀렸을 때의 경쟁자다. .8373 → .9는 후자다.

## 고정 사양

| 항목 | 값 |
|---|---|
| 베이스 | `BAAI/bge-small-en-v1.5` (33M) |
| 학습 모집단 | `populations/hitab_train_lookup_all.txt` (n=3,667) |
| triplet | 3,667 (2026-09-01 dry-run 실측, 탈락 0) |
| negative | 형제 2 + 타표 2, 채굴은 **base 인코더 dense top-50** |
| loss | `CachedMultipleNegativesRankingLoss` (mini_batch 16) |
| epochs / batch / lr | 2.0 / 32 / 2e-5, warmup 0.1, fp16 |
| seed | 42 |
| 쿼리 접두어 | **없음** — 평가기 `cell_rank_dump.py`가 안 붙인다 |
| 색인 단위 / 검색기 | S3c / hybrid α=0.7 (평가 시 변경 없음) |
| 출력 | `models/bge-cell-ft`, `results/rank_ft/` |

누출: HiTab train/dev/test는 표를 하나도 공유하지 않는다 (2,519 / 540 / 538, 교집합 0).
dev로 판정, test는 dev 판정이 끝난 뒤 한 번만.

## 주지표

**`hitab_dev_lookup_all` (n=830) 의 setEM@10.** 예산 없는 랭킹 지표
(`results/DEPRECATED.md`에 따라 OSC@512는 쓰지 않는다).

## 예측 (실행 전 커밋)

통제군 = base 인코더, 같은 코퍼스·검색기. 괄호는 `STAGE1_BOARD.md`의 기존 실측값.

| 모집단 | 지표 | 통제 | **예측 (점)** | 예측 구간 |
|---|---|---:|---:|---|
| `hitab_dev_lookup_all` n=830 | **setEM@10** | (.8373) | **.89** | .86 – .92 |
| `hitab_dev_lookup_all` | R@1 | (.5157) | **.62** | .56 – .68 |
| `hitab_dev_lookup_all` | setEM@50 | (.9120) | **.94** | .92 – .96 |
| `hitab_dev_lookup_all` | MRR | (.6299) | **.70** | .65 – .75 |
| `hitab_dev_lookup_all` | 표 recall@1 (셀 투표) | (.8458) | **.87** | .84 – .90 |
| `hitab_dev_lookup_multi` n=33 | setEM@10 | MEASURED: NO | **+.06** | −.03 – +.15 |

test(`hitab_test_lookup_all`, `hitab_test_lookup_multi`)는 dev 판정 후 1회, 예측은
dev에서 관측한 델타가 test에서 **절반 이상 재현된다**로 건다.

## 판정 규칙 (실행 전 고정)

1. **주검정**: 830건 페어드 setEM@10, exact McNemar 양측, α=.05.
2. **효과 없음**: 델타가 +.025 미만이면 노이즈 안 (n=830, p≈.84에서 SE≈.0127, 2×SE).
3. **목표 도달**: setEM@10 ≥ **.90**.
4. 주지표가 3을 못 넘어도 1을 통과하면 "유의하나 목표 미달"로 기록하고,
   다음 단계(형제/타표 비율 스윕, epoch·lr)는 **별도 사전등록**으로 간다.
5. **음수여도 그대로 보고한다.** S2t/S2h처럼 기각으로 남긴다.
6. 모든 지표를 표에 전부 넣는다. 주지표만 골라 보고하지 않는다.

## 미리 적어두는 교란 요인

- 채굴은 **dense 전용**인데 평가 색인은 hybrid α=0.7이다. 3,667 질의 × 291,403 셀
  BM25 비용 때문. 이 근사가 채굴 품질을 얼마나 깎는지는 **측정하지 않는다**.
- `--neg-cross 2`는 스윕하지 않은 임의 선택이다. 0(형제만)과 4(타표만)는 이번에 안 돈다.
- `hitab_dev_lookup_multi` n=33은 어떤 델타도 유의하게 만들지 못한다. 참고 수치다.
