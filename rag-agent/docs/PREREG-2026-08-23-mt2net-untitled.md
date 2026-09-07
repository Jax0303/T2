# 사전 등록 — MT2Net 정면대결을 제목 없는 두 코퍼스로 확장 (2026-08-23)

**결과를 보기 전에 커밋한다.** `PREREG-2026-08-23-mt2net.md`의 후속이며 같은 규율이다.

## 왜 하는가 — 이긴 데이터셋만 싣지 않기 위해

HiTab n=830에서 우리 색인 단위가 MT2Net을 이겼다
(.600 대 .518, Holm p=1.04e-05 / .599 대 .479, Holm p=5.22e-10;
`results/h2h830_{s3,mt2net}_{512,1024}.json`).

**한 데이터셋에서 이긴 것을 그 데이터셋만 실어 보고하면, "유리한 무대를 골랐다"는
지적이 그대로 성립한다.** 그래서 이기지 못할 것으로 예상하는 무대에서도 돌린다.

## 예측의 근거 — MT2Net 템플릿에는 제목 자리가 없다

HiTab 대조에서 우리가 이긴 이유로 지목한 것은 **제목**이다. MT2Net 템플릿에는 제목을
담을 자리가 아예 없고, HiTab은 표의 99.3%가 제목을 가진다.

AIT-QA와 RealHiTBench는 사정이 다르다.

| 코퍼스 | 제목 보유 표 | 우리 S3가 MT2Net보다 더 담는 정보 |
|---|---|---|
| HiTab | **99.3%** | 제목 (거의 모든 표) |
| AIT-QA | **0%** | **없음 — 문장 골격과 어순뿐** |
| RealHiTBench | 38.2% | 제목 (표의 38.2%에서만) |

AIT-QA에서 우리 S3와 MT2Net은 **같은 정보를 다른 문장으로 쓴다.** 정보량이 같으면
차이가 없어야 한다. 이것이 이 실험의 핵심이다.

## 예측

### Q3 (주) — AIT-QA에서 무승부

cell arm에서 |S3 − MT2Net| ≤ .04, McNemar 유의하지 않음 (p > .05).

**이것이 맞으면 HiTab의 승리는 "우리 문장이 그냥 더 좋다"가 아니라 "제목을 담았다"가
원인이라는 주장이 데이터로 뒷받침된다.** 이길 수 있는 조건과 못 이기는 조건이
사전에 예측한 대로 갈리기 때문이다.

### Q4 — RealHiTBench에서는 작은 양(+), 아마 유의하지 않음

제목 보유율이 38.2%라 HiTab(99.3%)과 AIT-QA(0%) 사이다.
예측 크기 **0 ~ +.04**, n=243에서 유의해지지 않을 가능성이 높다고 미리 적는다.

### Q5 — 검색 지표는 판정에 쓰지 않는다

OSC가 어느 쪽에서 높든 **채택 근거로 쓰지 않는다.** HiTab n=830에서 이미
MT2Net이 OSC는 더 높으면서 EM은 졌다(OSC/EM 역전). 이 역전이 여기서도
관측되면 기록하되, 그것으로 방법을 고르지 않는다.

## 무엇이 예측을 깨는가

| 관측 | 함의 |
|---|---|
| **Q3이 우리 쪽으로 유의** | 제목만이 원인이라는 설명이 부족하다. 문장 골격·어순이 따로 몫을 한다는 뜻이므로, HiTab의 승리를 제목으로만 귀속한 서술을 고쳐야 한다 |
| **Q3이 MT2Net 쪽으로 유의** | 제목이 없는 무대에서는 발표된 단위가 우리보다 낫다. 우리 주장은 **"제목이 있는 표에서"** 조건부가 되고, 논문에 그렇게 쓴다 |
| Q4가 크게 양 | 38.2%의 제목이 99.3%와 비슷한 몫을 한다는 뜻이라 제목 보유율과 이득의 단조 관계가 깨진다 |

**네 런의 수치를 전부 싣는다.** Holm 보정은 HiTab 2건 + 여기 2건, **총 4건**에 건다.

## 고정한 설계 (실행 전)

| 항목 | 값 |
|---|---|
| 코퍼스 A | AIT-QA, 답 매칭 모집단 n=451 |
| 코퍼스 B | RealHiTBench, 답 매칭 모집단 n=243 |
| 검색기 | dense (`BAAI/bge-small-en-v1.5`) |
| 예산 | 512 (세 데이터셋 정면대결 표와 같은 값) |
| arms | `flat,cell` (HiTab 정면대결과 동일) |
| 색인 단위 | `--cell-scheme S3` 대 `--cell-scheme mt2net` |
| 리더 | `local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit&dtype=float16` |
| 채점 | `hitab_exact_match_text` (네 데이터셋 공통) |
| 검정 | cell arm exact McNemar, `scripts/paired_em_between_runs.py` |

### AIT-QA의 S3를 다시 돌리는 이유 — 리더 출처가 기록돼 있지 않다

기존 `results/corpus_dump_vs_cell_s3_aitqa_dense_512.json`은 `reader` 필드에
`local:Qwen/Qwen2.5-7B-Instruct`만 남아 있고 **양자화·dtype이 지워져 있다**
(`reader_spec`을 기록하기 전의 런이다). 그 런과 오늘의 mt2net 런을 짝지으면
**리더가 달랐을 가능성이 대조에 섞인다.** 그래서 AIT-QA는 S3와 mt2net을
**오늘 같은 스펙으로 나란히 새로 돌린다.**

RealHiTBench는 `results/rhb_dense_512_s3_casefix.json`이 오늘 코드·위 스펙으로
돌아간 것이 확인되므로(`run_casefix.sh`) 그것을 S3 쪽으로 쓴다.

## 실행 (이 문서 커밋 후)

```
for SCH in S3 mt2net; do
  .venv/bin/python scripts/corpus_dump_vs_cell.py \
    --dataset aitqa --retriever dense --budget 512 --arms flat,cell \
    --cell-scheme $SCH \
    --reader "local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit&dtype=float16" \
    --out results/h2h_untitled_aitqa_${SCH}_512.json
done

.venv/bin/python scripts/corpus_dump_vs_cell.py \
  --dataset realhitbench --retriever dense --budget 512 --arms flat,cell \
  --cell-scheme mt2net \
  --reader "local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit&dtype=float16" \
  --out results/h2h_untitled_rhb_mt2net_512.json
```
