# 내일 할 일 — 연구실(로컬 GPU 없음, Colab GPU) (2026-08-25)

작성 2026-08-24 03시. `NEXT.md`의 규칙을 따른다: **출처 없는 수치는 쓰지 않는다.**

---

## 0. 어제(08-24) 끝난 것 한 줄

**HiTab test split 확인 런 완료. 사전 등록한 예측 7개가 전부 맞았다.**
`PREREG-2026-08-24-hitab-test.md`(커밋 `f78f5ac`) → 결과 `01e7e13`.
우리 .592 대 MT2Net .492 (+.100, Holm 1.6e-07) / 1024에서 +.133 (Holm 5.9e-11).
`flat` 통제가 두 예산 모두 0:0. 출처 `results/test_split_prereg_verdict.json`.

→ **"dev에서 고르고 dev에서 보고한다"는 지적이 닫혔다.** 이제 논문의 HiTab 수치는
test로 보고하고 dev를 개발 세트로 명시한다.

---

## 1. 오늘의 주 작업 — 어순(leaf-first) 몫 분리

**코드와 예측은 어제 이미 커밋했다.** `PREREG-2026-08-25-word-order.md`.
오늘은 **돌리기만 한다.** 손잡이를 새로 만들지 말 것.

`--cell-scheme S2r`은 S2의 각 축을 leaf-first로 뒤집기만 한다. 전수 검증에서
**토큰 multiset이 100% 동일**(HiTab 58,759셀 / AIT-QA 5,320셀, 둘 다 tok/cell 소수
둘째 자리까지 같음)이라, **이 대조에서 나오는 차이는 용량으로 반박할 수 없다.**
이 저장소의 다른 템플릿 대조에는 전부 열려 있던 반박이 여기서는 원리상 닫힌다.

예측: **W1 — HiTab에서 |S2r − S2| ≤ .03, 유의하지 않음.** 근거는
RESEARCH_STRUCTURE §4.1 주장 4(순열 6회에서 순서 효과 미검출). 깨지면 주장 4를 다시 쓴다.

### 실행 (5개 런, 순서대로)

```bash
COMMON="--retriever dense --budget 512 --arms flat,cell \
  --reader local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit&dtype=float16"

# HiTab dev n=830 — 주 무대 (각 ~30분)
python3 scripts/corpus_dump_vs_cell.py $COMMON --dataset hitab \
  --population hitab_dev_lookup_all --cell-scheme S2     --out results/wo_hitab_s2_512.json
python3 scripts/corpus_dump_vs_cell.py $COMMON --dataset hitab \
  --population hitab_dev_lookup_all --cell-scheme S2r    --out results/wo_hitab_s2r_512.json
python3 scripts/corpus_dump_vs_cell.py $COMMON --dataset hitab \
  --population hitab_dev_lookup_all --cell-scheme mt2net --out results/wo_hitab_mt2net_512.json

# AIT-QA n=451 — 참고 (각 ~17분). 셀의 25.8%가 1단 경로라 뒤집을 것이 없어 희석된다
python3 scripts/corpus_dump_vs_cell.py $COMMON --dataset aitqa --cell-scheme S2  --out results/wo_aitqa_s2_512.json
python3 scripts/corpus_dump_vs_cell.py $COMMON --dataset aitqa --cell-scheme S2r --out results/wo_aitqa_s2r_512.json
```

**총 ~2시간.** Colab 세션 하나에 들어간다. 리더 스펙을 5개 런에서 통일할 것 —
기존 `h2h830_mt2net_512.json`은 bfloat16이라 짝지으면 리더 교란이 섞인다.

### 끝나고 할 것 (순서 지킬 것)

1. **W3(통제)부터.** `flat` arm이 세 HiTab 런에서 **0:0**인지 확인. 아니면 거기서 멈춘다.
   ```bash
   python3 scripts/paired_em_between_runs.py results/wo_hitab_s2r_512_records.jsonl \
       results/wo_hitab_s2_512_records.jsonl --out results/wo_s2r_vs_s2_hitab_512.json
   python3 scripts/paired_em_between_runs.py results/wo_hitab_s2r_512_records.jsonl \
       results/wo_hitab_mt2net_512_records.jsonl --out results/wo_s2r_vs_mt2net_hitab_512.json
   ```
2. W1·W2 2건에 Holm. `exact_p`가 `0.0`으로 나오면 언더플로다 —
   `scipy.stats.binomtest`로 다시 계산할 것(어제 1.6e-07이 그렇게 0.0으로 찍혔다).
3. CLAUDE.md §"문장 형태는 무의미"가 흔들린다 절의 **잠정 보류를 해제하거나 확정**한다.

---

## 2. GPU가 필요 없는 작업 — Colab이 돌아가는 동안 노트북에서

### 2-1. CLAUDE.md §"제목이 전부다"를 조건부 서술로 다시 쓰기 (남은 일 4번)

지금 그 절은 제목 붙은 경고문과 본문이 어긋나 있다. 확정된 사실은 셋이다:

* 제목은 **HiTab에서** 크게 번다 (+.15~+.18, `corpus_dump_vs_cell_s3*`)
* RealHiTBench(38.2% 보유)에서는 **시험했고 재현되지 않았다** — 시험 불가가 아니다
* 메커니즘은 **읽기 이득 − 검색 용량 손해**이고, 부호는 제목 보유율이 정한다
  (99.3% +.082 / 38.2% +.004 / 0% −.062, `PREREG-2026-08-23-mt2net-untitled.md`)

→ 절 제목을 "제목이 전부다"에서 **"제목은 읽기를 사고 용량을 판다"**로 바꾸고,
단조 표를 절의 머리로 올린다. **새 실험이 필요 없는 순수 문서 작업이다.**

### 2-2. 논문 표 초안 — test 수치로

`scripts/paper_table_headtohead.py`가 규칙을 이미 구현하고 있다. HiTab 열을 test로
갈아끼우고 dev를 개발 세트로 표기. RealHiTBench·AIT-QA는 dev/test 구분이 없으므로 그대로.

---

## 3. 결정이 필요한 것 — 더 강한 리더 (남은 일 3번)

**돈이 든다. 돌리기 전에 물어볼 것.** 실측한 호출량:

| 범위 | 호출 수 | 입력 토큰 | 출력 토큰 |
|---|---|---|---|
| HiTab dev n=830, cell arm만, 2 scheme(S3c/mt2net) | 1,660 | **약 1.30M** | 약 25K |
| 위 + flat 통제 arm | 3,320 | 약 2.6M | 약 50K |
| 부분표본 n=300으로 축소 | 600 | 약 0.47M | 약 9K |

산출 근거: 시스템 프롬프트 201토큰(`_DIRECT_SYS` 실측) + 문맥 512(예산) +
`"CONTEXT:\n…\n\nQUESTION: …\n\nAnswer:"` 골격·질문 약 65 → **호출당 입력 ≈ 780토큰**,
출력은 짧은 답 하나라 ≈ 15토큰.

**단가는 기억으로 적지 않는다.** 실행 직전에 공급자 가격표를 조회해 원화 환산까지
계산해서 물어볼 것. 저장소에 배선된 백엔드는 `local:` / `groq:` / `openai:`
(`rag_agent/llm/factory.py`)뿐이다 — 다른 공급자를 쓰려면 백엔드부터 추가해야 한다.

⚠️ **왜 필요한가**: 로컬 4-bit 7B는 산술 레그에서 천장이 낮다(계산기를 붙이고 재료를
다 줘도 78% 틀림, CLAUDE.md §codegen). 조회형 레그(.59/.60)는 로컬로 충분하지만,
**"리더가 약해서 나온 결과 아니냐"**는 지적을 닫으려면 강한 리더 한 점이 필요하다.
**우선순위는 1번 아래다** — 어순 결론이 먼저다.

---

## 4. Colab 셋업 (처음 한 번만, 15~20분)

로컬 WSL 환경: RTX 3060 Ti 8GB / torch 2.12.1+cu130 / Python 3.12.3 / `rag-agent/.venv`.
Colab T4 16GB면 4-bit 7B(약 5.5GB)와 bge-small이 동시에 올라간다. **더 여유롭다.**

```python
# 1) 저장소
!git clone -b fix/encoder-provenance https://github.com/Jax0303/T2.git
%cd T2/rag-agent
!pip install -q -e . bitsandbytes accelerate

# 2) HiTab (279MB) — 스크립트가 GitHub에서 받아온다
!PYTHONPATH=. python3 scripts/download_hitab.py --dest data/hitab

# 3) AIT-QA — 다운로더가 없다. data/aitqa/{aitqa_tables,aitqa_questions}.jsonl
#    256KB짜리 둘뿐이니 Google Drive에 올려두고 복사하는 게 가장 빠르다
from google.colab import drive; drive.mount('/content/drive')
!mkdir -p data/aitqa && cp /content/drive/MyDrive/aitqa/*.jsonl data/aitqa/

# 4) 확인 — 모집단이 그대로 파생되는가 (여기서 DRIFT가 뜨면 데이터가 다른 것)
!PYTHONPATH=. python3 scripts/freeze_populations.py --check --only hitab_dev_lookup_all,hitab_test_lookup_all
```

**출발 전에 할 일**: `data/aitqa/*.jsonl` 두 개를 Google Drive `MyDrive/aitqa/`에
올려둘 것. 256KB라 몇 초면 끝난다. (`rag-agent/data/`가 gitignore라 저장소에 없다.)

### Colab에서 주의할 것

* **세션이 끊긴다.** 런은 resume을 지원하므로(`guard_resume`) 재접속 후 같은 명령을
  다시 치면 이어간다. `--out`을 바꾸지 말 것.
* **결과를 커밋해서 가져올 것.** Colab 디스크는 세션과 함께 사라진다.
  런이 끝날 때마다 `results/*.json`과 `*_records.jsonl`을 커밋·푸시한다.
* **Qwen 가중치 약 15GB를 매 세션 받는다.** Drive에 HF 캐시를 두면
  (`HF_HOME=/content/drive/MyDrive/hf`) 재다운로드를 피할 수 있으나, Drive I/O가
  느려 첫 로드가 오히려 더 걸릴 수 있다. **한 세션에 5개 런을 몰아서 돌리는 편이 낫다.**
* **`--reader` 문자열의 `&`를 셸이 먹는다.** 반드시 따옴표로 감쌀 것.

---

## 5. 오늘 하지 말 것

* **test split을 다시 건드리지 말 것.** 어제 한 번 썼고, 손잡이를 돌려 다시 돌리는
  순간 두 번째 dev가 된다 (RESEARCH_STRUCTURE §3.1).
* **검색 지표(OSC)로 방법을 고르지 말 것.** 어제 test에서도 네 칸 전부 MT2Net의
  OSC가 높은데 EM은 전부 졌다.
* **하이브리드·리랭커·큰 인코더를 다시 시도하지 말 것.** 셋 다 죽었다
  (CLAUDE.md §성능을 올리려고 시도한 검색기 레버).
* **`짝지은 비교에서 flat이 0:0인지 확인하지 않고 결론 내지 말 것.**
  2026-08-24에 그 검사가 리더 교란을 잡아냈고, 어제는 파이프 뒤의 종료 코드를 믿었다가
  크래시한 런을 `OK`로 읽을 뻔했다.
