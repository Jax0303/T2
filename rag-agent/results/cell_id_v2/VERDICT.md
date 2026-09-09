# 판정: 셀 ID 선택 + 코드 값 반환 — **기각** (2026-09-10)

사전등록 `PREREG-2026-09-10-cell-id-selection-v2.md`
(sha256 `f8c4dae529f04aa37edad4a73fc966a97aea48a43ffdc2b55211e015d8c7d3d9`).
설정은 dev 에서 고르고 test 는 한 번만 돌렸다. 아래가 그 한 번이다.

## 주지표 (단일 셀 조회 991문항)

    기준선 S3c 자유 생성   710/991 = .7164
    셀 ID 선택 + 값 반환   627/991 = .6327
    Δ                     -.0838   95% CI [-.1100, -.0575]
    새로 맞힘 49 / 새로 틀림 132 / 순 -83
    paired exact McNemar  p = 5.5e-10
    형식 파손율           3.23% (중단 조건 5% 미만 — 이번엔 장치가 건전하다)
    처리 시간             1,386.7s, 0.877 s/문항 (RTX 3060 Ti, 4bit)

**사용자 목표 0.8 (793/991) 미달이다.** 개선도 아니고 통계적으로 유의한 하락이다.
지지 조건(Δ ≥ .02 AND p < .05)은 방향이 반대라 충족되지 않는다.

## 나머지 유형 (기준선 → 장치)

    단일 조회 (주지표) 991   710 → 627
    복수 조회           38    19 →   9
    산술               216    26 →   7
    헤더가 답 (any)    336   142 →   1
    데이터셀 전체     1245   755 → 643
    채점 전체         1581   897 → 644

헤더가 답인 336건이 142 → 1 로 무너진 것은 **설계상 당연**하다. 이 장치는 데이터 셀의
저장값만 반환할 수 있고 헤더 문자열은 반환할 수 없다. 산술 216건도 같은 이유다.
주지표는 이 유형을 포함하지 않으므로 주지표 하락과는 별개다.

## v1 결함이 얼마였나 (같은 생성물, 파싱 규칙만 교체)

    hash (v1 규칙)       599/991 = .6044   파손율 15.75%
    bracket              627/991 = .6327   파손율  3.48%
    bracket_bare (채택)  627/991 = .6327   파손율  3.23%

v1 이 보고한 .5943 중 **+38문항이 파서 결함**이었다. 결함을 고쳐도 기준선을 넘지 않는다.
(v1 과 .6044 가 다른 것은 ID→값 사전도 함께 고쳐 unresolved 20건이 사라졌기 때문이다.)

## 어디서 지는가 — 검색 성공군 분해 (gold 는 여기서만 사용)

    주지표 991 중
      정답                       627
      검색 실패 (문맥에 셀 없음)   82
      **셀 선택 오류**            255
      값 반환 오류                 11
      ID 미출력 (기권 등)          16
      검색 성공인데 채점 가능한 값이 문맥에 없음   0

**값 채널은 닫혔다.** 사전 측정에서 58건이던 값 생성 오류가 11건으로 줄었고, 검색
성공 906건 전부가 단일 셀 선택으로 도달 가능함을 확인했다(장치 상한 910/991 = .9183).
그런데 셀 선택 오류가 139건(사전 측정)에서 255건으로 늘었다.

새로 틀린 132건의 내역: 단일 ID 를 냈는데 셀이 틀림 118, ID 미출력 9, 복수 ID 5.
복수 ID 를 낸 28건은 **전부 오답**이었다.

## 읽기

가설은 "셀은 맞게 고르고 값을 옮겨 적다 깨진다" 였다. 절반만 맞다. 값 채널은 실제로
있었고 닫혔다(58 → 11). 그러나 **답을 직접 쓰는 것보다 답이 있는 셀을 지목하는 것이
이 리더에게 더 어렵다.** 자유 생성은 20줄에서 값을 뽑아내는 데는 성공하면서 그 값이
몇 번 줄에 있었는지는 못 세는 것이다. 순위 번호를 세는 부담이 값 오류 회수분보다 크다.

이 방향은 dev 에서 먼저 같은 신호를 냈다(장치 .6826 vs 자유 생성 .7478, n=460).
test 는 그 예측을 확인했을 뿐이다.

## 하지 않은 것

- MT2Net 등 대조 방법에 같은 절차를 적용하지 않았다. 사용자 조건이 "효과가 확인되면"
  이었고 효과가 확인되지 않았다.
- 결과를 보고 프롬프트·파싱 규칙을 바꿔 재시도하지 않았다. test 는 한 번만 돌렸다.
- oracle 수치를 EM 으로 보고하지 않았다. 910/991 은 장치 상한이지 성능이 아니다.

## 재현

    PYTHONPATH=. .venv/bin/python scripts/retrieval_accuracy.py --split dev \
      --corpus split --unit cell --template s3c --alpha 0.7 --budget 20 \
      --dump-context 20 --tag d_s3c_hybrid --out-dir results/cell_id_v2
    PYTHONPATH=. .venv/bin/python scripts/cell_id_select.py audit --split test
    PYTHONPATH=. .venv/bin/python scripts/cell_id_select.py run --split dev --arm base --limit 700 \
      --prereg PREREG-2026-09-10-cell-id-selection-v2.md      # p1, p2 도 같은 형태
    PYTHONPATH=. .venv/bin/python scripts/cell_id_select.py score --split dev --limit 700
    PYTHONPATH=. .venv/bin/python scripts/cell_id_select.py run --split test --arm p1 \
      --prereg PREREG-2026-09-10-cell-id-selection-v2.md
    PYTHONPATH=. .venv/bin/python analysis/cell_id_report.py --arm p1 --rule bracket_bare

입력 해시: 검색 기록 `e6e8101d…7997c96cd` (test) / `abc53bbc…3d43c16925` (dev),
사전등록 `f8c4dae5…8d7c3d9`. 나머지 해시는 `test_p1.meta.json`.
리더 `local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit`, temp 0, seed 42, 64 tok, batch 1.
