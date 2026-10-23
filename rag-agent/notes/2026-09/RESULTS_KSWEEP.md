# RESULTS_KSWEEP — 컨텍스트 크기 k vs 리더 EM (2026-09-18)

실행 일시: 2026-09-18T06:49:13+0900
커밋 해시: ba1e8859b7d89d6cd0f874dc74cc29d8aa111597 (dirty=True)
실패 쿼리 수: 0 (LEG A, LEG B 전체 k 합산)

검색: P4_path_cell(template=s3c, unit=cell), BGE-Large-En(BAAI/bge-large-en-v1.5), top-20, corpus=split, 1회 실행
(`results/retrieval_accuracy/s3c_bgelarge_v2_records.jsonl`, records_sha256=71cea5eab085bcca422733b471f256e72592588fe156ac6cc7a79f75f1fb9d5e)
리더: local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit, prompt=neutral, max_new_tokens=64, seed=42, temperature=0, do_sample=False
모집단: HiTab primary population(mode=all, m=1, aggregation=none) 991건 중 stratified_sample(n=300, seed=42) 300건

## LEG A — 자연 절단 (main), query count=300

| k | EM | 95% CI (bootstrap B=10000) | 정답 셀 포함 쿼리 비율 |
|---|---:|---|---:|
| 1 | 0.5567 | [0.5000, 0.6133] | 0.5467 |
| 2 | 0.6200 | [0.5633, 0.6733] | 0.7000 |
| 3 | 0.6533 | [0.6000, 0.7067] | 0.7367 |
| 5 | 0.6967 | [0.6433, 0.7467] | 0.7867 |
| 10 | 0.7267 | [0.6767, 0.7767] | 0.8533 |
| 20 | 0.7667 | [0.7200, 0.8133] | 0.9167 |

## LEG B — GOLD-FORCED (not deployable), query count=275

대상 쿼리 수: 275 / 제외 쿼리 수: 25 (300건 중 top-20 검색 실패)

제외 쿼리 ID:
022faef43c2d5f631f5d83e9e75f346a, 0a2938b5924faf566b2ccd1061dc1412, 0c221c1ce1ef03c3f187b4c5b5ab7c1d,
0c863f592b8b2b1612dbded809c4f631, 28f3628abf672ac0297baa46b2a6253c, 34a68ac803015ace67d2b57caf7c27e0,
363f9bb3f74c7ec86026ee4e8707dfb8, 38dd34ceca312b80673c5eb605e97dec, 4b9ff117dd34bd1ec321834ea90ded6b,
5839d6a5a46d6c60a36aae0f3884f5e3, 61143c012eadeb4070c9f6ce7072fa8b, 6f6c406a0ed987e0fe507bc434c02a46,
720e2c9c6caa13f0942e5bc2e444d498, 78a19d0e2e5a0ad6398d3389b7d46e98, 8db22461be5d09fe27d62d3b758fad97,
8ec692fe85b783ce8c4caf34842f4335, a26d78b9369e34ddbc16b0abb4062bb7, a9f09b29ee1e45b5d9c2165b008015e5,
ba8d1b2dabb97dc7333783ced6df77bf, c16818cb84fdc3c87a1eeec3037e1b4b, c53375d3ec90313b7c70f6b6502769c4,
c926b3d861ea24989d873c6f781bfce9, dffc23f130e35c1cd12bea73411a878f, e1717106e44ab2792531adc4e4f23a94,
eea2dc4d8aa18842ce53df88d297236a

| k | EM | 95% CI (bootstrap B=10000) |
|---|---:|---|
| 1 | 0.9527 | [0.9273, 0.9782] |
| 2 | 0.8255 | [0.7818, 0.8691] |
| 3 | 0.8764 | [0.8364, 0.9127] |
| 5 | 0.8655 | [0.8255, 0.9055] |
| 10 | 0.8618 | [0.8218, 0.9018] |
| 20 | 0.8291 | [0.7855, 0.8727] |

최고 EM: k=1 (0.9527)

McNemar (k=1 vs k=20, exact, query count=275):
- k=1에서만 정답: 41
- k=20에서만 정답: 7
- discordant: 48
- p-value: 0.0000

## 산출물

- `results/ksweep_legA.json` — 쿼리 ID별 (leg, k, 예측, 정답, EM)
- `results/ksweep_legB.json` — 쿼리 ID별 (leg, k, 예측, 정답, EM)
- `results/ksweep_raw/leg{A,B}_k{k}.jsonl` — 실행 원본 로그 (재개 가능 형식)
- `results/ksweep_population_300.json` — 고정 300-쿼리 목록 및 표본 방법
- `results/retrieval_accuracy/s3c_bgelarge_v2*` — 검색 캐시 (1회 실행)
