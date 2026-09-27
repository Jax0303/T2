"""2026-09-27 병목 진단 절 초안용 숫자 목록 -> thesis_numbers.csv (값, 출처 파일, 필드). 값은 결과 JSON 의 잎 값을 그대로 옮긴다.
필드 = JSON 경로(점으로 이음). 경로·정의 문자열(inputs, meta, bucket_definitions, reranker, rule)은 뺀다.
diag upper_bound.906_plus_B_plus_20 은 뺀다 — '20' 이 결과 파일이 아니라 지시문 값이라서.
실행: python3 results/bottleneck_20260927/thesis_numbers.py
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
SOURCES = [("GATE1 진단(test, s3c)", "results/bottleneck_20260927/diag/summary.json"),
           ("재정렬 dev", "results/bottleneck_20260927/dev/summary.json"),
           ("재정렬 dev 처리량", "results/bottleneck_20260927/dev/throughput.json"),
           ("재정렬 test(s3c 단독)", "results/bottleneck_20260927/test/summary.json"),
           ("2026-09-17 재정렬(sleaf, 1위 셀 지표)", "results/cross_encoder_rerank_20260917/summary.json")]
SKIP_KEYS = {"inputs", "meta", "bucket_definitions", "reranker", "rule", "note"}
SKIP_PATHS = {"hitab_538.upper_bound.906_plus_B_plus_20"}


def leaves(x, path=()):
    if isinstance(x, dict):
        for k, v in x.items():
            if k not in SKIP_KEYS:
                yield from leaves(v, (*path, k))
    elif not isinstance(x, list):
        yield ".".join(path), x


with open(HERE / "thesis_numbers.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["구분", "값", "출처 파일", "필드"])
    n = 0
    for sec, rel in SOURCES:
        for field, val in leaves(json.loads((ROOT / rel).read_text())):
            if field not in SKIP_PATHS:
                w.writerow([sec, val, f"rag-agent/{rel}", field])
                n += 1
print(n, "rows")
