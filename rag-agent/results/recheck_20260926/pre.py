"""2026-09-26 MultiHiertt c·e 재실행 전 점검 (인코딩 전, 리더 없음).
c = 라벨 없음, e = L1 라벨. 둘 다 머리글 v2 · s3c 셀 문장 (results/mh_arms/mh_cell_hv2{,_L1}.json 과 같은 인자).
1. 현재 코드로 만든 셀 수 == 425,870 이고 셀 문장 해시 == 기존 실행인지. 아니면 종료 코드 1.
2. 인코더 한도(512 토큰) 초과 셀 수 — 기존 실행은 검사 없이 모델 한도에서 잘랐다.
3. 셀 1,000개(고정 시드 무작위)를 인코딩한 시간 -> 셀 전체 인코딩 예상 시간. 모델 적재·첫 배치(64개) 예열은 빼고 잰다.
실행: HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python results/recheck_20260926/pre.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.argv = sys.argv[:1]
import mh_arms as mh                                                  # noqa: E402

OUT = Path(__file__).parent
RUNS = {"c": ("none", "mh_cell_hv2"), "e": ("L1", "mh_cell_hv2_L1")}
N_CELLS = 425870

queries, docs, _ = mh.load_population("train")
enc = mh.default_encoder(model_name="BAAI/bge-base-en-v1.5")
res, ok = {}, True
for k, (label, stem) in RUNS.items():
    old = json.loads((ROOT / f"results/mh_arms/{stem}.json").read_text())
    tables, _hdr = mh.build_tables(docs, "v2", label)
    texts, *_ = mh.build_corpus("", sorted(tables), "s3c", "cell", {}, load=lambda tid, _d: tables.get(tid))
    audit = enc.audit_inputs(texts, overflow="truncate")
    sample = [texts[i] for i in np.random.default_rng(0).choice(len(texts), 1000, replace=False)]
    enc.encode(sample[:64])                                           # 예열
    t0 = time.perf_counter()
    enc.encode(sample)
    sec = time.perf_counter() - t0
    res[k] = {"old_summary": f"results/mh_arms/{stem}.json", "n_cells": len(texts),
              "n_cells_old": old["n_units"], "sha_equal_old": mh.digest(texts) == old["corpus_text_sha256"],
              "n_tables": len(tables), "n_overflow_512": audit["n_overflow"], "max_tokens": audit["max_tokens"],
              "sec_per_1000": round(sec, 2), "est_encode_sec": round(sec / 1000 * len(texts))}
    ok &= len(texts) == N_CELLS and res[k]["sha_equal_old"]
    del tables, texts
res["est_encode_sec_total"] = sum(res[k]["est_encode_sec"] for k in RUNS)
res["encoder"] = enc.metadata()
(OUT / "pre.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
sys.exit(0 if ok else 1)
