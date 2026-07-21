#!/usr/bin/env python3
"""진단 보강 — HiTab(계층) 질의를 난이도/집계유형별로 층화해 C0/C1/C2 R@1 분해.
어떤 질의 유형에서 전처리가 무너지는지(=방법론 조준점) 식별. 캐시된 임베딩 재사용.
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag_agent.eval.metrics import difficulty_class

EMB = Path("diag/emb_cache")


def load_ids(cond):
    return [json.loads(l)["table_id"] for l in open(f"diag/hier/serialized/{cond}.records.jsonl")]


def main():
    queries = [json.loads(l) for l in open("diag/hier/queries.jsonl")]
    queries = [q for q in queries if q["split"] == "dev"]
    golds = [q["gold_table_id"] for q in queries]
    q_emb = np.load(EMB / "hier_q_dev_1671.npy")
    qd = torch.tensor(q_emb, device="cuda")

    # 층화 키: difficulty_class(질문+집계라벨)
    def cls(q):
        return difficulty_class({"aggregation": q.get("aggregation_label") or ["none"],
                                 "answer_formulas": q.get("answer_formulas")})
    strata = [cls(q) for q in queries]

    per_cond_r1 = {}
    ids_ref = load_ids("C0")
    for cond in ("C0", "C1", "C2"):
        ids = load_ids(cond)
        cemb = np.load(EMB / f"hier_{cond}.npy")
        cd = torch.tensor(cemb, device="cuda").T
        r1 = []
        for i in range(0, qd.shape[0], 512):
            sims = qd[i:i+512] @ cd
            top1 = torch.topk(sims, 1, dim=1).indices.cpu().numpy().ravel()
            for j, gi in zip(top1, range(i, min(i+512, qd.shape[0]))):
                r1.append(int(ids[j] == golds[gi]))
        per_cond_r1[cond] = np.array(r1)

    # 층화 집계
    by = defaultdict(lambda: defaultdict(list))
    for k, s in enumerate(strata):
        for cond in ("C0", "C1", "C2"):
            by[s][cond].append(per_cond_r1[cond][k])
    rows = []
    for s in sorted(by):
        n = len(by[s]["C0"])
        r = {"class": s, "n": n}
        for cond in ("C0", "C1", "C2"):
            r[cond] = round(float(np.mean(by[s][cond])), 3)
        r["dC1_C0"] = round(r["C1"] - r["C0"], 3)
        r["dC2_C1"] = round(r["C2"] - r["C1"], 3)
        rows.append(r)
    out = {"dataset": "hier", "split": "dev", "strata": rows}
    json.dump(out, open("results/diag_hier_stratified.json", "w"), indent=2)
    print(f"{'class':28s} {'n':>5s} {'C0':>6s} {'C1':>6s} {'C2':>6s} {'dC1-C0':>7s} {'dC2-C1':>7s}")
    for r in rows:
        print(f"{r['class']:28s} {r['n']:5d} {r['C0']:6.3f} {r['C1']:6.3f} {r['C2']:6.3f} {r['dC1_C0']:+7.3f} {r['dC2_C1']:+7.3f}")


def _no_args() -> None:
    """This script takes no options. Without a parser, argparse-style flags are
    silently ignored and the full experiment runs anyway — which is how a bare
    ``--help`` sweep silently regenerated committed artifacts."""
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()


if __name__ == "__main__":
    _no_args()
    main()
