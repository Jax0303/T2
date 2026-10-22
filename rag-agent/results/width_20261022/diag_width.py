import json, sys, collections
from pathlib import Path
ROOT = Path("/home/ugh/T2/rag-agent")
sys.path[:0] = [str(ROOT / "results/format_dev_20261017"), str(ROOT), str(ROOT / "scripts")]
import score_formats2 as sf
from mh_arms import official_answers
from format_render import load_split
G = official_answers("train")
jobs = json.load(open(ROOT / "results/budget_forcing_20261022/jobs.json")); P = "R1X1F1B2D1C1E0"
def load(p): return {r["query_id"]: sf.score(r, G[r["query_id"]])["fin_inc"] for r in map(json.loads, open(p, encoding="utf-8"))}
for S in ("600b", "882"):
    Arec = [j for j in jobs if j["set"] == S and j["variant"] == P and j["arm"] == "A"][0]["records"]
    R = {r["query_id"]: r for r in map(json.loads, open(Arec, encoding="utf-8"))}
    ids = sorted(R); tables, hdr, sent, coord = load_split("train", ids)
    feat = {}
    for q in ids:
        tids = {f"{q}::{g.split('-')[0]}" for g in R[q]["gold_ids"]}
        ws, ds = [], []
        for tid in tids:
            if tid not in hdr: continue
            t = getattr(tables[tid], "table", tables[tid]); nhr, nhc = hdr[tid]
            ws.append(t.n_cols + nhc); ds.append(nhr)
        if not ws: continue
        feat[q] = {"width": max(ws), "depth": max(ds), "ntab": len(tids)}
    runs = sorted({j["run"] for j in jobs if j["set"] == S})
    sc = {}
    for run in runs:
        J = {j["arm"]: j["out"] for j in jobs if j["set"] == S and j["run"] == run and j["variant"] == P}
        sc[run] = {a: load(J[a]) for a in "DCA"}
    def bucket(name, fn):
        groups = collections.defaultdict(list)
        for q in feat: groups[fn(feat[q])].append(q)
        print(f"\n[{S}] {name}")
        for k in sorted(groups):
            qs = groups[k]; n = len(qs)
            dA = [sum(sc[r]["D"][q] - sc[r]["A"][q] for q in qs) for r in runs]
            cA = [sum(sc[r]["C"][q] - sc[r]["A"][q] for q in qs) for r in runs]
            a = [sum(sc[r]["A"][q] for q in qs) for r in runs]
            print(f"  {k}: {n}문항 | LangChain 맞힘 {sum(a)/len(a):.0f} ({100*sum(a)/len(a)/n:.1f}%) | 잘라낸 표−LC 실행별 {dA} 평균 {sum(dA)/len(dA)/n*100:+.1f}%p | 목록−LC {cA} 평균 {sum(cA)/len(cA)/n*100:+.1f}%p")
    bucket("표 너비(열 수)", lambda f: "1) ≤4" if f["width"] <= 4 else "2) 5~8" if f["width"] <= 8 else "3) 9+")
    bucket("위쪽 머리글 줄 수", lambda f: f"{min(f['depth'],3)}{'+' if f['depth']>=3 else ''}")
    bucket("정답이 걸친 표 수", lambda f: "1" if f["ntab"] == 1 else "2+")
