"""지시문 변형 채점·선택 표 — PREREG-2026-10-15-prompt-sweep.md.
주 채점 = 정답 포함 v2(`results/score_fix_20261014/score_v2.included_v2`; 예/아니오는 첫 yes/no 낱말 규칙, 판정표 없음).
선택 지표 W = 0.579·조회 정확도 + 0.421·산술 정확도(882 의 조회 511 / 산술 371 비율), S = A·C 두 전달 입력의 W 평균.
보조: 공식 EM('Final answer:' 줄 추출, 없으면 출력 전체), 잘림(finished=False) 수, 출력 토큰 평균, 축별 주변 평균.
  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/prompt_sweep_20261015/score_sweep.py <out-dir> [--split train] [--md 표.md]
  (dev 레코드는 answer 를 품고, 882 레코드는 --split train 공식 정답으로 읽는다)
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "results/cellform_20261012"), str(ROOT / "results/score_fix_20261014")]
from answer_accuracy_mh import extract                      # noqa: E402
from mh_arms import official_answers                        # noqa: E402
from rag_agent.eval.multihiertt_em import mh_exact_match    # noqa: E402
from score_inclusion import gold_kind                       # noqa: E402
from score_v2 import included_v2                            # noqa: E402
import prompt_variants as pv                                # noqa: E402

W_LOOKUP, W_ARITH = 511 / 882, 371 / 882


def score_file(p: Path, gold: dict):
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    out = {}
    for r in rows:
        g, arith = gold[r["query_id"]], r["kind"] == "arith"
        pred, marked = extract(r["raw"])
        out[r["query_id"]] = {"inc": int(included_v2(r["raw"], g, arith)), "em": int(mh_exact_match(pred, g, arith)),
                              "kind": r["kind"], "layer": r["layer"], "gk": gold_kind(g), "fin": int(r["finished"]),
                              "marked": int(marked), "n_out": r["n_tok_out"], "dk": int("don't know" in r["raw"].lower() or "do not know" in r["raw"].lower())}
    return out


def summarize(S: dict):
    n = len(S)
    by = lambda f: [v for v in S.values() if f(v)]
    acc = lambda vs, k="inc": (sum(v[k] for v in vs) / len(vs)) if vs else 0.0
    lk, ar = by(lambda v: v["kind"] == "lookup"), by(lambda v: v["kind"] == "arith")
    W = W_LOOKUP * acc(lk) + W_ARITH * acc(ar)
    return {"n": n, "inc": sum(v["inc"] for v in S.values()), "acc": round(acc(list(S.values())), 4),
            "lookup": round(acc(lk), 4), "arith": round(acc(ar), 4), "W": round(W, 4), "em": round(acc(list(S.values()), "em"), 4),
            "yesno_inc": sum(v["inc"] for v in S.values() if v["gk"] == "yesno"), "yesno_n": sum(v["gk"] == "yesno" for v in S.values()),
            "truncated": sum(1 - v["fin"] for v in S.values()), "marked": sum(v["marked"] for v in S.values()),
            "dont_know": sum(v["dk"] for v in S.values()), "out_tok": round(sum(v["n_out"] for v in S.values()) / n, 1),
            "by_layer": {L: round(acc(by(lambda v, L=L: v["layer"] == L)), 4) for L in sorted({v["layer"] for v in S.values()})}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--split", default="", help="비워 두면 레코드의 answer 필드(dev); train 이면 공식 정답")
    ap.add_argument("--records", default="", help="정답을 읽을 레코드(기본 results/prompt_sweep_20261015/dev_A_records.jsonl)")
    ap.add_argument("--md", default="")
    ap.add_argument("--expect", type=int, default=0, help="완료 판정 문항 수(기본 레코드 수)")
    a = ap.parse_args()
    D = Path(a.out_dir)
    if a.split:
        gold = official_answers(a.split)
    else:
        gold = {r["query_id"]: r["answer"] for r in map(json.loads, open(a.records or ROOT / "results/prompt_sweep_20261015/dev_A_records.jsonl", encoding="utf-8"))}
    expect = a.expect or len(gold)
    res = defaultdict(dict)                                      # variant -> arm -> summary
    per = defaultdict(dict)                                      # variant -> arm -> {qid: inc}
    for arm_dir in sorted(D.iterdir()):
        if not arm_dir.is_dir():
            continue
        for p in sorted(arm_dir.glob("*.jsonl")):
            S = score_file(p, gold)
            if len(S) < expect:
                res[p.stem][arm_dir.name] = {"n": len(S), "incomplete": True}
                continue
            res[p.stem][arm_dir.name] = summarize(S)
            per[p.stem][arm_dir.name] = {q: v["inc"] for q, v in S.items()}
    rows = []
    for v, arms in res.items():
        full = [x for x in arms.values() if not x.get("incomplete")]
        s = round(sum(x["W"] for x in full) / len(full), 4) if full and len(full) == len(arms) else None
        rows.append((v, s, arms))
    rows.sort(key=lambda t: (-(t[1] if t[1] is not None else -1), t[0]))
    base = "R0X0F0B0D0C0E0"
    lines = [f"| 변형 | S | " + " | ".join(f"{arm} 맞힘/W/조회/산술/EM/잘림/모른다/출력토큰" for arm in ("A", "C")) + " | 기준 대비 b:c (A, C) |", "|---|---|---|---|---|"]
    for v, s, arms in rows:
        cells = []
        for arm in ("A", "C"):
            x = arms.get(arm)
            if not x or x.get("incomplete"):
                cells.append(f"{(x or {}).get('n', 0)}/… 진행 중")
            else:
                cells.append(f"{x['inc']}/{x['W']:.3f}/{x['lookup']:.3f}/{x['arith']:.3f}/{x['em']:.3f}/{x['truncated']}/{x['dont_know']}/{x['out_tok']}")
        bc = []
        for arm in ("A", "C"):
            if arm in per.get(base, {}) and arm in per.get(v, {}) and v != base:
                b0, p1 = per[base][arm], per[v][arm]
                bc.append(f"{sum(b0[q] and not p1[q] for q in b0)}:{sum(p1[q] and not b0[q] for q in b0)}")
        lines.append(f"| {v} | {s if s is not None else '…'} | " + " | ".join(cells) + f" | {' '.join(bc)} |")
    # 축별 주변 평균(완료 변형만, S 기준)
    done = [(v, s) for v, s, _ in rows if s is not None]
    marg = {}
    for k in pv.ORDER:
        for lv in range(len(pv.AXES[k])):
            vs = [s for v, s in done if pv.parse(v)[k] == lv]
            if vs:
                marg[f"{k}{lv}"] = (round(sum(vs) / len(vs), 4), len(vs))
    lines.append("\n축별 주변 평균 S(완료 변형 수): " + ", ".join(f"{k}={m[0]:.3f}({m[1]})" for k, m in marg.items()))
    text = "\n".join(lines)
    print(text)
    json.dump({"rows": [{"variant": v, "S": s, "arms": arms} for v, s, arms in rows], "marginal": marg},
              open(D / "score.json", "w"), ensure_ascii=False, indent=1)
    if a.md:
        Path(a.md).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
