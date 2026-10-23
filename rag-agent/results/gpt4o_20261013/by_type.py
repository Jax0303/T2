"""GPT-4o 4조건을 2026-10-12 보고 PPT 표(조회 4유형 + 산술 2유형)와 같은 행으로 나눈다 (사후 분석, 2026-10-13).

PPT 표의 유형 분류 코드는 저장소에 없다(집 PC 세션에서 만듦). 여기 규칙은 그것을 가장 가깝게 되살린 것이다:
  예/아니오 = 공식 정답이 yes/no → 54(PPT 54)
  개수 세기 = 질문에 "how many" → 41(PPT 40)
  값을 답함 = 공식 정답이 문서 table_description 의 어느 셀 값과 같음(숫자는 $,% 공백 제거 뒤 상대 1e-6) → 167(PPT 166)
  머리글(항목 이름)을 답함 = 나머지 → 249(PPT 251)
  산술은 정답 셀 수(m) 1 / 2개 이상 → 71 / 300(PPT 와 같음)
채점 = 정답 포함(score_inclusion.included). 행마다 같은 문항 정확 McNemar(b = 앞 조건만 맞힘, c = 뒤 조건만 맞힘).
  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/gpt4o_20261013/by_type.py
"""
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "results/cellform_20261012")]
from score_inclusion import gold_kind, included                        # noqa: E402

D = ROOT / "results/gpt4o_20261013"
C = ROOT / "results/cellform_20261012"
FILES = {"closed": D / "ans_closed", "fulldoc": D / "ans_fulldoc", "ours": D / "ans_ours", "langchain": D / "ans_langchain",
         "qwen8b4_ours": C / "C_ours_budget", "qwen8b4_langchain": C / "A_chunk_md"}
path = hf_hub_download("yilunzhao/MultiHiertt", "multihiertt_data/train.json", repo_type="dataset",
                       revision="f18473da528dede3d9ce2274366d9ee8102ea0fd")
OFF = {x["uid"]: x for x in json.load(open(path, encoding="utf-8"))}
TYPES = ("조회: 머리글(항목 이름)을 답함", "조회: 값을 답함", "조회: 개수 세기", "조회: 예/아니오", "산술: 정답 셀 2개 이상", "산술: 정답 셀 1개")


def num(s):
    s = re.sub(r"[\$,%\s]", "", str(s)).strip("()")
    try:
        return float(s)
    except ValueError:
        return None


def same(v, a):
    return (num(v) is not None and num(a) is not None and abs(num(v) - num(a)) <= 1e-6 * max(1, abs(num(a)))) or v == a


def qtype(r):
    e = OFF[r["query_id"]]
    a = str(e["qa"]["answer"]).strip()
    if r["kind"] == "arith":
        return TYPES[5] if int(r["m"]) == 1 else TYPES[4]
    if gold_kind(a) == "yesno":
        return TYPES[3]
    if re.search(r"\bhow many\b", r["question"], re.I):
        return TYPES[2]
    vals = [d.strip().rstrip(".").strip().rsplit(" is ", 1)[-1].strip() for d in e["table_description"].values()]
    return TYPES[1] if any(same(v, a) for v in vals) else TYPES[0]


def load(d):
    rows = {}
    for r in map(json.loads, open(d / "langchain_chunk_answer.jsonl", encoding="utf-8")):
        r["inc"] = int(included(r["raw"], OFF[r["query_id"]]["qa"]["answer"], r["kind"] == "arith"))
        rows[r["query_id"]] = r
    return rows


def mcnemar(a, b, ids):
    bb = sum(a[q]["inc"] and not b[q]["inc"] for q in ids)
    cc = sum(b[q]["inc"] and not a[q]["inc"] for q in ids)
    return {"b": bb, "c": cc, "p": float(binomtest(min(bb, cc), bb + cc, 0.5).pvalue) if bb + cc else 1.0}


def main():
    R = {k: load(p) for k, p in FILES.items()}
    ids = sorted(R["ours"])
    assert all(set(v) == set(ids) for v in R.values()) and len(ids) == 882
    T = {q: qtype(R["ours"][q]) for q in ids}
    groups = {t: [q for q in ids if T[q] == t] for t in TYPES}
    groups["전체"] = ids
    out = {"rule": __doc__.strip().splitlines()[2:8], "type_counts": {t: len(g) for t, g in groups.items()}, "rows": {}}
    pairs = {"ours_vs_langchain": ("ours", "langchain"), "ours_vs_fulldoc": ("ours", "fulldoc"),
             "qwen_ours_vs_langchain": ("qwen8b4_ours", "qwen8b4_langchain")}
    print("| 쿼리 유형 | 문항 수 | 질문만 | 표 전부 | LangChain | 본 방법 | 본-LC | 본-표전부 | Qwen LC | Qwen 본 | 차이 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for t, g in groups.items():
        pct = {k: round(100 * sum(R[k][q]["inc"] for q in g) / len(g), 1) for k in FILES}
        mc = {k: mcnemar(R[a], R[b], g) for k, (a, b) in pairs.items()}
        out["rows"][t] = {"n": len(g), "pct": pct, "mcnemar": mc}
        f = lambda k: f"{round(pct[k[0]] - pct[k[1]], 1):+}({mc[k[2]]['b']}:{mc[k[2]]['c']} p={mc[k[2]]['p']:.3f})"
        print(f"| {t} | {len(g)} | {pct['closed']} | {pct['fulldoc']} | {pct['langchain']} | {pct['ours']} | "
              f"{f(('ours','langchain','ours_vs_langchain'))} | {f(('ours','fulldoc','ours_vs_fulldoc'))} | "
              f"{pct['qwen8b4_langchain']} | {pct['qwen8b4_ours']} | {f(('qwen8b4_ours','qwen8b4_langchain','qwen_ours_vs_langchain'))} |")
    (D / "by_type.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
