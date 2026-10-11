"""정답 포함 채점 — PREREG-2026-10-12-cellform.md (결과 전에 고정).

LLM 출력 전체(raw)에 정답이 들어 있으면 맞힘(Self-RAG·Mallen et al. 2023 식 inclusion).
- 정답이 yes/no: 출력에서 처음 나오는 낱말 yes/no 가 정답과 같으면 맞힘.
- 정답이 숫자: 출력의 숫자 토큰 중 하나라도 형식 허용 채점(format_match)으로 정답과 같으면 맞힘.
  정답이 공백으로 나뉜 숫자 여럿이면(예: "2006 2005") 하나하나가 모두 출력에 있어야 맞힘.
- 정답이 글자: 쉼표로 나눈 정답 조각이 모두 출력에 들어 있으면 맞힘(소문자, 영숫자 외 공백, 공백 하나로).
보조(민감도): 마지막 숫자 토큰 하나만 보는 채점, 출력의 숫자 토큰 수.
  PYTHONPATH=. .venv/bin/python results/cellform_20261012/score_inclusion.py <answer.jsonl> [...]
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from mh_arms import official_answers                              # noqa: E402
from rag_agent.eval.multihiertt_em import format_match            # noqa: E402

NUM = re.compile(r"\(?-?\$?\d[\d,]*(?:\.\d+)?%?\)?")
norm = lambda s: " ".join(re.sub(r"[^a-z0-9]", " ", str(s).lower()).split())


def gold_kind(g):
    g = str(g).strip().lower()
    if g in ("yes", "no"):
        return "yesno"
    return "num" if re.fullmatch(r"-?\$?[\d,]*\.?\d+%?", g.replace(" ", "")) else "text"


def nums(text):
    return [t.replace(",", "").replace("$", "") for t in NUM.findall(text or "")]


def included(raw, gold, arith):
    k = gold_kind(gold)
    if k == "yesno":
        m = re.search(r"\b(yes|no)\b", (raw or "").lower())
        return bool(m) and m.group(1) == str(gold).strip().lower()
    if k == "num":                                  # 정답 숫자가 여럿(예: "2006 2005")이면 모두 출력에 있어야 맞힘
        toks = nums(raw)
        return all(any(format_match(t, g, arith) for t in toks) for g in str(gold).split())
    parts = [norm(p) for p in str(gold).split(",") if norm(p)]
    return bool(parts) and all(p in norm(raw) for p in parts)


def last_num(raw, gold, arith):
    k = gold_kind(gold)
    if k != "num":
        return included(raw, gold, arith)
    t = nums(raw)
    return bool(t) and format_match(t[-1], gold, arith)


def self_check(G, rows):
    ok = sum(included(f"Based on the table, the answer is {G[r['query_id']]}.", G[r["query_id"]], r["kind"] == "arith") for r in rows)
    bad = sum(included("I don't know.", G[r["query_id"]], r["kind"] == "arith") for r in rows)
    return ok, bad, len(rows)


if __name__ == "__main__":
    G = official_answers("train")
    for p in sys.argv[1:]:
        rows = [json.loads(l) for l in open(p) if l.strip()]
        ok, bad, n = self_check(G, rows)
        inc = [included(r["raw"], G[r["query_id"]], r["kind"] == "arith") for r in rows]
        last = [last_num(r["raw"], G[r["query_id"]], r["kind"] == "arith") for r in rows]
        print(json.dumps({"file": p, "n": n, "self_check_gold_in_sentence": ok, "self_check_dont_know": bad,
                          "inclusion": sum(inc), "last_number": sum(last),
                          "lookup": sum(i for i, r in zip(inc, rows) if r["kind"] == "lookup"),
                          "arith": sum(i for i, r in zip(inc, rows) if r["kind"] == "arith"),
                          "numbers_per_output": round(sum(len(nums(r["raw"])) for r in rows) / n, 2)}, ensure_ascii=False))
