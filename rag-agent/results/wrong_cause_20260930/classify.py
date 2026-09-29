"""본 방법 882 답변 오답 474건 원인 분류 (2026-09-30, 새 실행 없음, 사후 분석).
위에서부터 처음 맞는 범주 하나에 넣는다. 입력: 본 방법 답변(cell_uniq), 공식 근거 셀만 입력 답변(gold_only),
비교·순위형 조회 판정(labels.csv S/C/R), 공식 계산식의 const_100."""
import csv, json, re, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from huggingface_hub import hf_hub_download
from mh_arms import OFFICIAL_REPO, OFFICIAL_REV

prog = {d["uid"]: d["qa"].get("program", "") for d in json.loads(Path(hf_hub_download(
    OFFICIAL_REPO, "multihiertt_data/train.json", repo_type="dataset", revision=OFFICIAL_REV)).read_text())}
L = lambda p: {r["query_id"]: r for r in map(json.loads, open(ROOT / p))}
o = L("results/mh_arms/cap300_20260924/cell_uniq.jsonl")
g = L("results/gold_only_20260929/mh_gold_cot384.jsonl")
A = {r["query_id"] for r in csv.DictReader(open(ROOT / "results/mh_lookup1_evidence_20260928/labels.csv")) if r["label"] in ("S", "C", "R")}
B = {q for q in o if "const_100" in str(prog.get(q, "")) and o[q]["kind"] == "arith"}
MON = r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?"
nums = lambda s: sorted(float(x) for x in re.findall(r"-?\d+\.?\d*", re.sub(MON, "", str(s).lower()).replace(",", "").replace("$", "").replace("%", "")))
words = lambda s: re.sub(r"[^a-z0-9. ]", "", re.sub(MON, "", str(s).lower()).replace("level", "")).split()
ok = lambda r: r["answer_correct"] == 1


def cause(q, r):
    ar, pn, an = r["kind"] == "arith", nums(r["pred"]), nums(r["answer"])
    if not r.get("marker_found", True): return "출력 잘림(384토큰)"
    if r["answer_correct_docmath"]: return "표기 차이(100배 등, DocMath 허용 시 정답)"
    if ar and pn and an and an[-1] != 0 and abs(pn[-1] - an[-1]) <= abs(an[-1]) * 0.01: return "산술 반올림(1% 이내)"
    if not ar and ((pn and pn == an) or (not an and words(r["pred"]) == words(r["answer"]))): return "조회 형식 차이(값 같음)"
    if ar and pn and an and abs(pn[-1]) == abs(an[-1]): return "부호만 다름"
    if q in A: return "비교·순위형 조회(근거 셀 하나로 답 안 정해짐)"
    if q in B: return "시작값 100 상수 산술"
    if not r["retrieval_correct"]: return "검색 실패(근거 셀만 주면 맞힘)" if ok(g[q]) else "검색 실패 + 근거 셀만 줘도 틀림"
    return "잡음 셀(근거 셀만 주면 맞힘)" if ok(g[q]) else "LLM 오답(근거 셀만 줘도 틀림)"


c = Counter(cause(q, r) for q, r in o.items() if not ok(r))
out = {"n": len(o), "correct": sum(ok(r) for r in o.values()), "wrong": sum(c.values()), "causes": dict(c.most_common())}
(Path(__file__).parent / "causes.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
