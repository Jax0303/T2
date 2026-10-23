"""2차 진단: (1) 마지막 줄 채점, (2) 머리글 경로 결함 빈도와 정확도, (3) 표 수/정답 셀 흩어짐, (4) 운 좋은 승리 점검."""
import csv, json, math, re, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path("/home/ugh/T2/rag-agent")
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "results/cellform_20261012"), str(ROOT / "results/score_fix_20261014")]
from huggingface_hub import hf_hub_download
from score_inclusion import gold_kind, NUM
from score_v2 import included_v2
from scipy.stats import binomtest
D = Path(__file__).resolve().parent
VAR = "R1X1F1B2D1C1E0"
FD = ROOT / "results/prompt_sweep_20261015/final_882"
path = hf_hub_download("yilunzhao/MultiHiertt", "multihiertt_data/train.json", repo_type="dataset", revision="f18473da528dede3d9ce2274366d9ee8102ea0fd")
OFF = {x["uid"]: x for x in json.load(open(path, encoding="utf-8"))}
J = {(r["query_id"], r["cond"]): int(r["judge"]) for r in csv.DictReader(open(FD / "judgments.csv", encoding="utf-8"))}
REC = {a: {r["query_id"]: r for r in map(json.loads, open(ROOT / f"results/cellform_20261012/{f}_records.jsonl", encoding="utf-8"))} for a, f in (("C", "C_ours_budget"), ("A", "A_chunk_md"))}
OUTS = {a: {r["query_id"]: r for r in map(json.loads, open(FD / a / f"{VAR}.jsonl", encoding="utf-8"))} for a in ("C", "A")}
sets = json.load(open(D / "sets.json"))
ids = sorted(OUTS["C"])

def yesno_first(raw):
    m = re.search(r"\b(yes|no)\b", (raw or "").lower()); return m.group(1) if m else None
def score_full(arm, q):
    r = OUTS[arm][q]; g = OFF[q]["qa"]["answer"]; arith = r["kind"] == "arith"
    s = int(included_v2(r["raw"], g, arith))
    if gold_kind(g) == "yesno" and not yesno_first(r["raw"]): s = J.get((q, f"{arm}:{VAR}"), 0)
    return s
def final_line(raw):
    m = list(re.finditer(r"final answer\s*[:：]\s*(.*)", raw or "", re.I))
    return m[-1].group(1).strip() if m else None
def score_final(arm, q):
    """마지막 'Final answer:' 줄만 채점(없으면 0). 예/아니오는 그 줄의 yes/no."""
    r = OUTS[arm][q]; g = OFF[q]["qa"]["answer"]; arith = r["kind"] == "arith"
    fl = final_line(r["raw"])
    if fl is None: return 0
    if gold_kind(g) == "yesno":
        w = yesno_first(fl); return int(w == str(g).strip().lower()) if w else 0
    return int(included_v2(fl, g, arith))

S = {arm: {q: (score_full(arm, q), score_final(arm, q)) for q in ids} for arm in ("C", "A")}
def mcn(a, b, qs, k):
    bb = sum(S[a][q][k] and not S[b][q][k] for q in qs); cc = sum(S[b][q][k] and not S[a][q][k] for q in qs)
    return bb, cc, (binomtest(min(bb, cc), bb + cc, 0.5).pvalue if bb + cc else 1.0)
print("== 882 전체 채점 (전체 출력 포함 / 마지막 줄만) ==")
for k, name in ((0, "전체 출력 포함(현행)"), (1, "마지막 Final answer 줄만")):
    c = sum(S["C"][q][k] for q in ids); a = sum(S["A"][q][k] for q in ids)
    bb, cc, p = mcn("C", "A", ids, k)
    print(f"  {name}: 본 {c} ({100*c/882:.1f}%) 대 LC {a} ({100*a/882:.1f}%)  b:c {bb}:{cc} p={p:.3f}")
    for kind in ("lookup", "arith"):
        qs = [q for q in ids if OUTS["C"][q]["kind"] == kind]
        bb, cc, p = mcn("C", "A", qs, k)
        print(f"     {kind}: 본 {sum(S['C'][q][k] for q in qs)} 대 LC {sum(S['A'][q][k] for q in qs)} b:c {bb}:{cc} p={p:.3f}")
# 전체 포함 맞힘이지만 마지막 줄은 틀린 건수(운)
for arm in ("C", "A"):
    lucky = [q for q in ids if S[arm][q][0] and not S[arm][q][1]]
    nofinal = [q for q in lucky if final_line(OUTS[arm][q]["raw"]) is None]
    print(f"  {arm}: 전체 포함 맞힘이지만 마지막 줄 틀림/없음 {len(lucky)} (Final answer 줄 없음 {len(nofinal)}, 그중 잘림 {sum(not OUTS[arm][q]['finished'] for q in nofinal)})")
# L/W 안에서
for name, qs in (("L(본 방법만 틀림)", sets["L"]), ("W(LC만 틀림)", sets["W"])):
    print(f"  {name}: LC 마지막 줄 맞힘 {sum(S['A'][q][1] for q in qs)}/{len(qs)}, 본 마지막 줄 맞힘 {sum(S['C'][q][1] for q in qs)}/{len(qs)}")

# (2) 머리글 경로 결함 탐지 (본 방법 입력 줄 기준)
PH = re.compile(r"(?:^|> )(?:column|row) \d+(?: >|:)")          # column N / row N 자리표시
FOOT = re.compile(r"\b(?:19|20)\d{2}\d\b|\b(?:19|20)\d{2} \d(?:, \d)?(?: >|:)")  # 20063, '2007 2, 3'
DASH = re.compile(r"> \$?—(?: >|:)")                                 # '$—' 머리글
def defects(ctx):
    lines = ctx.split("\n")
    d = Counter()
    d["placeholder"] = sum(bool(PH.search(l)) for l in lines)
    d["footnote_year"] = sum(bool(FOOT.search(l)) for l in lines)
    d["dash_header"] = sum(bool(DASH.search(l)) for l in lines)
    # 같은 경로(표 번호 제외)가 다른 표에 중복
    paths = Counter(re.sub(r"^Table \d+ > ", "", l.rsplit(": ", 1)[0]) for l in lines if ": " in l)
    tabs = Counter(re.match(r"^Table (\d+) > ", l).group(1) for l in lines if re.match(r"^Table \d+ > ", l))
    d["dup_path_across_tables"] = sum(v - 1 for v in paths.values() if v > 1) if len(tabs) > 1 else 0
    d["n_tables_tagged"] = len(tabs)
    return d, len(lines)
DEF = {q: defects(REC["C"][q]["context"]) for q in ids}
print("\n== 본 방법 입력의 머리글 경로 결함 (882 문항) ==")
for key in ("placeholder", "footnote_year", "dash_header", "dup_path_across_tables"):
    has = [q for q in ids if DEF[q][0][key] > 0]
    no = [q for q in ids if DEF[q][0][key] == 0]
    def acc(arm, qs): return f"{sum(S[arm][q][0] for q in qs)}/{len(qs)} ({100*sum(S[arm][q][0] for q in qs)/max(1,len(qs)):.1f}%)"
    bb, cc, p = mcn("C", "A", has, 0)
    print(f"  {key}: 있는 문항 {len(has)} — 본 {acc('C', has)} LC {acc('A', has)} b:c {bb}:{cc} p={p:.3f} | 없는 문항 {len(no)} — 본 {acc('C', no)} LC {acc('A', no)}")
    print(f"      L 에서 {sum(1 for q in sets['L'] if DEF[q][0][key] > 0)}/66, W 에서 {sum(1 for q in sets['W'] if DEF[q][0][key] > 0)}/54")
anyd = [q for q in ids if any(DEF[q][0][k] > 0 for k in ("placeholder", "footnote_year", "dash_header"))]
print(f"  자리표시/각주/대시 중 하나라도: {len(anyd)} 문항, 본 {sum(S['C'][q][0] for q in anyd)} LC {sum(S['A'][q][0] for q in anyd)} b:c {mcn('C','A',anyd,0)[:2]}; 없는 {882-len(anyd)}: 본 {sum(S['C'][q][0] for q in ids if q not in set(anyd))} LC {sum(S['A'][q][0] for q in ids if q not in set(anyd))} b:c {mcn('C','A',[q for q in ids if q not in set(anyd)],0)[:2]}")
# 자리표시 줄 비율
tot_lines = sum(DEF[q][1] for q in ids); ph_lines = sum(DEF[q][0]["placeholder"] for q in ids)
print(f"  전달 셀 줄 {tot_lines} 중 column N/row N 자리표시 포함 {ph_lines} ({100*ph_lines/tot_lines:.1f}%), 각주 연도 {sum(DEF[q][0]['footnote_year'] for q in ids)}, $— 머리글 {sum(DEF[q][0]['dash_header'] for q in ids)}")

# (3) 표 수와 정답 셀 흩어짐
def ntab_ctx(arm, q):
    ctx = REC[arm][q]["context"]
    if arm == "C":
        t = set(re.match(r"^Table (\d+) > ", l).group(1) for l in ctx.split("\n") if re.match(r"^Table \d+ > ", l))
        return len(t) if t else None   # 표 번호 없으면 알 수 없음(표가 하나거나 중복 없음)
    return ctx.count("\n\n|") + 1
tofloat = lambda s: (lambda t: (-float(t.strip("()").replace("$","").replace(",","").replace("%","")) if t.startswith("(") else float(t.strip("()").replace("$","").replace(",","").replace("%",""))) if re.fullmatch(r"\(?-?\$?[\d,]*\.?\d+%?\)?", t) else None)(s.strip())
def gold_pos(q):
    ctx = REC["C"][q]["context"].split("\n"); desc = OFF[q]["table_description"]
    pos = []
    for g in OFF[q]["qa"]["table_evidence"]:
        v = desc.get(g, "").strip().rstrip(".").rsplit(" is ", 1)[-1].strip(); fv = tofloat(v)
        idx = next((i for i, l in enumerate(ctx) if ": " in l and ((fv is not None and tofloat(l.rsplit(": ",1)[-1]) == fv) or (fv is None and l.rsplit(": ",1)[-1] == v))), None)
        pos.append(idx)
    return pos
print("\n== 정답 셀 2개 이상 산술(300): 본 방법 입력에서 정답 셀 위치 흩어짐(최대-최소 줄 간격) 대 정확도 ==")
m2 = [q for q in ids if OUTS["C"][q]["layer"] == "arith_m2+" and REC["C"][q]["correct"]]
spread = {}
for q in m2:
    p = [x for x in gold_pos(q) if x is not None]
    if len(p) >= 2: spread[q] = max(p) - min(p)
for lo, hi in ((0, 3), (4, 10), (11, 25), (26, 999)):
    qs = [q for q in spread if lo <= spread[q] <= hi]
    if qs: print(f"  간격 {lo}~{hi if hi<999 else ''}: {len(qs)} 문항, 본 {sum(S['C'][q][0] for q in qs)} ({100*sum(S['C'][q][0] for q in qs)/len(qs):.0f}%), LC 같은 문항 {sum(S['A'][q][0] for q in qs)} ({100*sum(S['A'][q][0] for q in qs)/len(qs):.0f}%)")
print("\n== 본 방법 입력에 'Table k' 표시가 붙은(표 간 같은 문장이 있는) 문항 ==")
tagged = [q for q in ids if ntab_ctx("C", q)]
untag = [q for q in ids if not ntab_ctx("C", q)]
for name, qs in (("표시 있음", tagged), ("표시 없음", untag)):
    bb, cc, p = mcn("C", "A", qs, 0)
    print(f"  {name}: {len(qs)} 문항, 본 {sum(S['C'][q][0] for q in qs)} ({100*sum(S['C'][q][0] for q in qs)/len(qs):.1f}%) LC {sum(S['A'][q][0] for q in qs)} ({100*sum(S['A'][q][0] for q in qs)/len(qs):.1f}%) b:c {bb}:{cc} p={p:.3f} | L {sum(q in qs for q in sets['L'])} W {sum(q in qs for q in sets['W'])}")
print(f"  LC 청크 4개가 걸친 표 수 평균 {sum(ntab_ctx('A',q) for q in ids)/882:.2f}; 문서 표 수 평균 {sum(len(OFF[q]['tables']) for q in ids)/882:.2f}")
# (4) 잘림과 출력 길이: 둘 다 끝난 문항에서의 차이, 본 방법이 더 긴 이유(열거 줄 수)
fin = [q for q in ids if OUTS["C"][q]["finished"] and OUTS["A"][q]["finished"]]
print(f"\n== 둘 다 1,536 안에 끝난 {len(fin)} 문항: 본 {sum(S['C'][q][0] for q in fin)} LC {sum(S['A'][q][0] for q in fin)} b:c {mcn('C','A',fin,0)[:2]}; 마지막 줄 채점 본 {sum(S['C'][q][1] for q in fin)} LC {sum(S['A'][q][1] for q in fin)} b:c {mcn('C','A',fin,1)[:2]}")
json.dump({q: {"C": S["C"][q], "A": S["A"][q]} for q in ids}, open(D / "scores2.json", "w"))
