"""원고 2026-09-30 개정(200문항·형식 허용 채점)에 쓰는 수치 중 rescore.json 에 없는 것. 새 생성 없음."""
import csv, json, sys
from collections import Counter
from math import comb
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from huggingface_hub import hf_hub_download
from mh_arms import OFFICIAL_REPO, OFFICIAL_REV, official_answers
from rag_agent.eval.multihiertt_em import format_match

R = ROOT / "results"
gold = official_answers("train")
prog = {d["uid"]: d["qa"].get("program", "") for d in json.loads(Path(hf_hub_download(
    OFFICIAL_REPO, "multihiertt_data/train.json", repo_type="dataset", revision=OFFICIAL_REV)).read_text())}
ids = [json.loads(l)["query_id"] for l in open(R / "mh_only_20260928/baselines/ids200.jsonl") if l.strip()]
L = lambda p: {r["query_id"]: r for r in map(json.loads, open(R / p))}
ok = lambda r: format_match(r["pred"], gold[r["query_id"]], r["kind"] == "arith")
P = lambda a, b: min(1.0, 2 * sum(comb(a + b, i) for i in range(min(a, b) + 1)) / 2 ** (a + b)) if a + b else 1.0
o, v, g = L("mh_arms/cap300_20260924/cell_uniq.jsonl"), L("header_ablation_20260929/mh_value_answer.jsonl"), L("gold_only_20260929/mh_gold_cot384.jsonl")
out = {}
for name, d in (("ours", o), ("header_removed", v), ("gold_only", g)):
    grp = {}
    for q in ids:
        x = grp.setdefault(o[q]["layer"], [0, 0]); x[0] += ok(d[q]); x[1] += 1
    out[name] = {"groups": grp, "total": sum(ok(d[q]) for q in ids)}
for name, d in (("header_removed", v), ("gold_only", g)):
    a = sum(ok(o[q]) and not ok(d[q]) for q in ids); b = sum(ok(d[q]) and not ok(o[q]) for q in ids)
    out[name]["vs_ours"] = [a, b, P(a, b)]
A = {r["query_id"] for r in csv.DictReader(open(R / "mh_lookup1_evidence_20260928/labels.csv")) if r["label"] in ("S", "C", "R")}
import re
B = {q for q in ids if re.search(r"subtract\([^)]*const_100", str(prog.get(q, "")))}  # PREREG-2026-09-29-evidence-groups B 정의
out["lookup1_not_determined_200"] = [len(A & set(ids)), sum(o[q]["layer"] == "lookup_m1" for q in ids)]
out["const100_arith_200"] = len(B)
# LangChain 상위 4개 청크 재현율(200)
lc = L("mh_only_20260928/baselines/langchain_chunk/langchain_chunk_records.jsonl")
rec4 = []
for q in ids:
    r = lc[q]; G = set(r["gold_ids"]); units = [set(u["cells"]) for u in r["ranked_units"][:4]]
    rec4.append(len(G & set().union(*units)) / len(G))
out["langchain_R@4_200"] = round(100 * sum(rec4) / len(rec4), 1)
# 오답 원인(200, 형식 허용). 'R@20<100%' = 상위 20개 셀에 공식 근거 셀이 다 들지 않음
c = Counter()
for q in ids:
    r = o[q]
    if ok(r): continue
    if not r.get("marker_found", True): k = "출력 잘림"
    elif q in A: k = "비교·순위형 조회"
    elif q in B: k = "시작값 100 상수"
    elif not r["retrieval_correct"]: k = "R@20<100%, 근거 셀만 주면 맞힘" if ok(g[q]) else "R@20<100%, 근거 셀만 줘도 틀림"
    else: k = "다른 셀 탓(근거 셀만 주면 맞힘)" if ok(g[q]) else "LLM 오답(근거 셀만 줘도 틀림)"
    c[k] += 1
out["wrong_causes_200"] = dict(c.most_common()); out["wrong_200"] = sum(c.values())
(Path(__file__).parent / "thesis_numbers.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
