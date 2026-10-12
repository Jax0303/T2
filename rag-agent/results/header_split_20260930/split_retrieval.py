"""원고 3.1절 추가 문장 근거 — 머리글 분리 문항과 아닌 문항의 LangChain 청크 상위 1개 재현율 비교.

header_split.py 와 같은 기록·같은 대리 기준('any'). 정답 셀이 1개인 문항만 센다(여러 셀이면 셀이 여러 청크에 흩어져 비교가 섞임).
문서 청크 수별로도 나눠 센다(분리 문항이 청크 많은 문서에 몰려 있어서). 새 실행 없음. .venv 의 scipy 사용.
"""
import json
from collections import defaultdict
from pathlib import Path
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[2]
REC = ROOT / "results/mh_only_20260928/baselines/langchain_chunk/langchain_chunk_records.jsonl"
parse = lambda c: tuple(int(v) for v in c.split("-"))          # "표-행-열"
cnt = defaultdict(lambda: [0, 0])                                # (구간, 분리여부) -> [문항, 상위 1개에 정답]
for l in open(REC, encoding="utf-8"):
    r = json.loads(l)
    if len(r["gold_ids"]) != 1:
        continue
    chunks = [set(u["cells"]) for u in r["ranked_units"]]
    allc = set().union(*chunks)
    rows = defaultdict(set)
    for c in allc:
        rows[parse(c)[0]].add(parse(c)[1])
    first = {t: {c for c in allc if parse(c)[0] == t and parse(c)[1] == min(v)} for t, v in rows.items()}
    g = r["gold_ids"][0]
    sep = not any(ch & first[parse(g)[0]] for ch in chunks if g in ch)
    nc = len(chunks)
    for b in ("all", "<=3" if nc <= 3 else "4-5" if nc <= 5 else "6+"):
        s = cnt[(b, sep)]
        s[0] += 1
        s[1] += g in chunks[0]
out = {}
for b in ("all", "<=3", "4-5", "6+"):
    (n1, h1), (n0, h0) = cnt[(b, True)], cnt[(b, False)]
    out[b] = {"sep": [h1, n1, round(100 * h1 / n1, 1) if n1 else None],
              "ok": [h0, n0, round(100 * h0 / n0, 1) if n0 else None],
              "fisher_p": fisher_exact([[h1, n1 - h1], [h0, n0 - h0]]).pvalue}
(Path(__file__).parent / "split_retrieval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False))
