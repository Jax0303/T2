"""지시문 선택용 dev 입력 — PREREG-2026-10-15-prompt-sweep.md.
dev = MultiHiertt validation 332(`results/reader_format_20260927/mh_dev_pop332.jsonl`, 기존 dev A·C 레코드 그대로)
    + train 의 882 평가 문항 밖 lookup_m2+ 전부(풀이 예시로 쓴 문항 제외).
A = LangChain 청크 4개 markdown(기존 검색 레코드), C = 본 방법 점수 순 셀 문장을 같은 토큰 길이 T 까지(cellform build_inputs.py 의 [다] 규칙).
레코드에 공식 정답(answer)·분할(split)을 넣는다. 기존 882 C 레코드 재현을 같은 코드로 확인한다.
  PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python results/prompt_sweep_20261015/build_dev.py
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from mh_arms import build_tables, cell_id, load_population, official_answers, resolve_gold, layer   # noqa: E402
from retrieval_accuracy import build_corpus                                                        # noqa: E402
from transformers import AutoTokenizer                                                             # noqa: E402

OUT = ROOT / "results/prompt_sweep_20261015"
DEV332 = ROOT / "results/reader_format_20260927/mh_dev_pop332.jsonl"
DEV_A, DEV_C = ROOT / "results/cellform_dev_20261012/A_records.jsonl", ROOT / "results/cellform_dev_20261012/C_records.jsonl"
LC = ROOT / "results/mh_only_20260928/baselines/langchain_chunk/langchain_chunk_records.jsonl"
R100 = ROOT / "results/mh_only_20260928/retrieval_r100/mh_train_s3c_records.jsonl"
IDS882 = ROOT / "results/prf_20261015/train_ids882.json"
C882 = ROOT / "results/cellform_20261012/C_ours_budget_records.jsonl"
FEWSHOT = OUT / "fewshot_ids.json"

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
n_tok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])


def ours_rows(ids, split):
    """[다] 규칙(cellform build_inputs.py 그대로): 상위 100 기록 순서대로 셀 문장, T 직전까지."""
    lc = {r["query_id"]: r for r in map(json.loads, open(LC, encoding="utf-8")) if r["query_id"] in set(ids)}
    r100 = {r["query_id"]: r for r in map(json.loads, open(R100, encoding="utf-8")) if r["query_id"] in set(ids) and "doc" in r}
    assert set(lc) == set(r100) == set(ids), (len(lc), len(r100), len(ids))
    _q, docs, _ = load_population(split)
    tables, hdr = build_tables({u: docs[u] for u in ids}, "v3.3u", "none")
    texts, covers, _, unit_tids, _ = build_corpus("", sorted(tables), "s3c", "cell", {}, load=lambda t, _d: tables.get(t))
    sent = defaultdict(dict)
    for txt, cov, tid in zip(texts, covers, unit_tids):
        sent[tid.split("::")[0]][cell_id(next(iter(cov)), hdr).split("::", 1)[1]] = txt
    A, C = [], []
    for q in ids:
        L, R, S = lc[q], r100[q], sent[q]
        gold = set(L["gold_ids"])
        assert [S[u[0]] for u in R["doc"]["ranked_units"][:20]] == R["doc"]["context"], q
        base = {"query_id": q, "kind": L["kind"], "layer": L["layer"], "m": L["m"], "question": L["question"], "gold_ids": L["gold_ids"]}
        T = n_tok(L["context"])
        a_cells = set().union(*(set(u["cells"]) for u in L["ranked_units"][:4]))
        A.append({**base, "context": L["context"], "correct": int(gold <= a_cells), "cells_in_context": len(a_cells), "n_tok_ctx": T})
        sel = []
        for u in R["doc"]["ranked_units"]:
            cand = sel + [u[0]]
            if n_tok("\n".join(S[c] for c in cand)) > T:
                break
            sel = cand
        ctx = "\n".join(S[c] for c in sel)
        C.append({**base, "context": ctx, "correct": int(gold <= set(sel)), "cells_in_context": len(sel), "n_tok_ctx": n_tok(ctx)})
    return A, C


# 1) 기존 882 C 레코드 재현 확인(같은 코드 경로)
ids882 = json.load(open(IDS882))
_, C_chk = ours_rows(ids882, "train")
old = {r["query_id"]: r for r in map(json.loads, open(C882, encoding="utf-8"))}
same = sum(old[r["query_id"]]["context"] == r["context"] for r in C_chk)
print(f"[check] 882 본 방법 입력 재현: {same}/{len(C_chk)} 같음")
assert same == len(C_chk) == 882

# 2) train 추가분: 882 밖 lookup_m2+ 전부, 풀이 예시 제외
few = set(json.load(open(FEWSHOT)))
Gt = official_answers("train")
q, d, _ = load_population("train")
tables, hdr = build_tables(d, "v3.3u", "none")
live = {(tid, i, j) for tid, tab in tables.items() for i, row in enumerate(tab.table.data) for j, v in enumerate(row) if str(v).strip()}
qs = [x for x in resolve_gold(q, tables, hdr, live) if x["gold"] and not x["excluded"]]
extra = sorted(x["uid"] for x in qs if x["uid"] not in set(ids882) and x["uid"] not in few and layer(x) == "lookup_m2+")
print(f"[extra] train lookup_m2+ 882 밖 {sum(1 for x in qs if x['uid'] not in set(ids882) and layer(x) == 'lookup_m2+')} 중 예시 제외 → {len(extra)}")
A_ex, C_ex = ours_rows(extra, "train")

# 3) validation 332: 기존 dev 레코드
Gv = official_answers("validation")
ids332 = [json.loads(l)["query_id"] for l in open(DEV332) if l.strip()]
A_v = {r["query_id"]: r for r in map(json.loads, open(DEV_A, encoding="utf-8"))}
C_v = {r["query_id"]: r for r in map(json.loads, open(DEV_C, encoding="utf-8"))}
assert set(ids332) <= set(A_v) and set(ids332) <= set(C_v)

rows = {"A": [], "C": []}
for qid in ids332:
    for arm, src in (("A", A_v), ("C", C_v)):
        rows[arm].append({**src[qid], "split": "validation", "answer": Gv[qid]})
for arm, lst in (("A", A_ex), ("C", C_ex)):
    for r in lst:
        rows[arm].append({**r, "split": "train", "answer": Gt[r["query_id"]]})
for arm in rows:
    p = OUT / f"dev_{arm}_records.jsonl"
    with open(p, "x", encoding="utf-8") as f:
        for r in rows[arm]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rs = rows[arm]
    print(f"{p.name}: {len(rs)}건, 정답 셀 포함 {sum(r['correct'] for r in rs)}, 셀 평균 {sum(r['cells_in_context'] for r in rs)/len(rs):.1f}, "
          f"토큰 평균 {sum(r['n_tok_ctx'] for r in rs)/len(rs):.0f}, 층 {dict(Counter(r['layer'] for r in rs))}")
assert [r["query_id"] for r in rows["A"]] == [r["query_id"] for r in rows["C"]]
json.dump([r["query_id"] for r in rows["A"]], open(OUT / "dev_ids.json", "w"), indent=0)
