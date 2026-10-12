"""dev 332 입력 후보 — PREREG-2026-10-12-cellform-dev.md. 모든 조건의 토큰 예산 T = 문항별 LangChain 청크 4개 원문 토큰 수.
  A    LangChain 청크 4개(markdown)                         — 기준
  C    본 방법: 점수 순 셀 문장, 줄바꿈, T 직전까지          — 현재 설정
  K1   C 와 같은 셀, 순서만 (표, 행, 열)                    — 같은 행 모으기
  K2   점수 순으로 셀을 고르되 고른 셀의 행 나머지 셀을 먼저 채움(행 첫 등장 순, 행 안은 열 순), T 직전까지 — 행 채우기
  K3   점수 순으로 셀을 고르고 같은 행 셀을 한 줄로: "행 경로 > 열 경로1: 값1 | 열 경로2: 값2 …", 행은 첫 등장 순, T 직전까지 — 행 한 줄 묶기
  K4   C 와 같은 규칙, 단 경로에서 인위 식별자 마디(Table k, row N, column N)를 뺀 문장으로 T 직전까지 — 식별자 빼기
  PYTHONPATH=. .venv/bin/python results/cellform_dev_20261012/build_candidates.py
"""
import json, re, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from mh_arms import build_tables, cell_id, load_population          # noqa: E402
from retrieval_accuracy import build_corpus                          # noqa: E402
from rag_agent.serialization.caption import caption_sentence        # noqa: E402
from rag_agent.serialization.templates import STRUCTURAL_COMPACT, join_path  # noqa: E402
from transformers import AutoTokenizer                               # noqa: E402

D = ROOT / "results/cellform_dev_20261012"
IDS = ROOT / "results/reader_format_20260927/mh_dev_pop332.jsonl"
R100 = ROOT / "results/mh_only_20260928/dev_k/dev_s3c_k100_records.jsonl"
ids = [json.loads(l)["query_id"] for l in open(IDS) if l.strip()]
lc = {r["query_id"]: r for r in map(json.loads, open(D / "lc_retrieve/langchain_chunk_records.jsonl"))}
r100 = {r["query_id"]: r for r in map(json.loads, open(R100)) if "doc" in r}
assert set(ids) <= set(lc) and set(ids) <= set(r100)

_q, docs, _ = load_population("validation")
tables, hdr = build_tables({u: docs[u] for u in ids}, "v3.3u", "none")
texts, covers, _, unit_tids, _ = build_corpus("", sorted(tables), "s3c", "cell", {}, load=lambda t, _d: tables.get(t))
sent, coord = defaultdict(dict), defaultdict(dict)
for txt, cov, tid in zip(texts, covers, unit_tids):
    c = next(iter(cov)); s = cell_id(c, hdr).split("::", 1)[1]
    sent[tid.split("::")[0]][s] = txt; coord[tid.split("::")[0]][s] = c

def paths(q, s):
    tid, i, j = coord[q][s]; t = tables[tid].table
    rp, cp, v = list(t.row_path(i)), list(t.col_path(j)), t.data[i][j]
    assert caption_sentence("", rp, cp, value=v, template=STRUCTURAL_COMPACT) == sent[q][s], (q, s)   # 경로로 다시 만든 문장 = 색인 문장
    return rp, cp, v

ART = re.compile(r"^(Table \d+|row \d+|column \d+)$")
def strip_art(rp, cp, v):
    keep = lambda p: [x for x in p if not ART.match(str(x).strip())]
    return caption_sentence("", keep(rp), keep(cp), value=v, template=STRUCTURAL_COMPACT)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
n_tok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
key = lambda c: tuple(int(x) for x in c.split("-"))

def greedy(order, render, T):
    """order 순서로 셀을 하나씩 더해 render(골라진 셀) 의 토큰이 T 를 넘기 직전에 멈춘다."""
    sel = []
    for c in order:
        if c in sel:
            continue
        if n_tok(render(sel + [c])) > T:
            break
        sel.append(c)
    return sel, render(sel)

def k3_render(q, cells):
    rows, first = defaultdict(list), []
    for c in cells:
        tid, i, _ = coord[q][c]
        if (tid, i) not in rows:
            first.append((tid, i))
        rows[(tid, i)].append(c)
    lines = []
    for r in first:
        cs = sorted(rows[r], key=key)
        rp = paths(q, cs[0])[0]
        parts = [f"{join_path(paths(q, c)[1])}: {paths(q, c)[2]}" if paths(q, c)[1] else f"{paths(q, c)[2]}" for c in cs]
        head = join_path(rp)
        lines.append((head + " > " if head else "") + " | ".join(parts))
    return "\n".join(lines)

out = defaultdict(list)
for q in ids:
    L, R, S = lc[q], r100[q], sent[q]
    rank = [u[0] for u in R["doc"]["ranked_units"]]
    for s in rank:
        paths(q, s)
    T = n_tok(L["context"])
    gold = set(R["gold_ids"])
    base = {k: L[k] for k in ("query_id", "kind", "layer", "m", "question", "gold_ids")}
    a_cells = set().union(*(set(u["cells"]) for u in L["ranked_units"][:4]))
    cand = {"A": (sorted(a_cells), L["context"])}
    plain = lambda cs: "\n".join(S[c] for c in cs)
    cand["C"] = greedy(rank, plain, T)
    cand["K1"] = (cand["C"][0], plain(sorted(cand["C"][0], key=key)))
    # K2: 행 채우기 — 다음 순위 셀의 행 전체(비어 있지 않은 셀, 열 순)를 한 묶음으로 더한다
    order2 = []
    for c in rank:
        tid, i, _ = coord[q][c]
        row = sorted((s for s, (t2, i2, _j) in coord[q].items() if t2 == tid and i2 == i), key=key)
        order2 += [c] + [s for s in row if s != c]
    def k2_render(cs):
        rows, first = defaultdict(list), []
        for c in cs:
            r = coord[q][c][:2]
            if r not in rows:
                first.append(r)
            rows[r].append(c)
        return "\n".join(S[c] for r in first for c in sorted(rows[r], key=key))
    cand["K2"] = greedy(order2, k2_render, T)
    cand["K3"] = greedy(rank, lambda cs: k3_render(q, cs), T)
    cand["K4"] = greedy(rank, lambda cs: "\n".join(strip_art(*paths(q, c)) for c in cs), T)
    for name, (cells, ctx) in cand.items():
        out[name].append({**base, "context": ctx, "correct": int(gold <= set(cells)), "cells_in_context": len(cells), "n_tok_ctx": n_tok(ctx)})

for name, rs in out.items():
    with open(D / f"{name}_records.jsonl", "x") as f:
        for r in rs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(rs)
    print(f"{name}: 정답 셀 포함 {sum(r['correct'] for r in rs)}/{n}, 셀 평균 {sum(r['cells_in_context'] for r in rs)/n:.1f}, "
          f"토큰 평균 {sum(r['n_tok_ctx'] for r in rs)/n:.0f}")
ex = ids[0]
for name in ("K3", "K4"):
    print(f"--- {name} 예시\n" + next(r for r in out[name] if r["query_id"] == ex)["context"][:500])
