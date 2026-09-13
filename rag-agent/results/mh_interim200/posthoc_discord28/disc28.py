"""interim200: 본 방법(cell_hv2) 대 chunk_hv2 답변 EM 불일치 28건 사후 분석. 점수·설정은 바꾸지 않는다.

문맥 문자열을 같은 build_corpus 인자로 다시 만든 단위 텍스트에 대응시켜 셀 좌표로 되돌리고,
레코드의 cells_in_context·doc.correct·context_sha256 과 일치하는지 확인한 뒤 센다.
"""
import json, re, sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

ROOT = Path("/home/user/T2-1/rag-agent")
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from mh_arms import build_tables, load_population, resolve_gold, NO_TITLE  # noqa
from retrieval_accuracy import build_corpus  # noqa
from rag_agent.eval.artifacts import digest, file_digest  # noqa
from rag_agent.eval.multihiertt_em import str_to_num  # noqa

OUT = Path(__file__).parent
H, D = ROOT / "results/mh_interim200", ROOT / "results/mh_arms"
ANS = {"ours": H / "mh_cell_hv2_answer_doc_qwen3_8b_cot_200.jsonl",
       "chunk": H / "mh_chunk_hv2_answer_doc_qwen3_8b_cot_200.jsonl"}
REC = {"ours": D / "mh_cell_hv2_records.jsonl", "chunk": D / "mh_chunk_hv2_records.jsonl"}
UNIT = {"ours": "cell", "chunk": "chunk"}   # 두 레그 요약 arguments: template s3c, chunk_chars 1000, v2, label none

load = lambda p: {r["query_id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}
ans = {k: load(p) for k, p in ANS.items()}
ids = sorted(i for i in ans["ours"] if ans["ours"][i]["answer_correct"] != ans["chunk"][i]["answer_correct"])
assert set(ans["ours"]) == set(ans["chunk"]) and len(ans["ours"]) == 200
want = set(ids)
rec = {k: {r["query_id"]: r for r in map(json.loads, open(p, encoding="utf-8")) if r["query_id"] in want}
       for k, p in REC.items()}

queries, docs, _ = load_population("train")
queries = [q for q in queries if q["uid"] in want]
progs = {}
from datasets import load_dataset  # noqa
for row in load_dataset("bevaya/MultiHiertt", split="train"):
    if row["uid"] in want:
        progs[row["uid"]] = row.get("program") or ""
docs = {u: docs[u] for u in want}
tables, hdr = build_tables(docs, "v2", "none")
live = {(tid, i, j) for tid, tab in tables.items() for i, row in enumerate(tab.table.data)
        for j, v in enumerate(row) if str(v).strip()}
gold = {q["uid"]: q["gold"] for q in resolve_gold(queries, tables, hdr, live)}

NUM = re.compile(r"\(?-?\$?\d[\d,]*(?:\.\d+)?%?\)?")


def nums(text):
    out = []
    for m in NUM.findall(text or ""):
        v = str_to_num(m.strip("()"))
        if v != "n/a":
            out.append(abs(v))
    return out


def has_num(v, pool):
    return any(abs(v - x) <= max(abs(v) / 1000, 0.01) for x in pool)


def near_miss(pred, answer):
    """채점과 무관한 진단: 표기·자릿수 차이로 설명되는 오답 후보."""
    p, g = str_to_num(pred), str_to_num(answer)
    if p == "n/a" or g == "n/a":
        return None
    if g == 0:
        return None
    for name, x in (("rel<=1%", p), ("x100", p / 100), ("/100", p * 100)):
        if abs(x - g) <= abs(g) * 0.01:
            return name
    return None


rows, checks = [], Counter()
for i in ids:
    uid_tabs = sorted(t for t in tables if t.split("::")[0] == i)
    r = {"query_id": i, "winner": "ours" if ans["ours"][i]["answer_correct"] else "chunk",
         "layer": ans["ours"][i]["layer"], "kind": ans["ours"][i]["kind"], "m": ans["ours"][i]["m"],
         "n_gold_tables": len({c[0] for c in gold[i]}),
         "ops": sorted(set(re.findall(r"([a-z_]+)\(", progs[i]))), "program": progs[i],
         "question": ans["ours"][i]["question"], "answer": ans["ours"][i]["answer"],
         "gold_cells": [{"cell": list(c), "row_path": tables[c[0]].table.row_path(c[1]),
                         "col_path": tables[c[0]].table.col_path(c[2]),
                         "value": tables[c[0]].table.data[c[1]][c[2]]} for c in sorted(gold[i])]}
    assert len(gold[i]) == r["m"], i
    for k in ("ours", "chunk"):
        a, rc = ans[k][i], rec[k][i]["doc"]
        ctx = rc["context"]
        assert digest(ctx) == a["context_sha256"], (k, i)
        texts, covers, _, utids, _ = build_corpus("", uid_tabs, "s3c", UNIT[k], {}, 1000, "leaf",
                                                 "sentence", 200, None,
                                                 load=lambda tid, _d: tables.get(tid))
        by_text = {}
        for t, c in zip(texts, covers):
            by_text.setdefault(t, []).append(c)
        unmatched = sum(1 for t in ctx if t not in by_text)
        ambiguous = sum(1 for t in ctx if len(by_text.get(t, [])) > 1)
        # 같은 문장이 여러 셀이면(경로+값 충돌) 아직 안 쓴 후보부터 배정한다. 후보가 등장 횟수보다
        # 많으면 어느 셀인지 문맥만으로는 정해지지 않는다 — gold 포함은 레코드 판정과 대조한다.
        used, covs = Counter(), []
        for t in ctx:
            if t in by_text:
                covs.append(by_text[t][used[t] % len(by_text[t])])
                used[t] += 1
        undetermined = sum(1 for t in used if len(by_text[t]) > used[t])
        got = frozenset().union(*covs) if covs else frozenset()
        ok = (unmatched == 0 and len(got) == rc["cells_in_context"]
              and int(gold[i] <= got) == rc["correct"] == a["retrieval_correct"])
        checks[f"{k}_reconstruct_ok"] += ok
        checks[f"{k}_ambiguous_units"] += ambiguous
        lines = [l for t in ctx for l in t.split("\n") if l.strip()]
        lc = Counter(lines)
        # 본 방법만: 같은 경로(': ' 앞)가 다른 값으로 두 번 이상 나오는 문장 — 리더가 셀을 구별할 수 없는 자리
        path_col = gold_path_col = None
        if k == "ours":
            paths = Counter(t.rsplit(": ", 1)[0] for t in ctx)
            path_col = sum(n for n in paths.values() if n > 1)
            cell_text = {next(iter(c)): t for t, c in zip(texts, covers)}
            gold_path_col = sum(1 for c in gold[i] if c in got and paths[cell_text[c].rsplit(": ", 1)[0]] > 1)
        gold_rows = {(c[0], c[1]) for c in gold[i]}
        gold_cols = {(c[0], c[2]) for c in gold[i]}
        row_live = {c for c in live if (c[0], c[1]) in gold_rows}
        col_live = {c for c in live if (c[0], c[2]) in gold_cols}
        pool = nums(a["raw"])
        r[k] = {"em": a["answer_correct"], "retrieved_all_gold": a["retrieval_correct"],
                "gold_in_ctx": len(gold[i] & got), "pred": a["pred"], "marker_found": a["marker_found"],
                "n_tok": a["n_tok"], "units": len(ctx), "cells": len(got),
                "cell_repeats": sum(len(c) for c in covs) - len(got),
                "dup_lines": sum(n - 1 for n in lc.values() if n > 1),
                "dup_line_chars": sum((n - 1) * len(l) for l, n in lc.items() if n > 1),
                "tables_in_ctx": len({c[0] for c in got}),
                "gold_row_cells_share": round(len(got & row_live) / len(row_live), 3) if row_live else None,
                "gold_col_cells_share": round(len(got & col_live) / len(col_live), 3) if col_live else None,
                "gold_values_in_output": sum(has_num(abs(v), pool) for v in
                                             (str_to_num(g["value"]) for g in r["gold_cells"]) if v != "n/a"),
                "gold_values_numeric": sum(str_to_num(g["value"]) != "n/a" for g in r["gold_cells"]),
                "answer_in_output": (has_num(abs(str_to_num(r["answer"])), pool)
                                     if str_to_num(r["answer"]) != "n/a" else None),
                "near_miss": None if a["answer_correct"] else near_miss(a["pred"], r["answer"]),
                "path_collision_lines": path_col, "gold_path_collision": gold_path_col,
                "unmatched_units": unmatched, "ambiguous_units": ambiguous,
                "undetermined_texts": undetermined, "reconstruct_ok": ok}
    rows.append(r)

# 전체 200 의 입력 토큰·셀·마커 (기준선)
base = {k: {"n_tok_mean": round(mean(x["n_tok"] for x in v.values()), 1),
            "n_tok_median": median(x["n_tok"] for x in v.values()),
            "cells_mean": round(mean(x["cells_in_context"] for x in v.values()), 1),
            "marker_missing": sum(not x["marker_found"] for x in v.values())} for k, v in ans.items()}
summary = {"analysis": "post hoc, exploratory; scores/settings unchanged",
           "inputs_sha256": {p.name: file_digest(p) for p in [*ANS.values(), *REC.values()]},
           "n_discordant": len(ids), "wins": Counter(r["winner"] for r in rows),
           "checks": checks, "all200": base}
(OUT / "disc28.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
(OUT / "disc28.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

# 사례 파일: 원표(gold 표시) + 두 문맥 + 두 출력
cases = OUT / "cases"
cases.mkdir(exist_ok=True)
for r in rows:
    i = r["query_id"]
    s = [f"# {i} winner={r['winner']} {r['layer']} ops={r['ops']}\nQ: {r['question']}\nA: {r['answer']}\nprogram: {r['program']}\n"]
    for g in r["gold_cells"]:
        s.append(f"gold {g['cell']} {g['row_path']} / {g['col_path']} = {g['value']}")
    marks = {}
    for tid, ci, cj in gold[i]:
        t = tables[tid].table
        marks.setdefault(tid, set()).add((ci + t.nhr, cj + t.nhc))
    for tid in sorted(marks):
        t = tables[tid].table
        s.append(f"\n## 원표 {tid} (nhr={t.nhr}, nhc={t.nhc})")
        for ri, row in enumerate(t.grid):
            s.append("| " + " | ".join((f"**{v}**" if (ri, cj) in marks[tid] else str(v)) for cj, v in enumerate(row)) + " |")
    for k in ("ours", "chunk"):
        x = r[k]
        s.append(f"\n## {k} em={x['em']} retrieved={x['retrieved_all_gold']} gold_in_ctx={x['gold_in_ctx']}/{r['m']} "
                 f"tok={x['n_tok']} cells={x['cells']} marker={x['marker_found']} pred={x['pred']!r}")
        s.append("### context\n" + "\n".join(rec[k][i]["doc"]["context"]))
        s.append("### raw\n" + ans[k][i]["raw"])
    (cases / f"{i}.md").write_text("\n".join(s), encoding="utf-8")

print(json.dumps(summary, ensure_ascii=False, indent=1))
