"""본 방법(C) 대 LangChain(A) P* 882 — 불일치 쌍 진단 (사후 분석)."""
import csv, json, math, re, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/home/ugh/T2/rag-agent")
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "results/cellform_20261012"), str(ROOT / "results/score_fix_20261014"), str(ROOT / "results/gpt4o_20261013")]
from huggingface_hub import hf_hub_download
from score_inclusion import gold_kind, NUM
from score_v2 import included_v2
from rag_agent.eval.multihiertt_em import format_match
import by_type

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
VAR = sys.argv[2] if len(sys.argv) > 2 else "R1X1F1B2D1C1E0"
FD = ROOT / "results/prompt_sweep_20261015/final_882"
path = hf_hub_download("yilunzhao/MultiHiertt", "multihiertt_data/train.json", repo_type="dataset", revision="f18473da528dede3d9ce2274366d9ee8102ea0fd")
OFF = {x["uid"]: x for x in json.load(open(path, encoding="utf-8"))}
J = {(r["query_id"], r["cond"]): int(r["judge"]) for r in csv.DictReader(open(FD / "judgments.csv", encoding="utf-8"))}
REC = {a: {r["query_id"]: r for r in map(json.loads, open(ROOT / f"results/cellform_20261012/{f}_records.jsonl", encoding="utf-8"))}
       for a, f in (("C", "C_ours_budget"), ("A", "A_chunk_md"))}

def tofloat(tok):
    s = tok.strip()
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace("$", "").replace(",", "").replace("%", "")
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v

def ctx_nums(text):
    out = []
    for t in NUM.findall(text or ""):
        v = tofloat(t)
        if v is not None:
            out.append(v)
    return out

def close(a, b):
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)

def operands(program):
    lits = []
    for m in re.finditer(r"[a-z_]+\(([^)]*)\)", program or ""):
        for a in m.group(1).split(","):
            a = a.strip().replace(",", "")
            if re.fullmatch(r"-?\d+(\.\d+)?", a):
                lits.append(float(a))
            elif a.startswith("const_"):
                pass
    return lits

def operand_cov(lits, nums):
    found = 0
    for x in lits:
        if any(close(abs(x), abs(n)) for n in nums):
            found += 1
    return found, len(lits)

def final_answer(raw):
    m = list(re.finditer(r"Final answer:\s*(.*)", raw or ""))
    return m[-1].group(1).strip() if m else None

def score(r, cond, g, arith):
    s = int(included_v2(r["raw"], g, arith))
    if gold_kind(g) == "yesno" and not re.search(r"\b(yes|no)\b", r["raw"].lower()):
        s = J.get((r["query_id"], cond), 0)
    return s

def refusal(raw):
    t = (raw or "").lower()
    return bool(re.search(r"don't know|do not know|not provided|cannot be determined|can't be determined|not available in the context|not present in the context|no information|insufficient", t))

def desc_cells(uid):
    """{cell_id: value_float or text} from table_description."""
    d = {}
    for k, s in OFF[uid]["table_description"].items():
        v = s.strip().rstrip(".").strip().rsplit(" is ", 1)[-1].strip()
        d[k] = v
    return d

def cell_matches(uid, val):
    """cells in the doc whose value equals val (float)."""
    out = []
    for k, v in desc_cells(uid).items():
        fv = tofloat(v)
        if fv is not None and val is not None and (close(abs(fv), abs(val))):
            out.append(k)
    return out

def rel(cid, gold_ids):
    t, r, c = cid.split("-")
    kinds = set()
    for g in gold_ids:
        gt, gr, gc = g.split("-")
        if g == cid: kinds.add("gold")
        elif gt == t and gr == r: kinds.add("same_row")
        elif gt == t and gc == c: kinds.add("same_col")
        elif gt == t: kinds.add("same_table")
        else: kinds.add("other_table")
    for k in ("gold", "same_row", "same_col", "same_table", "other_table"):
        if k in kinds: return k
    return "none"

def gold_positions_ours(ctx, uid, gold_ids):
    """line index of each gold cell in ours context (by matching value+path end); approximate via value match."""
    lines = ctx.split("\n")
    dc = desc_cells(uid)
    pos = []
    for g in gold_ids:
        gv = tofloat(dc.get(g, ""))
        idx = None
        for i, ln in enumerate(lines):
            tail = ln.rsplit(": ", 1)[-1] if ": " in ln else ""
            tv = tofloat(tail)
            if gv is not None and tv is not None and close(gv, tv):
                idx = i; break
            if gv is None and dc.get(g, "") and dc.get(g, "") == tail:
                idx = i; break
        pos.append(idx)
    return pos, len(lines)

rows = {}
for arm in ("C", "A"):
    for r in map(json.loads, open(FD / arm / f"{VAR}.jsonl", encoding="utf-8")):
        q = r["query_id"]; e = OFF[q]; g = e["qa"]["answer"]; arith = r["kind"] == "arith"
        rec = REC[arm][q]
        lits = operands(e["qa"]["program"]) if arith else []
        cn = ctx_nums(rec["context"])
        fa = final_answer(r["raw"])
        fav = tofloat(fa) if fa else None
        d = {"inc": score(r, f"{arm}:{VAR}", g, arith), "gold_in": rec["correct"], "fin": int(r["finished"]), "n_out": r["n_tok_out"],
             "ref": int(refusal(r["raw"])), "op_found": operand_cov(lits, cn), "fa": fa, "fav": fav,
             "cells": rec["cells_in_context"], "n_tok_ctx": rec["n_tok_ctx"], "raw": r["raw"], "ctx": rec["context"]}
        # which cell did it answer (lookup/value or arith wrong)
        if fav is not None:
            cm = cell_matches(q, fav)
            d["fa_cell_rel"] = Counter(rel(c, e["qa"]["table_evidence"]) for c in cm).most_common(1)[0][0] if cm else "not_a_cell"
            d["fa_cells"] = cm[:4]
        else:
            d["fa_cell_rel"] = "no_final" if fa is None else "text"
            d["fa_cells"] = []
        if arm == "C":
            d["gold_pos"], d["n_lines"] = gold_positions_ours(rec["context"], q, e["qa"]["table_evidence"])
        rows.setdefault(q, {})[arm] = d

ids = sorted(rows)
for q in ids:
    e = OFF[q]; rec = REC["C"][q]
    rows[q]["meta"] = {"kind": rec["kind"], "layer": rec["layer"], "m": rec["m"], "question": rec["question"], "gold": e["qa"]["answer"],
                       "program": e["qa"]["program"], "evidence": e["qa"]["table_evidence"], "n_tables": len(e["tables"]),
                       "qtype": by_type.qtype({"query_id": q, "kind": rec["kind"], "m": rec["m"], "question": rec["question"]})}

L = [q for q in ids if rows[q]["A"]["inc"] and not rows[q]["C"]["inc"]]   # ours loses
W = [q for q in ids if rows[q]["C"]["inc"] and not rows[q]["A"]["inc"]]   # ours wins
BW = [q for q in ids if not rows[q]["C"]["inc"] and not rows[q]["A"]["inc"]]
print(f"variant {VAR}: C {sum(rows[q]['C']['inc'] for q in ids)} A {sum(rows[q]['A']['inc'] for q in ids)} | ours-loses {len(L)} ours-wins {len(W)} both-wrong {len(BW)}")

def summarize(name, qs):
    c = Counter()
    for q in qs:
        C, A, M = rows[q]["C"], rows[q]["A"], rows[q]["meta"]
        c["n"] += 1
        c[f"layer:{M['layer']}"] += 1
        c[f"qtype:{M['qtype']}"] += 1
        c["C_gold_missing"] += (not C["gold_in"]); c["A_gold_missing"] += (not A["gold_in"])
        c["C_trunc"] += (not C["fin"]); c["A_trunc"] += (not A["fin"])
        c["C_refusal"] += C["ref"]; c["A_refusal"] += A["ref"]
        if M["kind"] == "arith":
            fc, n = C["op_found"]; fa_, _ = A["op_found"]
            c["arith_n"] += 1
            c["C_ops_all"] += (fc == n); c["A_ops_all"] += (fa_ == n)
            c["C_ops_missing_some"] += (fc < n); c["A_ops_missing_some"] += (fa_ < n)
        c[f"C_fa_rel:{C['fa_cell_rel']}"] += 1
    print(f"\n== {name} ({len(qs)}) ==")
    for k, v in sorted(c.items()):
        print(f"  {k}: {v}")

summarize("ours loses (A right, C wrong)", L)
summarize("ours wins (C right, A wrong)", W)
summarize("both wrong", BW)

# population-level: arith operand coverage and conditional accuracy
print("\n== arith 371: operand coverage (all program literals present in context) ==")
for arm in ("C", "A"):
    ar = [q for q in ids if rows[q]["meta"]["kind"] == "arith"]
    allin = [q for q in ar if rows[q][arm]["op_found"][0] == rows[q][arm]["op_found"][1]]
    gin = [q for q in ar if rows[q][arm]["gold_in"]]
    print(f"  {arm}: gold cells in {len(gin)}/{len(ar)} | all operands in {len(allin)}/{len(ar)} | acc|all-ops-in {sum(rows[q][arm]['inc'] for q in allin)}/{len(allin)} | acc|not-all {sum(rows[q][arm]['inc'] for q in ar if q not in set(allin))}/{len(ar)-len(allin)}")
    for L_ in ("arith_m1", "arith_m2+"):
        sub = [q for q in ar if rows[q]["meta"]["layer"] == L_]
        ai = [q for q in sub if rows[q][arm]["op_found"][0] == rows[q][arm]["op_found"][1]]
        print(f"     {L_}: all-ops-in {len(ai)}/{len(sub)}, acc {sum(rows[q][arm]['inc'] for q in sub)}/{len(sub)}, acc|all-in {sum(rows[q][arm]['inc'] for q in ai)}/{len(ai)}")
print("\n== truncation ==")
for arm in ("C", "A"):
    tr = [q for q in ids if not rows[q][arm]["fin"]]
    print(f"  {arm}: truncated {len(tr)}, acc among truncated {sum(rows[q][arm]['inc'] for q in tr)}, arith among truncated {sum(rows[q]['meta']['kind']=='arith' for q in tr)}")
both_fin = [q for q in ids if rows[q]["C"]["fin"] and rows[q]["A"]["fin"]]
print(f"  both finished {len(both_fin)}: C {sum(rows[q]['C']['inc'] for q in both_fin)} A {sum(rows[q]['A']['inc'] for q in both_fin)} b:c {sum(rows[q]['C']['inc'] and not rows[q]['A']['inc'] for q in both_fin)}:{sum(rows[q]['A']['inc'] and not rows[q]['C']['inc'] for q in both_fin)}")
print("\n== context size ==")
for arm in ("C", "A"):
    print(f"  {arm}: cells avg {sum(rows[q][arm]['cells'] for q in ids)/len(ids):.1f}, tok avg {sum(rows[q][arm]['n_tok_ctx'] for q in ids)/len(ids):.0f}, out tok avg {sum(rows[q][arm]['n_out'] for q in ids)/len(ids):.0f}")
# gold positions in ours
print("\n== gold cell position in ours context (line idx / n_lines), losses vs wins ==")
for name, qs in (("L", L), ("W", W)):
    ps = [p for q in qs for p in rows[q]["C"]["gold_pos"] if p is not None]
    print(f"  {name}: found {len(ps)} gold cells, mean idx {sum(ps)/max(1,len(ps)):.1f}, >20 {sum(p>=20 for p in ps)}, missing {sum(p is None for q in qs for p in rows[q]['C']['gold_pos'])}")

# dump cases
def dump(name, qs):
    with open(OUT / f"{name}.txt", "w", encoding="utf-8") as f:
        for i, q in enumerate(qs, 1):
            C, A, M = rows[q]["C"], rows[q]["A"], rows[q]["meta"]
            f.write(f"\n########## {i}. {q} | {M['layer']} | {M['qtype']}\n")
            f.write(f"Q: {M['question']}\nGOLD: {M['gold']} | program: {M['program']} | evidence: {M['evidence']} | doc tables {M['n_tables']}\n")
            f.write(f"OURS: inc={C['inc']} gold_in={C['gold_in']} fin={C['fin']} out={C['n_out']} ref={C['ref']} ops={C['op_found']} cells={C['cells']} gold_pos={C['gold_pos']}/{C['n_lines']} fa={C['fa']!r} fa_rel={C['fa_cell_rel']} {C['fa_cells']}\n")
            f.write(f"LC  : inc={A['inc']} gold_in={A['gold_in']} fin={A['fin']} out={A['n_out']} ref={A['ref']} ops={A['op_found']} cells={A['cells']} fa={A['fa']!r} fa_rel={A['fa_cell_rel']}\n")
            f.write("--- OURS raw (tail 900) ---\n" + C["raw"][-900:] + "\n")
            f.write("--- LC raw (tail 500) ---\n" + A["raw"][-500:] + "\n")
dump("L_ours_loses", L); dump("W_ours_wins", W)
json.dump({"L": L, "W": W, "BW": BW}, open(OUT / "sets.json", "w"))
# context dump for L (ours context + LC context) for manual reading
with open(OUT / "L_contexts.txt", "w", encoding="utf-8") as f:
    for i, q in enumerate(L, 1):
        C, A, M = rows[q]["C"], rows[q]["A"], rows[q]["meta"]
        f.write(f"\n########## {i}. {q}\nQ: {M['question']}\nGOLD: {M['gold']} | {M['program']} | {M['evidence']}\n--- OURS ctx ---\n{C['ctx']}\n--- LC ctx ---\n{A['ctx']}\n")
