#!/usr/bin/env python3
"""Experiment 2: does contrasting the title/row/column conditions help, and does
emitting a cell ID rather than the value help or hurt? dev holdout only.

  PYTHONPATH=. .venv/bin/python scripts/cell_id_conditions.py run --arm base
  PYTHONPATH=. .venv/bin/python scripts/cell_id_conditions.py gate
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys, time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag_agent.serialization.cell_id import build_value_map, render_cellid, predict

OUT = ROOT / "results/cell_id_v3"
RECORDS = ROOT / "results/cell_id_v2/d_s3c_hybrid_records.jsonl"
PREREG = ROOT / "PREREG-2026-09-10-cell-id-conditions.md"
READER = "local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit"
#: The first 700 dev records chose experiment 1's prompt and parse rule, so the
#: evaluation starts after them.
HOLDOUT_START = 700

from scripts.answer_accuracy import BASE
from scripts.cell_id_select import P1

_CONTRAST = (
    "You answer questions about a table. The context lines are cells of the table, "
    "each written as its headers and its value, and each numbered [#n]. Use only the "
    "context.\nFirst write the conditions the question requires, one per line:\n"
    "Table: <the table the question is about>\nRow: <the row header the question asks for>\n"
    "Column: <the column header the question asks for>\n"
    "Then list the numbers of the context lines that satisfy all three:\n"
    "Matching: [#n], [#n], ...\nThen write one final line, exactly:\n")

A = _CONTRAST + ("Answer: [#n]\nOn the Answer line write only the number of the cell "
                 "that holds the answer — do not write the value itself. If the question "
                 "asks for several values, give several numbers separated by commas.")
B = _CONTRAST + ("Answer: <value>\nOn the Answer line write the value alone, copied "
                 "exactly as the context writes it — no sentence, no units, no "
                 "explanation. If the question asks for several values, separate them "
                 "with commas.")

#: arm -> (system prompt, output mode, max_new_tokens)
ARMS = {"base": (BASE, "value", 64),
        "c0b": (P1, "id", 256),
        "a": (A, "id", 256),
        "b": (B, "value", 256)}

_ANSWER = re.compile(r"(?i)^[ \t]*answer[ \t]*:[ \t]*(.*)$", re.M)
_ID = re.compile(r"\[\s*#?\s*(\d+)\s*\]|#\s*(\d+)")


def answer_line(raw):
    """Text after the LAST 'Answer:'; None when the model never wrote one."""
    m = _ANSWER.findall(str(raw))
    return m[-1].strip() if m else None


def resolve(raw, arm, context, vmap):
    mode = ARMS[arm][1]
    line = answer_line(raw)
    if arm == "c0b":                       # experiment 1's rule, unchanged
        return predict(raw, context, vmap, "bracket_bare")
    if mode == "value":
        if arm == "base":
            return str(raw), {}
        text = line if line is not None else str(raw)
        return text, {"failure": None if line is not None else "no_answer_line",
                      "abstained": int(bool(re.fullmatch(r"(?i)\s*none\.?\s*", text)))}
    src = line if line is not None else str(raw)
    pred, audit = predict(src, context, vmap, "bracket_bare")
    audit["answer_line_present"] = line is not None
    audit["abstained"] = int(bool(line is not None and re.fullmatch(r"(?i)\s*none\.?\s*", line)))
    return pred, audit


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def holdout():
    return [r for r in jl(RECORDS) if "correct" in r][HOLDOUT_START:]


def primary(rows):
    return {r["query_id"] for r in rows
            if r["mode"] == "all" and (r.get("aggregation") or "none") == "none" and r["m"] == 1}


def run(arm):
    import torch, transformers, bitsandbytes
    from rag_agent.llm.factory import build_llm
    from rag_agent.eval.metrics import hitab_exact_match_text
    rows = holdout()
    prompt, mode, max_tok = ARMS[arm]
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"dev_holdout_{arm}.jsonl"
    pt = json.loads((ROOT / "results/tableconf/totto_page_titles.json").read_text())
    vmap = ({} if mode == "value" and arm == "base"
            else build_value_map(str(ROOT / "data/hitab"), {r["table_id"] for r in rows}, pt, "s3c"))
    config = {"arm": arm, "reader": READER, "split": "dev", "holdout_start": HOLDOUT_START,
              "records": str(RECORDS.relative_to(ROOT)), "records_sha256": sha(RECORDS),
              "prereg_sha256": sha(PREREG), "module_sha256": sha(ROOT / "rag_agent/serialization/cell_id.py"),
              "runner_sha256": sha(__file__), "scorer_sha256": sha(ROOT / "rag_agent/eval/metrics.py"),
              "reader_backend_sha256": sha(ROOT / "rag_agent/llm/local_qwen.py"),
              "seed": 42, "temperature": 0, "batch_size": 1, "max_new_tokens": max_tok,
              "output_mode": mode, "prompt": prompt, "n_expected": len(rows),
              "torch": torch.__version__, "transformers": transformers.__version__,
              "bitsandbytes": bitsandbytes.__version__, "cuda": torch.version.cuda}
    meta = out.with_suffix(".meta.json")
    if meta.exists():
        assert json.loads(meta.read_text()) == config, "resume config differs"
    else:
        meta.write_text(json.dumps(config, ensure_ascii=False, indent=2))
    done = jl(out) if out.exists() else []
    assert [r["query_id"] for r in done] == [r["query_id"] for r in rows[:len(done)]]
    if len(done) == len(rows):
        print(f"already complete: {out}")
        return
    llm = build_llm(READER)
    torch.manual_seed(42)
    prim = primary(rows)
    t0 = time.time()
    with out.open("a") as f:
        for n, r in enumerate(rows[len(done):], len(done) + 1):
            ctx = "\n".join(r["context"]) if arm == "base" else render_cellid(r["context"])
            user = "Context:\n" + ctx + f"\n\nQuestion: {r['question']}\nAnswer:"
            n_tok = llm.n_prompt_tokens(prompt, user)
            assert n_tok + max_tok <= llm.context_limit
            t = time.time()
            raw = llm.complete(prompt, user, max_tokens=max_tok, temperature=0.0)
            pred, audit = resolve(raw, arm, r["context"], vmap)
            row = {"query_id": r["query_id"], "pred": pred, "raw_pred": raw,
                   "answer": r["answer"], "retrieval_correct": r["correct"],
                   "mode": r["mode"], "aggregation": r.get("aggregation"), "m": r["m"],
                   "table_id": r["table_id"], "context": r["context"], "audit": audit,
                   "n_prompt_tok": n_tok,
                   "n_gen_tok": len(llm.tokenizer(raw)["input_ids"]),
                   "input_sha256": hashlib.sha256((prompt + "\0" + user).encode()).hexdigest(),
                   "seconds": time.time() - t}
            row["answer_correct"] = int(hitab_exact_match_text(pred, r["answer"]))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            done.append(row)
            if n % 50 == 0 or n == len(rows):
                p = [x for x in done if x["query_id"] in prim]
                print(json.dumps({"arm": arm, "done": n, "total": len(rows),
                                  "elapsed_s": round(time.time() - t0), "primary_n": len(p),
                                  "primary_correct": sum(x["answer_correct"] for x in p)}), flush=True)
    out.with_suffix(".summary.json").write_text(json.dumps(
        {**config, "complete": True, "n": len(done), "elapsed_s": round(time.time() - t0)}, indent=2))
    print("wrote", out)


def gate():
    import numpy as np
    from scipy.stats import binomtest
    rows = holdout()
    prim = sorted(primary(rows))
    data = {}
    for arm in ARMS:
        f = OUT / f"dev_holdout_{arm}.jsonl"
        if f.exists() and len(jl(f)) == len(rows):
            data[arm] = {r["query_id"]: r for r in jl(f)}
    assert "base" in data, "the free-generation baseline must be run"
    rep = {"prereg_sha256": sha(PREREG), "holdout_n": len(rows), "primary_n": len(prim),
           "note": "dev holdout (records 700+), never used to choose a prompt", "arms": {}}
    b = np.array([data["base"][k]["answer_correct"] for k in prim], dtype=np.int8)
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(prim), (10000, len(prim)))
    base_secs = sum(r["seconds"] for r in data["base"].values())
    base_gen = np.mean([r["n_gen_tok"] for r in data["base"].values()])
    raw_p = {}
    for arm, rowsd in data.items():
        y = np.array([rowsd[k]["answer_correct"] for k in prim], dtype=np.int8)
        d = (y - b).astype(float)
        ci = np.quantile(d[idx].mean(axis=1), [.025, .975]).tolist()
        gain, loss = int(((b == 0) & (y == 1)).sum()), int(((b == 1) & (y == 0)).sum())
        p = float(binomtest(min(gain, loss), gain + loss).pvalue) if gain + loss else 1.0
        br = Counter(r["audit"].get("failure") for r in rowsd.values() if r["audit"].get("failure"))
        secs = sum(r["seconds"] for r in rowsd.values())
        gen = float(np.mean([r["n_gen_tok"] for r in rowsd.values()]))
        e = {"primary_correct": int(y.sum()), "primary_em": round(float(y.mean()), 4),
             "delta_vs_base": round(float(d.mean()), 4),
             "delta_ci95": [round(c, 4) for c in ci],
             "gain": gain, "loss": loss, "mcnemar_exact_p": p,
             "format_breakage": dict(br),
             "format_breakage_rate": round(sum(br.values()) / len(rowsd), 4),
             "abstained": sum(r["audit"].get("abstained", 0) for r in rowsd.values()),
             "seconds_total": round(secs, 1),
             "seconds_per_query": round(secs / len(rowsd), 3),
             "time_vs_base_x": round(secs / base_secs, 2),
             "prompt_tok_mean": round(float(np.mean([r["n_prompt_tok"] for r in rowsd.values()])), 1),
             "gen_tok_mean": round(gen, 1), "gen_tok_vs_base": round(gen - base_gen, 1)}
        if arm != "base":
            raw_p[arm] = p
            e["gate_em_margin_met"] = bool(d.mean() >= .02)
            e["gate_ci_lower_above_zero"] = bool(ci[0] > 0)
            e["gate_passed"] = bool(d.mean() >= .02 and ci[0] > 0)
        rep["arms"][arm] = e
    # Holm over the two pre-registered candidate arms only.
    cand = [k for k in ("a", "b") if k in raw_p]
    order = sorted(cand, key=lambda k: raw_p[k])
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, raw_p[k] * (len(cand) - rank)))
        rep["arms"][k]["holm_p"] = running
    if "a" in data and "b" in data:
        ya = np.array([data["a"][k]["answer_correct"] for k in prim], dtype=np.int8)
        yb = np.array([data["b"][k]["answer_correct"] for k in prim], dtype=np.int8)
        d = (ya - yb).astype(float)
        rep["id_output_effect_a_minus_b"] = {
            "delta": round(float(d.mean()), 4),
            "ci95": [round(c, 4) for c in np.quantile(d[idx].mean(axis=1), [.025, .975]).tolist()],
            "note": "condition contrast held constant; isolates emitting an ID vs the value"}
        rep["contrast_effect_b_minus_base"] = rep["arms"]["b"]["delta_vs_base"]
    rep["test_run_authorised"] = bool(any(rep["arms"].get(k, {}).get("gate_passed") for k in ("a", "b")))
    (OUT / "GATE.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print(json.dumps(rep, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "gate"])
    ap.add_argument("--arm", choices=sorted(ARMS), default="base")
    a = ap.parse_args()
    run(a.arm) if a.action == "run" else gate()


if __name__ == "__main__":
    main()
