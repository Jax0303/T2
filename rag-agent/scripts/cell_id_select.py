#!/usr/bin/env python3
"""Cell-ID selection: the reader names a retrieved cell, the code returns its value.

Fixed reader, fixed retrieval, fixed scorer. Prompt and ID-parsing rule are
chosen on dev and then frozen for the single test run.

  PYTHONPATH=. .venv/bin/python scripts/cell_id_select.py audit --split test
  PYTHONPATH=. .venv/bin/python scripts/cell_id_select.py run  --split dev --arm p1 --limit 700
  PYTHONPATH=. .venv/bin/python scripts/cell_id_select.py score --split dev
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag_agent.serialization.cell_id import (build_value_map, audit_value_map,
                                             render_cellid, predict, RULES)

OUT = ROOT / "results/cell_id_v2"
READER = "local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit"
RECORDS = {"test": ROOT / "results/retrieval_accuracy/t_s3c_hybrid_records.jsonl",
           "dev": OUT / "d_s3c_hybrid_records.jsonl"}

from scripts.answer_accuracy import BASE

#: BASE with the output requirement swapped: name the cell, do not write the value.
P1 = ("You answer questions about a table. The context lines are cells of the "
      "table, each written as its headers and its value, and each numbered [#n]. "
      "Use only the context. Answer with the number of the cell that holds the "
      "answer, written as [#n] and nothing else — do not write the value itself. "
      "If the question asks for several values, give several numbers separated "
      "by commas.")
#: P1 plus a worked format example, the one knob dev is asked to settle.
P2 = P1 + (" For example, if the cell numbered [#7] holds the answer, the whole "
           "reply is: [#7]")
ARMS = {"base": BASE, "p1": P1, "p2": P2}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def records(split, limit=0):
    rows = [r for r in read_jsonl(RECORDS[split]) if "correct" in r]
    return rows[:limit] if limit else rows


def value_map(split, rows):
    pt = json.loads((ROOT / "results/tableconf/totto_page_titles.json").read_text())
    return build_value_map(str(ROOT / "data/hitab"), {r["table_id"] for r in rows}, pt, "s3c")


def primary(rows):
    return {r["query_id"] for r in rows
            if r["mode"] == "all" and (r.get("aggregation") or "none") == "none" and r["m"] == 1}


def audit(split):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = records(split)
    vm = value_map(split, rows)
    a = audit_value_map([r["context"] for r in rows], vm)
    # Input preservation: the numbered context must be the frozen lines, verbatim.
    for r in rows:
        assert len(r["context"]) == r["cells_in_context"] == 20
        lines = render_cellid(r["context"]).split("\n")
        assert len(lines) == 20
        for k, (line, orig) in enumerate(zip(lines, r["context"]), 1):
            assert line == f"[#{k}] {orig}", "context line altered"
    rep = {"split": split, "n": len(rows), "records_sha256": sha(RECORDS[split]),
           "map_size": len(vm), "context_preserved": True,
           "n_primary": len(primary(rows)), **a}
    (OUT / f"AUDIT_{split}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "disagreement_examples"}, indent=2))
    assert a["disagree"] == 0 and a["unreadable"] == 0, "value map is not trustworthy"


def run(split, arm, limit, prereg):
    import torch, transformers, bitsandbytes
    from rag_agent.llm.factory import build_llm
    from rag_agent.eval.metrics import hitab_exact_match_text
    rows = records(split, limit)
    prompt = ARMS[arm]
    out = OUT / f"{split}_{arm}{f'_n{limit}' if limit else ''}.jsonl"
    OUT.mkdir(parents=True, exist_ok=True)
    vm = {} if arm == "base" else value_map(split, rows)
    config = {"split": split, "arm": arm, "reader": READER,
              "records": str(RECORDS[split].relative_to(ROOT)),
              "records_sha256": sha(RECORDS[split]),
              "module_sha256": sha(ROOT / "rag_agent/serialization/cell_id.py"),
              "runner_sha256": sha(__file__),
              "reader_backend_sha256": sha(ROOT / "rag_agent/llm/local_qwen.py"),
              "scorer_sha256": sha(ROOT / "rag_agent/eval/metrics.py"),
              "prereg_sha256": sha(ROOT / prereg) if prereg else None,
              "seed": 42, "max_new_tokens": 64, "temperature": 0, "batch_size": 1,
              "prompt": prompt, "n_expected": len(rows),
              "torch": torch.__version__, "transformers": transformers.__version__,
              "bitsandbytes": bitsandbytes.__version__, "cuda": torch.version.cuda}
    meta = out.with_suffix(".meta.json")
    if meta.exists():
        assert json.loads(meta.read_text()) == config, "resume config differs"
    else:
        meta.write_text(json.dumps(config, ensure_ascii=False, indent=2))
    done = read_jsonl(out) if out.exists() else []
    assert [r["query_id"] for r in done] == [r["query_id"] for r in rows[:len(done)]]
    if len(done) == len(rows):
        print(f"already complete: {out}")
        return
    llm = build_llm(READER)
    torch.manual_seed(42)
    out.with_suffix(".runtime.json").write_text(json.dumps(
        {"name": llm.name, "commit_hash": getattr(llm.model.config, "_commit_hash", None),
         "attn_implementation": getattr(llm.model.config, "_attn_implementation", None),
         "gpu": torch.cuda.get_device_name(0)}, indent=2))
    prim = primary(rows)
    t0 = time.time()
    with out.open("a") as f:
        for n, r in enumerate(rows[len(done):], len(done) + 1):
            ctx = "\n".join(r["context"]) if arm == "base" else render_cellid(r["context"])
            user = "Context:\n" + ctx + f"\n\nQuestion: {r['question']}\nAnswer:"
            assert llm.n_prompt_tokens(prompt, user) + 64 <= llm.context_limit
            t = time.time()
            raw = llm.complete(prompt, user, max_tokens=64, temperature=0.0)
            row = {"query_id": r["query_id"], "raw_pred": raw, "answer": r["answer"],
                   "retrieval_correct": r["correct"], "mode": r["mode"],
                   "aggregation": r.get("aggregation"), "m": r["m"],
                   "table_id": r["table_id"], "context": r["context"],
                   "input_sha256": hashlib.sha256((prompt + "\0" + user).encode()).hexdigest(),
                   "seconds": time.time() - t}
            if arm == "base":
                row["pred"], row["audit"] = raw, {}
            else:
                # Every declared rule is scored from the same generation.
                for rule in RULES:
                    p, a = predict(raw, r["context"], vm, rule)
                    row[f"pred__{rule}"], row[f"audit__{rule}"] = p, a
                row["pred"] = row["pred__bracket"]
            row["answer_correct"] = int(hitab_exact_match_text(row["pred"], r["answer"]))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            done.append(row)
            if n % 50 == 0 or n == len(rows):
                pr = [x for x in done if x["query_id"] in prim]
                print(json.dumps({"split": split, "arm": arm, "done": n, "total": len(rows),
                                  "elapsed_s": round(time.time() - t0), "primary_n": len(pr),
                                  "primary_correct": sum(x["answer_correct"] for x in pr)}),
                      flush=True)
    (out.with_suffix(".summary.json")).write_text(json.dumps(
        {**config, "complete": True, "n": len(done),
         "elapsed_s": round(time.time() - t0),
         "seconds_per_query": round((time.time() - t0) / max(1, len(rows) - 0), 3)}, indent=2))
    print("wrote", out)


def score(split, limit, arms):
    from rag_agent.eval.metrics import hitab_exact_match_text
    rows = records(split, limit)
    prim = primary(rows)
    tag = f"_n{limit}" if limit else ""
    res = {}
    for arm in arms:
        f = OUT / f"{split}_{arm}{tag}.jsonl"
        if not f.exists():
            continue
        got = read_jsonl(f)
        entry = {"n": len(got), "complete": len(got) == len(rows)}
        rules = ["base"] if arm == "base" else list(RULES)
        for rule in rules:
            key = "pred" if arm == "base" else f"pred__{rule}"
            ok = {x["query_id"]: int(hitab_exact_match_text(x[key], x["answer"])) for x in got}
            br = Counter(x[f"audit__{rule}"].get("failure") for x in got
                         if arm != "base" and x[f"audit__{rule}"].get("failure"))
            p = [q for q in ok if q in prim]
            entry[rule] = {"primary_n": len(p),
                           "primary_correct": sum(ok[q] for q in p),
                           "primary_em": round(sum(ok[q] for q in p) / max(1, len(p)), 4),
                           "all_correct": sum(ok.values()),
                           "all_em": round(sum(ok.values()) / max(1, len(ok)), 4),
                           "breakage": dict(br),
                           "breakage_rate": round(sum(br.values()) / max(1, len(got)), 4)}
        res[arm] = entry
    (OUT / f"SCORE_{split}{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["audit", "run", "score"])
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--arm", choices=sorted(ARMS), default="p1")
    ap.add_argument("--arms", default="base,p1,p2")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prereg", default="")
    a = ap.parse_args()
    if a.action == "audit":
        audit(a.split)
    elif a.action == "run":
        run(a.split, a.arm, a.limit, a.prereg)
    else:
        score(a.split, a.limit, a.arms.split(","))


if __name__ == "__main__":
    main()
