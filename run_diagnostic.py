"""
HiTab Dense Embedder Hierarchy Loss Diagnostic
================================================

4 experiments diagnosing whether dense embedders encode table structure:
  1. Scale-up + multi-embedder comparison (full dev set, 3 models)
  2. Complexity-stratified retrieval analysis
  3. Structure sensitivity probing (shuffle / header removal / header swap)
  4. Failure query analysis
"""

import csv as csv_mod
import json
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
from sentence_transformers import SentenceTransformer
from tabulate import tabulate
from tqdm import tqdm

# ----------------------------- paths ------------------------------
HITAB_ROOT = Path(r"C:\Users\user\hitab_pilot\HiTab")
TABLES_DIR = HITAB_ROOT / "data" / "tables" / "raw"
DEV_PATH = HITAB_ROOT / "data" / "dev_samples.jsonl"
OUTPUT_DIR = Path(r"C:\Users\user\hitab_pilot\results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
KS = [1, 3, 5, 10]

MODELS = [
    {
        "name": "bge-small",
        "hf": "BAAI/bge-small-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "doc_prefix": "",
        "batch_table": 16,
        "batch_query": 32,
    },
    {
        "name": "bge-base",
        "hf": "BAAI/bge-base-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "doc_prefix": "",
        "batch_table": 8,
        "batch_query": 16,
    },
    {
        "name": "e5-small",
        "hf": "intfloat/e5-small-v2",
        "query_prefix": "query: ",
        "doc_prefix": "passage: ",
        "batch_table": 16,
        "batch_query": 32,
    },
]

# ===================== REUSE PILOT UTILITIES ======================
# Import parser + serializer from pilot
sys.path.insert(0, str(Path(__file__).parent))
from run_pilot import (
    load_table,
    serialize_markdown,
    build_merge_map,
    cell_value,
    get_header_path,
)


# ===================== COMPLEXITY METRICS =========================
def tree_depth(node: dict, depth: int = 0) -> int:
    """Max depth of a header tree (sentinel root = depth 0, its children = 1, etc.)."""
    children = node.get("children", []) or node.get("children_dict", []) or []
    if not children:
        return depth
    return max(tree_depth(c, depth + 1) for c in children)


def compute_complexity(table_path: Path) -> dict:
    """Read raw JSON and compute complexity metrics."""
    with open(table_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    hd = tree_depth(raw["top_root"])  # 1 = flat, 2+ = hierarchical
    merged = len(raw.get("merged_regions", []))
    texts = raw.get("texts") or raw.get("data") or []
    n_rows = len(texts)
    n_cols = max((len(r) for r in texts), default=0)

    lh_tree = tree_depth(raw["left_root"])

    return {
        "header_depth": hd,
        "num_merged_cells": merged,
        "table_size": n_rows * n_cols,
        "left_header_depth": lh_tree,
        "n_rows": n_rows,
        "n_cols": n_cols,
    }


def classify_complexity(cm: dict) -> str:
    hd = cm["header_depth"]
    mc = cm["num_merged_cells"]
    if hd >= 3 or mc >= 4:
        return "complex"
    if hd == 2 or (1 <= mc <= 3):
        return "moderate"
    return "simple"


# ===================== DATA LOADING ===============================
def load_all_dev() -> tuple[list[str], list[dict], list[dict], list[dict]]:
    """Load entire dev set. Returns (table_ids, parsed_tables, qas, complexity_list)."""
    table_to_qas: dict[str, list[dict]] = {}
    with open(DEV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            qa = json.loads(line)
            table_to_qas.setdefault(qa["table_id"], []).append(qa)

    table_ids = sorted(tid for tid in table_to_qas if (TABLES_DIR / f"{tid}.json").exists())
    print(f"  Total dev tables with files: {len(table_ids)}")

    parsed_tables = []
    complexity_list = []
    fail_count = 0
    for tid in tqdm(table_ids, desc="  parsing"):
        p = TABLES_DIR / f"{tid}.json"
        try:
            parsed_tables.append(load_table(p))
        except Exception:
            parsed_tables.append(None)
            fail_count += 1
        try:
            complexity_list.append(compute_complexity(p))
        except Exception:
            complexity_list.append({"header_depth": 0, "num_merged_cells": 0, "table_size": 0, "left_header_depth": 0, "n_rows": 0, "n_cols": 0})
    if fail_count:
        print(f"  WARNING: {fail_count} tables failed to parse")

    qas = []
    for tid in table_ids:
        qas.extend(table_to_qas[tid])

    return table_ids, parsed_tables, qas, complexity_list


# ===================== EVALUATION =================================
def compute_ranks(q_embs: np.ndarray, t_embs: np.ndarray, gold_idx: list[int]) -> np.ndarray:
    sims = q_embs @ t_embs.T
    ranks = np.zeros(len(gold_idx), dtype=int)
    for i, g in enumerate(gold_idx):
        order = np.argsort(-sims[i])
        ranks[i] = int(np.where(order == g)[0][0]) + 1
    return ranks


def metrics_from_ranks(ranks: np.ndarray) -> dict:
    out = {}
    for k in KS:
        out[f"R@{k}"] = float((ranks <= k).mean())
    out["MRR"] = float((1.0 / ranks).mean())
    return out


def get_top1_idx(q_embs: np.ndarray, t_embs: np.ndarray) -> np.ndarray:
    sims = q_embs @ t_embs.T
    return np.argmax(sims, axis=1)


# ================== SERIALIZE + EMBED HELPERS =====================
def serialize_all_markdown(parsed_tables: list) -> list[str]:
    out = []
    for p in parsed_tables:
        if p is None:
            out.append("")
        else:
            try:
                out.append(serialize_markdown(p))
            except Exception:
                out.append("")
    return out


def serialize_markdown_no_header(parsed: dict) -> str:
    """Markdown but header row stripped -- data only."""
    texts = parsed["texts"]
    data_rows = parsed["data_rows"]
    data_cols = parsed["data_cols"]
    left_cols = parsed["left_header_cols"]
    merge_map = build_merge_map(parsed)

    n_data_cols = len(data_cols)
    out = []
    title = parsed["title"]
    if title:
        out.append(f"# {title}")
        out.append("")
    generic_headers = [f"col{i}" for i in range(1 + n_data_cols)]
    out.append("| " + " | ".join(generic_headers) + " |")
    out.append("| " + " | ".join(["---"] * (1 + n_data_cols)) + " |")
    for r in data_rows:
        left_vals = " / ".join(cell_value(parsed, merge_map, r, lc) for lc in range(left_cols)).strip(" /")
        row_vals = [cell_value(parsed, merge_map, r, c) for c in data_cols]
        out.append("| " + left_vals + " | " + " | ".join(row_vals) + " |")
    return "\n".join(out)


def serialize_markdown_shuffled(parsed: dict, rng: random.Random) -> str:
    """Markdown with data rows in random order."""
    p2 = deepcopy(parsed)
    rows = list(p2["data_rows"])
    rng.shuffle(rows)
    p2["data_rows"] = rows
    return serialize_markdown(p2)


def serialize_markdown_swapped_header(parsed_body: dict, parsed_header_src: dict) -> str:
    """Use body (data rows) from parsed_body but headers from parsed_header_src."""
    p2 = deepcopy(parsed_body)
    p2["top_paths"] = deepcopy(parsed_header_src["top_paths"])
    p2["left_paths"] = deepcopy(parsed_header_src["left_paths"])
    p2["title"] = parsed_header_src["title"]
    return serialize_markdown(p2)


# ==================== EXPERIMENT 1: MULTI EMBEDDER ================
def run_experiment1(table_ids, parsed_tables, qas):
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Scale-up + Multi-Embedder Comparison")
    print("=" * 80)

    tid_to_idx = {tid: i for i, tid in enumerate(table_ids)}
    qa_gold = [tid_to_idx[qa["table_id"]] for qa in qas]

    serialized = serialize_all_markdown(parsed_tables)
    avg_chars = float(np.mean([len(s) for s in serialized]))
    print(f"  Tables: {len(table_ids)}, QAs: {len(qas)}, avg markdown chars: {avg_chars:.0f}")

    results = {}
    all_model_data = {}

    for mcfg in MODELS:
        mname = mcfg["name"]
        print(f"\n  --- {mname} ({mcfg['hf']}) ---")
        t0 = time.time()
        model = SentenceTransformer(mcfg["hf"], device="cpu")

        docs = [mcfg["doc_prefix"] + s for s in serialized]
        queries = [mcfg["query_prefix"] + qa["question"] for qa in qas]

        print(f"    Embedding {len(docs)} tables...")
        t_embs = model.encode(docs, normalize_embeddings=True, show_progress_bar=True,
                              batch_size=mcfg["batch_table"])
        print(f"    Embedding {len(queries)} queries...")
        q_embs = model.encode(queries, normalize_embeddings=True, show_progress_bar=True,
                              batch_size=mcfg["batch_query"])

        ranks = compute_ranks(q_embs, t_embs, qa_gold)
        metrics = metrics_from_ranks(ranks)
        elapsed = time.time() - t0

        metrics["time_sec"] = round(elapsed, 1)
        results[mname] = metrics
        all_model_data[mname] = {
            "q_embs": q_embs,
            "t_embs": t_embs,
            "ranks": ranks,
            "model_obj": model,
            "cfg": mcfg,
        }
        print(f"    R@1={metrics['R@1']:.3f}  R@5={metrics['R@5']:.3f}  R@10={metrics['R@10']:.3f}  MRR={metrics['MRR']:.3f}  ({elapsed:.0f}s)")

        del model

    # Print comparison table
    headers = ["Model"] + [f"R@{k}" for k in KS] + ["MRR", "Time(s)"]
    rows = []
    for mname in [m["name"] for m in MODELS]:
        m = results[mname]
        rows.append([mname, *[f"{m[f'R@{k}']:.3f}" for k in KS], f"{m['MRR']:.3f}", f"{m['time_sec']}"])
    print("\n" + tabulate(rows, headers=headers, tablefmt="grid"))

    return results, all_model_data, serialized


# ================ EXPERIMENT 2: COMPLEXITY STRATIFICATION =========
def run_experiment2(table_ids, parsed_tables, qas, complexity_list, all_model_data):
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Complexity-Stratified Retrieval Analysis")
    print("=" * 80)

    tid_to_idx = {tid: i for i, tid in enumerate(table_ids)}
    groups = {}
    for i, (tid, cm) in enumerate(zip(table_ids, complexity_list)):
        g = classify_complexity(cm)
        groups.setdefault(g, []).append(i)

    # Print group stats
    print("\n  Complexity groups:")
    for g in ["simple", "moderate", "complex"]:
        indices = groups.get(g, [])
        depths = [complexity_list[i]["header_depth"] for i in indices]
        merges = [complexity_list[i]["num_merged_cells"] for i in indices]
        sizes = [complexity_list[i]["table_size"] for i in indices]
        print(f"    {g:10s}: {len(indices):4d} tables | avg depth={np.mean(depths) if depths else 0:.1f} "
              f"| avg merges={np.mean(merges) if merges else 0:.1f} | avg size={np.mean(sizes) if sizes else 0:.0f}")

    # Map each QA to its table's complexity group
    qa_groups = []
    for qa in qas:
        idx = tid_to_idx[qa["table_id"]]
        qa_groups.append(classify_complexity(complexity_list[idx]))

    results = {}
    for mname, mdata in all_model_data.items():
        q_embs = mdata["q_embs"]
        t_embs = mdata["t_embs"]
        qa_gold = [tid_to_idx[qa["table_id"]] for qa in qas]
        ranks_all = mdata["ranks"]

        for g in ["simple", "moderate", "complex"]:
            mask = [i for i, qg in enumerate(qa_groups) if qg == g]
            if not mask:
                continue
            sub_ranks = ranks_all[mask]
            metrics = metrics_from_ranks(sub_ranks)
            metrics["n_queries"] = len(mask)
            results[(mname, g)] = metrics

    # Cross table: model x complexity
    print("\n  Model x Complexity (R@1 / R@5 / R@10 / MRR):")
    headers = ["Model", "Group", "N_q", "R@1", "R@5", "R@10", "MRR"]
    rows = []
    for mname in [m["name"] for m in MODELS]:
        for g in ["simple", "moderate", "complex"]:
            key = (mname, g)
            if key not in results:
                continue
            m = results[key]
            rows.append([mname, g, m["n_queries"],
                         f"{m['R@1']:.3f}", f"{m['R@5']:.3f}", f"{m['R@10']:.3f}", f"{m['MRR']:.3f}"])
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    return results, qa_groups


# ================ EXPERIMENT 3: STRUCTURE PROBING =================
def run_experiment3(table_ids, parsed_tables, qas, serialized_orig):
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Structure Sensitivity Probing (bge-small)")
    print("=" * 80)

    rng = random.Random(SEED)
    tid_to_idx = {tid: i for i, tid in enumerate(table_ids)}
    qa_gold = [tid_to_idx[qa["table_id"]] for qa in qas]

    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    bge_prefix = "Represent this sentence for searching relevant passages: "

    # Embed originals (recompute to keep this self-contained)
    print("\n  Embedding original tables...")
    orig_embs = model.encode(serialized_orig, normalize_embeddings=True, show_progress_bar=True, batch_size=16)
    queries = [bge_prefix + qa["question"] for qa in qas]
    print("  Embedding queries...")
    q_embs = model.encode(queries, normalize_embeddings=True, show_progress_bar=True, batch_size=32)

    # --------- 3-1: Row Shuffle Test ----------
    print("\n  [3-1] Row Shuffle Test (100 tables)...")
    t31 = time.time()
    sample_idx = rng.sample(range(len(table_ids)), min(100, len(table_ids)))
    shuffle_sims = []
    for idx in tqdm(sample_idx, desc="    shuffle"):
        parsed = parsed_tables[idx]
        if parsed is None or len(parsed["data_rows"]) < 2:
            continue
        shuffled_md = serialize_markdown_shuffled(parsed, rng)
        emb_orig = orig_embs[idx:idx + 1]
        emb_shuf = model.encode([shuffled_md], normalize_embeddings=True, show_progress_bar=False)
        sim = float((emb_orig @ emb_shuf.T)[0, 0])
        shuffle_sims.append(sim)
    shuffle_mean = float(np.mean(shuffle_sims))
    shuffle_std = float(np.std(shuffle_sims))
    shuffle_min = float(np.min(shuffle_sims))
    shuffle_max = float(np.max(shuffle_sims))
    t31_elapsed = time.time() - t31
    print(f"    Row-shuffle cosine similarity: mean={shuffle_mean:.4f} std={shuffle_std:.4f} "
          f"min={shuffle_min:.4f} max={shuffle_max:.4f} ({t31_elapsed:.0f}s)")

    # --------- 3-2: Header Removal Test ----------
    print("\n  [3-2] Header Removal Test...")
    t32 = time.time()
    noheader_serialized = []
    for p in parsed_tables:
        if p is None:
            noheader_serialized.append("")
        else:
            try:
                noheader_serialized.append(serialize_markdown_no_header(p))
            except Exception:
                noheader_serialized.append("")
    noheader_embs = model.encode(noheader_serialized, normalize_embeddings=True, show_progress_bar=True, batch_size=16)

    removal_sims = []
    for i in range(len(table_ids)):
        sim = float((orig_embs[i:i + 1] @ noheader_embs[i:i + 1].T)[0, 0])
        removal_sims.append(sim)
    removal_mean = float(np.mean(removal_sims))
    removal_std = float(np.std(removal_sims))

    ranks_orig = compute_ranks(q_embs, orig_embs, qa_gold)
    ranks_noheader = compute_ranks(q_embs, noheader_embs, qa_gold)
    metrics_orig = metrics_from_ranks(ranks_orig)
    metrics_noheader = metrics_from_ranks(ranks_noheader)
    t32_elapsed = time.time() - t32

    print(f"    Header-removal cosine similarity: mean={removal_mean:.4f} std={removal_std:.4f}")
    print(f"    Original   -> R@1={metrics_orig['R@1']:.3f} R@5={metrics_orig['R@5']:.3f} R@10={metrics_orig['R@10']:.3f}")
    print(f"    No-header  -> R@1={metrics_noheader['R@1']:.3f} R@5={metrics_noheader['R@5']:.3f} R@10={metrics_noheader['R@10']:.3f}")
    print(f"    R@1 change = {(metrics_noheader['R@1'] - metrics_orig['R@1']) * 100:+.1f}pp ({t32_elapsed:.0f}s)")

    # --------- 3-3: Header Swap Test ----------
    print("\n  [3-3] Header Swap Test (50 pairs)...")
    t33 = time.time()
    valid_indices = [i for i, p in enumerate(parsed_tables) if p is not None and len(p["data_cols"]) > 0]
    pair_indices = rng.sample(valid_indices, min(100, len(valid_indices)))
    pairs = list(zip(pair_indices[:50], pair_indices[50:]))
    if len(pairs) < 50:
        pairs = [(pair_indices[i], pair_indices[(i + 1) % len(pair_indices)]) for i in range(min(50, len(pair_indices)))]

    swap_sims = []
    for a, b in tqdm(pairs, desc="    swap"):
        pa, pb = parsed_tables[a], parsed_tables[b]
        if pa is None or pb is None:
            continue
        try:
            swapped_md = serialize_markdown_swapped_header(pa, pb)
        except Exception:
            continue
        emb_orig = orig_embs[a:a + 1]
        emb_swap = model.encode([swapped_md], normalize_embeddings=True, show_progress_bar=False)
        sim = float((emb_orig @ emb_swap.T)[0, 0])
        swap_sims.append(sim)
    swap_mean = float(np.mean(swap_sims)) if swap_sims else 0.0
    swap_std = float(np.std(swap_sims)) if swap_sims else 0.0
    t33_elapsed = time.time() - t33
    print(f"    Header-swap cosine similarity: mean={swap_mean:.4f} std={swap_std:.4f} ({t33_elapsed:.0f}s)")

    del model

    probing_results = {
        "row_shuffle": {
            "n_tables": len(shuffle_sims),
            "mean": round(shuffle_mean, 4),
            "std": round(shuffle_std, 4),
            "min": round(shuffle_min, 4),
            "max": round(shuffle_max, 4),
            "time_sec": round(t31_elapsed, 1),
        },
        "header_removal": {
            "cosine_sim_mean": round(removal_mean, 4),
            "cosine_sim_std": round(removal_std, 4),
            "original_metrics": {k: round(v, 4) for k, v in metrics_orig.items()},
            "noheader_metrics": {k: round(v, 4) for k, v in metrics_noheader.items()},
            "r1_change_pp": round((metrics_noheader["R@1"] - metrics_orig["R@1"]) * 100, 1),
            "time_sec": round(t32_elapsed, 1),
        },
        "header_swap": {
            "n_pairs": len(swap_sims),
            "mean": round(swap_mean, 4),
            "std": round(swap_std, 4),
            "time_sec": round(t33_elapsed, 1),
        },
    }

    return probing_results, ranks_orig, q_embs, orig_embs


# ================ EXPERIMENT 4: FAILURE ANALYSIS ==================
def run_experiment4(table_ids, parsed_tables, qas, complexity_list, ranks, q_embs, t_embs):
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: Failure Query Analysis (bge-small, Markdown)")
    print("=" * 80)

    tid_to_idx = {tid: i for i, tid in enumerate(table_ids)}
    qa_gold = [tid_to_idx[qa["table_id"]] for qa in qas]
    top1_indices = get_top1_idx(q_embs, t_embs)

    # Find R@1 failures
    failures = []
    for i, (rank, qa) in enumerate(zip(ranks, qas)):
        if rank > 1:
            failures.append(i)

    print(f"  Total QAs: {len(qas)}, R@1 failures: {len(failures)} ({len(failures) / len(qas) * 100:.1f}%)")

    # Check hierarchical header reference
    hier_ref_count = 0
    for fi in failures:
        qa = qas[fi]
        gold_idx = tid_to_idx[qa["table_id"]]
        parsed = parsed_tables[gold_idx]
        if parsed is None:
            continue
        question_lower = qa["question"].lower()

        # Gather all ancestor (non-leaf) header tokens from the gold table
        ancestor_tokens = set()
        for col_idx, path in parsed["top_paths"].items():
            if len(path) > 1:
                for ancestor in path[:-1]:
                    for word in ancestor.lower().split():
                        if len(word) > 2:
                            ancestor_tokens.add(word)
        for row_idx, path in parsed["left_paths"].items():
            if len(path) > 1:
                for ancestor in path[:-1]:
                    for word in ancestor.lower().split():
                        if len(word) > 2:
                            ancestor_tokens.add(word)

        if any(tok in question_lower for tok in ancestor_tokens):
            hier_ref_count += 1

    hier_pct = hier_ref_count / len(failures) * 100 if failures else 0
    print(f"  Failures referencing hierarchical headers: {hier_ref_count}/{len(failures)} ({hier_pct:.1f}%)")

    # Complexity distribution of failures
    fail_by_group = {"simple": 0, "moderate": 0, "complex": 0}
    total_by_group = {"simple": 0, "moderate": 0, "complex": 0}
    for i, qa in enumerate(qas):
        idx = tid_to_idx[qa["table_id"]]
        g = classify_complexity(complexity_list[idx])
        total_by_group[g] += 1
        if ranks[i] > 1:
            fail_by_group[g] += 1

    print("\n  Failure rate by complexity:")
    for g in ["simple", "moderate", "complex"]:
        tot = total_by_group[g]
        fail = fail_by_group[g]
        rate = fail / tot * 100 if tot > 0 else 0
        print(f"    {g:10s}: {fail}/{tot} failed ({rate:.1f}%)")

    # Top 10 failure examples
    rng = random.Random(SEED)
    example_indices = rng.sample(failures, min(10, len(failures)))
    examples = []
    for fi in sorted(example_indices):
        qa = qas[fi]
        gold_idx = tid_to_idx[qa["table_id"]]
        pred_idx = int(top1_indices[fi])
        gold_cm = complexity_list[gold_idx]
        examples.append({
            "query": qa["question"],
            "gold_table_id": qa["table_id"],
            "gold_title": parsed_tables[gold_idx]["title"] if parsed_tables[gold_idx] else "(parse failed)",
            "pred_table_id": table_ids[pred_idx],
            "pred_title": parsed_tables[pred_idx]["title"] if parsed_tables[pred_idx] else "(parse failed)",
            "gold_complexity": classify_complexity(gold_cm),
            "gold_header_depth": gold_cm["header_depth"],
            "gold_merged": gold_cm["num_merged_cells"],
            "rank": int(ranks[fi]),
        })

    # Print examples
    print(f"\n  Sample failure cases ({len(examples)}):")
    for j, ex in enumerate(examples, 1):
        print(f"\n  [{j}] rank={ex['rank']} | complexity={ex['gold_complexity']} (depth={ex['gold_header_depth']}, merged={ex['gold_merged']})")
        print(f"      Q: {ex['query'][:120]}")
        print(f"      Gold: {ex['gold_table_id']}  -- {ex['gold_title'][:80]}")
        print(f"      Pred: {ex['pred_table_id']}  -- {ex['pred_title'][:80]}")

    failure_analysis = {
        "total_qas": len(qas),
        "total_failures": len(failures),
        "failure_rate_pct": round(len(failures) / len(qas) * 100, 1),
        "hier_ref_in_failures": hier_ref_count,
        "hier_ref_pct": round(hier_pct, 1),
        "fail_by_group": fail_by_group,
        "total_by_group": total_by_group,
        "failure_rate_by_group": {
            g: round(fail_by_group[g] / total_by_group[g] * 100, 1) if total_by_group[g] > 0 else 0
            for g in ["simple", "moderate", "complex"]
        },
    }

    return failure_analysis, examples


# ========================= MAIN ===================================
def main():
    t_start = time.time()
    print("[0/4] Loading all dev data...")
    table_ids, parsed_tables, qas, complexity_list = load_all_dev()

    # ---- Exp 1 ----
    exp1_results, all_model_data, serialized = run_experiment1(table_ids, parsed_tables, qas)

    # ---- Exp 2 ----
    exp2_results, qa_groups = run_experiment2(table_ids, parsed_tables, qas, complexity_list, all_model_data)

    # Free large model data except bge-small
    for mname in list(all_model_data.keys()):
        if mname != "bge-small":
            del all_model_data[mname]

    # ---- Exp 3 ----
    exp3_results, bge_ranks, bge_q_embs, bge_t_embs = run_experiment3(
        table_ids, parsed_tables, qas, serialized)

    # ---- Exp 4 ----
    exp4_results, exp4_examples = run_experiment4(
        table_ids, parsed_tables, qas, complexity_list,
        bge_ranks, bge_q_embs, bge_t_embs)

    total_time = time.time() - t_start

    # ======================== KEY FINDINGS ============================
    print("\n" + "=" * 80)
    print("===== KEY FINDINGS =====")
    # 1. Embedder gap
    best_model = max(exp1_results, key=lambda m: exp1_results[m]["R@1"])
    worst_model = min(exp1_results, key=lambda m: exp1_results[m]["R@1"])
    r1_diff = (exp1_results[best_model]["R@1"] - exp1_results[worst_model]["R@1"]) * 100
    print(f"1. Embedder gap: {best_model} vs {worst_model} R@1 diff = {r1_diff:.1f}pp")

    # 2. Complexity effect
    simple_r1 = exp2_results.get(("bge-small", "simple"), {}).get("R@1", 0) * 100
    complex_r1 = exp2_results.get(("bge-small", "complex"), {}).get("R@1", 0) * 100
    print(f"2. Complexity effect (bge-small): simple R@1 = {simple_r1:.1f}% vs complex R@1 = {complex_r1:.1f}% (diff = {simple_r1 - complex_r1:.1f}pp)")

    # 3. Row shuffle
    rs = exp3_results["row_shuffle"]
    encodes_order = "encodes" if rs["mean"] < 0.95 else "does NOT encode"
    print(f"3. Row shuffle similarity: mean={rs['mean']:.4f} -> embedder {encodes_order} row order")

    # 4. Header removal
    hr = exp3_results["header_removal"]
    header_important = "significant" if abs(hr["r1_change_pp"]) > 3 else "negligible"
    print(f"4. Header removal R@1 change: {hr['r1_change_pp']:+.1f}pp (header info {header_important})")

    # 5. Header swap
    hs = exp3_results["header_swap"]
    encodes_hd = "encodes" if hs["mean"] < 0.90 else "does NOT encode"
    print(f"5. Header swap similarity: mean={hs['mean']:.4f} -> header-data link {encodes_hd}")

    # 6. Hierarchical reference
    print(f"6. Failures referencing hierarchical headers: {exp4_results['hier_ref_pct']:.1f}%")

    print(f"\nTotal experiment time: {total_time:.0f}s ({total_time / 60:.1f}min)")
    print("=" * 80)

    # ======================== SAVE FILES ==============================
    # embedding_diagnostic.json
    exp2_serializable = {}
    for (mname, g), v in exp2_results.items():
        exp2_serializable[f"{mname}_{g}"] = v
    diag_json = {
        "experiment1_multi_embedder": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in exp1_results.items()},
        "experiment2_stratified": exp2_serializable,
        "experiment4_failure_analysis": exp4_results,
        "total_time_sec": round(total_time, 1),
    }
    with open(OUTPUT_DIR / "embedding_diagnostic.json", "w", encoding="utf-8") as f:
        json.dump(diag_json, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {OUTPUT_DIR / 'embedding_diagnostic.json'}")

    # complexity_breakdown.csv
    csv_path = OUTPUT_DIR / "complexity_breakdown.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv_mod.writer(f)
        writer.writerow(["model", "complexity", "n_queries", "R@1", "R@3", "R@5", "R@10", "MRR"])
        for mname in [m["name"] for m in MODELS]:
            for g in ["simple", "moderate", "complex"]:
                key = (mname, g)
                if key in exp2_results:
                    m = exp2_results[key]
                    writer.writerow([mname, g, m["n_queries"],
                                     round(m["R@1"], 4), round(m.get("R@3", 0), 4),
                                     round(m["R@5"], 4), round(m["R@10"], 4), round(m["MRR"], 4)])
    print(f"Saved -> {csv_path}")

    # probing_results.json
    with open(OUTPUT_DIR / "probing_results.json", "w", encoding="utf-8") as f:
        json.dump(exp3_results, f, ensure_ascii=False, indent=2)
    print(f"Saved -> {OUTPUT_DIR / 'probing_results.json'}")

    # failure_examples.txt
    fe_path = OUTPUT_DIR / "failure_examples.txt"
    with open(fe_path, "w", encoding="utf-8") as f:
        f.write("FAILURE QUERY EXAMPLES (bge-small, Markdown)\n")
        f.write(f"Total QAs: {exp4_results['total_qas']}, Failures (R@1): {exp4_results['total_failures']}\n")
        f.write(f"Hier-ref in failures: {exp4_results['hier_ref_pct']:.1f}%\n")
        f.write("=" * 80 + "\n\n")
        for j, ex in enumerate(exp4_examples, 1):
            f.write(f"[{j}] rank={ex['rank']} | complexity={ex['gold_complexity']} "
                    f"(depth={ex['gold_header_depth']}, merged={ex['gold_merged']})\n")
            f.write(f"  Query: {ex['query']}\n")
            f.write(f"  Gold:  {ex['gold_table_id']}  -- {ex['gold_title']}\n")
            f.write(f"  Pred:  {ex['pred_table_id']}  -- {ex['pred_title']}\n\n")
    print(f"Saved -> {fe_path}")


if __name__ == "__main__":
    main()
