# SPDX-License-Identifier: MIT
"""Strict Recall — 리더에게 실제로 넘긴 셀 집합 전체를 K 없이 채점한다.

G = 질문의 gold 셀, R = 시스템이 최종 선택해 리더 문맥에 넣은 셀 전부. 평가자는 R 을
다시 자르지 않는다 — top-K 후보 목록이 아니라 최종 evidence pack 이 R 이다.

    TP = |G ∩ R|   FP = |R - G|   FN = |G - R|
    Strict Recall = [G ⊆ R]   Exact Set Match = [G == R]

R 이 비면 precision·recall·F1·Strict Recall 이 모두 0 이다. G 는 비어 있으면 안 된다.

헤더 gold (HiTab 에서 답이 헤더인 질문): 셀 색인에 헤더라는 단위가 없고 헤더는 그 아래
데이터 셀의 문장 경로에 실려 배달된다. ``carriers[g]`` 가 헤더 g 를 싣는 데이터 셀이다.
R 이 그중 하나를 담으면 g 는 회수된 것이고, 그 데이터 셀은 무관 셀(FP)로 세지 않는다.
carriers 가 없으면 위 식 그대로다.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

from .artifacts import file_digest, write_pair

TYPES = ("single_cell", "multi_cell", "arithmetic")
SUFFIXES = ("_per_query.jsonl", "_summary.json", "_summary.csv", "_anomalies.jsonl",
            "_excluded.jsonl")


def question_type(arithmetic: bool, n_gold: int) -> str:
    return "arithmetic" if arithmetic else "single_cell" if n_gold == 1 else "multi_cell"


def score(gold, pred, carriers=None) -> dict:
    gold, pred, carriers = set(gold), set(pred), carriers or {}
    if not gold:
        raise ValueError("gold must be non-empty")
    hit = {g for g in gold if g in pred or pred & carriers.get(g, set())}
    carried = set().union(*(carriers.get(g, set()) for g in gold))
    relevant = {r for r in pred if r in gold or r in carried}
    p = len(relevant) / len(pred) if pred else 0.0
    rec = len(hit) / len(gold)
    return {"tp": len(hit), "tp_predicted": len(relevant),
            "fp": len(pred) - len(relevant), "fn": len(gold) - len(hit),
            "precision": p, "recall": rec, "f1": 2 * p * rec / (p + rec) if p + rec else 0.0,
            "strict_recall": int(hit == gold),
            "exact_set_match": int(hit == gold and relevant == pred),
            "missing_gold_cells": sorted(gold - hit, key=str),
            "irrelevant_predicted_cells": sorted(pred - relevant, key=str)}


def row(dataset, split, query_id, qtype, gold, pred, carriers=None, table_of=None,
        table_metrics=False, **extra) -> dict:
    """질문 하나의 레코드. ``table_metrics`` 는 셀에서 역산한 표 집합(Derived Table Coverage)."""
    gold, pred = set(gold), set(pred)
    out = {"query_id": query_id, "dataset": dataset, "split": split, "question_type": qtype,
           "gold_cells": sorted(gold, key=str), "predicted_cells": sorted(pred, key=str),
           **score(gold, pred, carriers),
           "num_gold_cells": len(gold), "num_predicted_cells": len(pred)}
    gt = sorted({table_of(c) for c in gold}) if table_of else []
    pt = sorted({table_of(c) for c in pred}) if table_of else []
    out["gold_tables"], out["predicted_tables"] = gt, pt
    if table_metrics:
        t = score(gt, pt)
        out["derived_table_coverage"] = {k: t[k] for k in ("strict_recall", "precision",
                                                           "recall", "f1")}
    return {**out, **extra}


def summarize(rows) -> dict:
    n = len(rows)
    if not n:
        return {"N": 0}

    def mean(key):
        return sum(r[key] for r in rows) / n

    npred = sorted(r["num_predicted_cells"] for r in rows)
    tp, tpr, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "tp_predicted", "fp", "fn"))
    mp = tpr / (tpr + fp) if tpr + fp else 0.0
    mr = tp / (tp + fn)
    out = {"N": n, "strict_recall": mean("strict_recall"),
           "exact_set_match": mean("exact_set_match"),
           "macro_precision": mean("precision"), "macro_recall": mean("recall"),
           "macro_f1": mean("f1"), "micro_precision": mp, "micro_recall": mr,
           "micro_f1": 2 * mp * mr / (mp + mr) if mp + mr else 0.0,
           "tp": tp, "fp": fp, "fn": fn,
           "mean_predicted_cells": mean("num_predicted_cells"),
           "median_predicted_cells": statistics.median(npred),
           "p95_predicted_cells": npred[math.ceil(0.95 * n) - 1],       # nearest rank
           "mean_gold_cells": mean("num_gold_cells"),
           "gold_to_predicted_cell_ratio": (sum(r["num_gold_cells"] for r in rows) / sum(npred)
                                            if sum(npred) else None)}
    tokens = [r["context_tokens"] for r in rows if r.get("context_tokens") is not None]
    if len(tokens) == n:
        out["mean_context_tokens"] = sum(tokens) / n
        out["median_context_tokens"] = statistics.median(tokens)
    if all("derived_table_coverage" in r for r in rows):
        for k in ("strict_recall", "precision", "recall", "f1"):
            out[f"derived_table_{k}"] = sum(r["derived_table_coverage"][k] for r in rows) / n
    return out


def outputs(out_dir, tag):
    return [Path(out_dir) / f"{tag}{s}" for s in SUFFIXES]


def write(out_dir, tag, rows, groups, meta, anomalies=(), excluded=()) -> dict:
    """질문별 JSONL, 그룹별 요약 JSON·CSV, gold 이상 로그, 제외 목록. 덮어쓰지 않는다."""
    paths = outputs(out_dir, tag)
    if any(p.exists() for p in paths):
        raise FileExistsError(f"strict_no_k output exists for tag {tag}")
    blocks = {name: summarize([r for r in rows if keep(r)]) for name, keep in groups.items()}
    summary = {**meta, "groups": blocks, "n_gold_anomalies": len(anomalies),
               "n_excluded": len(excluded)}
    write_pair(paths[0], rows, summary, summary_path=paths[1])
    cols = list(dict.fromkeys(k for b in blocks.values() for k in b))
    with paths[2].open("x", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, ["group", *cols])
        w.writeheader()
        w.writerows({"group": name, **b} for name, b in blocks.items())
    for path, items in ((paths[3], anomalies), (paths[4], excluded)):
        with path.open("x", encoding="utf-8", newline="\n") as fh:
            fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in items)
    return summary


def token_counter(name: str):
    """리더 토크나이저로 문맥 블록(``"\\n".join(context)``)의 토큰 수를 센다."""
    if not name:
        return None, None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    info = {"name": name, "revision": tok.init_kwargs.get("_commit_hash"),
            "counts": "tokens of '\\n'.join(context), no special tokens, no prompt template"}
    return (lambda texts: len(tok("\n".join(texts), add_special_tokens=False)["input_ids"])), info


# 예산 없는 선택(strict_no_k)의 임계값. 규칙은 dev 결과를 보기 전에 여기 고정했다.
GRID = tuple(i / 100 for i in range(101))
THRESHOLD_METHOD = ("grid 0.00..1.00 step 0.01 on the selection split; the threshold with the "
                    "highest macro F1 wins, ties go to the higher threshold; a threshold that "
                    "returns every candidate unit is refused; never selected on test")


def sweep(cases, grid=GRID) -> list:
    """``cases`` = ``[(gold, carriers, [(score, cells), ...]), ...]`` — 임계값마다 요약 지표."""
    out = []
    n_candidates = sum(len(set().union(*(c for _s, c in ranked))) for _g, _c, ranked in cases)
    for tau in grid:
        rows = []
        for gold, carriers, ranked in cases:
            pred = set().union(*(c for s, c in ranked if s >= tau))
            rows.append({**score(gold, pred, carriers), "num_gold_cells": len(set(gold)),
                         "num_predicted_cells": len(pred)})
        out.append({"threshold": tau, **summarize(rows),
                    "mean_candidate_cells": n_candidates / len(cases)})
    return out


def choose_threshold(sweep_rows) -> float:
    ok = [r for r in sweep_rows if r["mean_predicted_cells"] < r["mean_candidate_cells"]]
    if not ok:
        raise ValueError("every threshold returns all candidate units")
    return max(ok, key=lambda r: (r["macro_f1"], r["threshold"]))["threshold"]


def selected_threshold(sweep_rows, split, score_definition, config) -> dict:
    if split == "test":
        raise ValueError("the threshold is never selected on test")
    tau = choose_threshold(sweep_rows)
    return {"role": "selected_here", "method": THRESHOLD_METHOD, "score": score_definition,
            "config": config, "selected_on_split": split, "selected_threshold": tau,
            "applied_threshold": tau, "sweep": sweep_rows}


def load_threshold(path, dataset, split, config) -> dict:
    """다른 분할에서 고른 임계값을 그대로 가져온다. 같은 분할·다른 점수 설정이면 거부한다."""
    s = json.loads(Path(path).read_text(encoding="utf-8"))
    t = s.get("threshold") or {}
    if s.get("dataset") != dataset or t.get("role") != "selected_here":
        raise ValueError(f"{path}: not a {dataset} threshold-selection summary")
    if t["selected_on_split"] == split:
        raise ValueError("the threshold was selected on this split; apply it to another split")
    if t["config"] != config:
        raise ValueError(f"threshold score config differs: {t['config']} vs {config}")
    return {**{k: t[k] for k in ("method", "score", "config", "selected_on_split",
                                 "selected_threshold")},
            "role": "applied_from_file", "applied_threshold": t["selected_threshold"],
            "source_summary": str(path), "source_summary_sha256": file_digest(path)}


def print_groups(blocks) -> None:
    f = lambda x: "n/a" if x is None else f"{x:.4f}" if isinstance(x, float) else str(x)
    for name, b in blocks.items():
        if not b["N"]:
            print(f"{name}: N=0")
            continue
        line = (f"{name}: N={b['N']}  Strict Recall={f(b['strict_recall'])}  "
                f"Exact Set Match={f(b['exact_set_match'])}  "
                f"macro P/R/F1={f(b['macro_precision'])}/{f(b['macro_recall'])}/{f(b['macro_f1'])}  "
                f"micro P/R/F1={f(b['micro_precision'])}/{f(b['micro_recall'])}/{f(b['micro_f1'])}  "
                f"pred cells mean/median/p95={f(b['mean_predicted_cells'])}/"
                f"{f(b['median_predicted_cells'])}/{f(b['p95_predicted_cells'])}  "
                f"gold cells mean={f(b['mean_gold_cells'])}")
        if "mean_context_tokens" in b:
            line += (f"  context tokens mean/median={f(b['mean_context_tokens'])}/"
                     f"{f(b['median_context_tokens'])}")
        if "derived_table_strict_recall" in b:
            line += (f"  Derived Table Coverage strict/P/R/F1={f(b['derived_table_strict_recall'])}/"
                     f"{f(b['derived_table_precision'])}/{f(b['derived_table_recall'])}/"
                     f"{f(b['derived_table_f1'])}")
        print(line)
