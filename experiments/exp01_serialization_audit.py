# SPDX-License-Identifier: MIT
"""Experiment 01 — Serialization damage audit.

Evaluates 5 serializers × 4 metrics on a random sample of HiTab tables.
Outputs CSV + Markdown summary tables, a box-plot PDF, and paired
bootstrap confidence intervals.

Usage:
    uv run python experiments/exp01_serialization_audit.py
    uv run python experiments/exp01_serialization_audit.py smoke_test=true
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig
from tqdm import tqdm

# Ensure the project root is on sys.path so that `src` is importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.io.hitab_loader import load_tables
from src.io.table_schema import Table
from src.metrics.cell_coord_preserve import cell_coord_preservation
from src.metrics.header_path_acc import header_path_accuracy
from src.metrics.merged_cell_recovery import merged_cell_recovery
from src.metrics.teds import teds
from src.serializers.csv_ser import CsvSerializer
from src.serializers.html_ser import HtmlSerializer
from src.serializers.json_tree_ser import JsonTreeSerializer
from src.serializers.markdown_ser import MarkdownSerializer
from src.serializers.otsl_ser import OtslSerializer
from src.utils.logging import get_logger
from src.utils.seed import set_seed

matplotlib.use("Agg")  # non-interactive backend

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SERIALIZER_MAP: dict[str, Callable[[], Any]] = {
    "html": HtmlSerializer,
    "markdown": MarkdownSerializer,
    "csv": CsvSerializer,
    "json_tree": JsonTreeSerializer,
    "otsl": OtslSerializer,
}

METRIC_MAP: dict[str, Callable[[Table, Table], float]] = {
    "teds": teds,
    "header_path_accuracy": header_path_accuracy,
    "cell_coord_preservation": cell_coord_preservation,
    "merged_cell_recovery": merged_cell_recovery,
}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate_one(
    table: Table,
    serializer_name: str,
    metric_funcs: dict[str, Callable[[Table, Table], float]],
) -> dict[str, float]:
    """Serialize → parse → measure all metrics for one table."""
    try:
        ser = SERIALIZER_MAP[serializer_name]()
        text = ser.serialize(table)
        recovered = ser.parse(text)
        results: dict[str, float] = {}
        for m_name, m_fn in metric_funcs.items():
            try:
                results[m_name] = m_fn(table, recovered)
            except Exception:
                results[m_name] = float("nan")
        return results
    except Exception:
        return {m_name: float("nan") for m_name in metric_funcs}


def _paired_bootstrap_ci(
    scores: np.ndarray,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via paired bootstrap.

    Args:
        scores: 1-D array of per-table scores for one (serializer, metric).
        n_resamples: Number of bootstrap resamples.
        confidence: Confidence level.
        rng: Numpy random generator.

    Returns:
        Tuple of (mean, ci_lower, ci_upper).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    # Drop NaN values (e.g. TEDS skipped for large tables).
    scores = scores[~np.isnan(scores)]
    if len(scores) == 0:
        return float("nan"), float("nan"), float("nan")
    n = len(scores)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = scores[idx].mean()
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.percentile(means, 100 * alpha))
    hi = float(np.percentile(means, 100 * (1 - alpha)))
    return float(scores.mean()), lo, hi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="configs", config_name="audit_default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the serialisation audit experiment."""
    set_seed(cfg.seed)
    log.info("Starting exp01_serialization_audit")

    # Resolve paths relative to project root (Hydra changes cwd).
    project_root = _PROJECT_ROOT
    raw_dir = project_root / cfg.data.raw_dir
    if cfg.data.table_subdir:
        raw_dir = raw_dir / cfg.data.table_subdir

    # ---- Load tables ----
    log.info("Loading tables from %s", raw_dir)
    all_tables = load_tables(raw_dir)
    log.info("Loaded %d tables total", len(all_tables))

    if len(all_tables) == 0:
        log.error(
            "No tables found in %s. Please download HiTab data first "
            "(see README.md).",
            raw_dir,
        )
        sys.exit(1)

    # ---- Sample ----
    n = cfg.smoke_test_n if cfg.smoke_test else cfg.n_tables
    n = min(n, len(all_tables))
    rng = np.random.default_rng(cfg.seed)
    indices = rng.choice(len(all_tables), size=n, replace=False)
    tables = [all_tables[i] for i in indices]
    log.info("Sampled %d tables (smoke_test=%s)", n, cfg.smoke_test)

    # ---- Build metric function dict ----
    metric_funcs = {m: METRIC_MAP[m] for m in cfg.metrics}

    # ---- Evaluate grid ----
    records: list[dict[str, Any]] = []
    for tbl_idx, table in enumerate(tqdm(tables, desc="Evaluating")):
        for s_name in cfg.serializers:
            scores = _evaluate_one(table, s_name, metric_funcs)
            record: dict[str, Any] = {
                "table_idx": tbl_idx,
                "table_id": table.metadata.get("table_id", f"table_{tbl_idx}"),
                "serializer": s_name,
            }
            record.update(scores)
            records.append(record)

    df = pd.DataFrame(records)

    # ---- Aggregate with bootstrap CI ----
    summary_rows: list[dict[str, Any]] = []
    for s_name in cfg.serializers:
        row: dict[str, Any] = {"serializer": s_name}
        sub = df[df["serializer"] == s_name]
        for m_name in cfg.metrics:
            arr = sub[m_name].to_numpy()
            mean, lo, hi = _paired_bootstrap_ci(
                arr,
                n_resamples=cfg.bootstrap.n_resamples,
                confidence=cfg.bootstrap.confidence_level,
                rng=np.random.default_rng(cfg.seed),
            )
            row[f"{m_name}_mean"] = round(mean, 4)
            row[f"{m_name}_ci_lo"] = round(lo, 4)
            row[f"{m_name}_ci_hi"] = round(hi, 4)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # ---- Save outputs ----
    csv_path = project_root / cfg.output.csv_path
    md_path = project_root / cfg.output.markdown_path
    plot_path = project_root / cfg.output.boxplot_path

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(csv_path, index=False)
    log.info("Raw results saved to %s", csv_path)

    # Markdown summary.
    md_text = summary_df.to_markdown(index=False)
    md_path.write_text(md_text, encoding="utf-8")
    log.info("Summary table saved to %s", md_path)
    log.info("\n%s", md_text)

    # ---- Box plot ----
    metric_names = list(cfg.metrics)
    fig, axes = plt.subplots(1, len(metric_names), figsize=(5 * len(metric_names), 5))
    if len(metric_names) == 1:
        axes = [axes]  # type: ignore[list-item]
    for ax, m_name in zip(axes, metric_names):
        sns.boxplot(data=df, x="serializer", y=m_name, ax=ax)
        ax.set_title(m_name)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Serialization Audit — Metric Distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Box plot saved to %s", plot_path)

    log.info("exp01 complete.")


if __name__ == "__main__":
    main()
