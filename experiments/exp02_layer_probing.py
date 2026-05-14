# SPDX-License-Identifier: MIT
"""Experiment 02 — Layer-wise probing of table-structure knowledge.

For each embedder (BGE-small, E5-small), extract hidden states from every
transformer layer, then train linear / MLP probes on three structural
tasks.  Reports accuracy + selectivity per layer and produces Tenney-2019
style layer curves.

Usage:
    uv run python experiments/exp02_layer_probing.py
    uv run python experiments/exp02_layer_probing.py smoke_test=true
"""

from __future__ import annotations

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

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.io.hitab_loader import load_tables
from src.io.table_schema import Table
from src.probing.extract_hidden import HiddenStateExtractor, PoolingStrategy
from src.probing.probe_classifier import ProbeResult, train_probe
from src.probing.probe_tasks import (
    ProbeDataset,
    build_cell_coord_task,
    build_parent_header_task,
    build_same_row_task,
)
from src.serializers.html_ser import HtmlSerializer
from src.utils.logging import get_logger
from src.utils.seed import set_seed

matplotlib.use("Agg")

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Task builder dispatch
# ---------------------------------------------------------------------------

TASK_BUILDERS: dict[
    str,
    Callable[..., ProbeDataset],
] = {
    "parent_header": build_parent_header_task,
    "cell_coord": build_cell_coord_task,
    "same_row": build_same_row_task,
}


def _collect_cell_texts(tables: list[Table]) -> list[str]:
    """Gather unique cell value strings across all tables."""
    texts: set[str] = set()
    for table in tables:
        for row in table.cells:
            for cell in row:
                if cell.value:
                    texts.add(cell.value)
    return sorted(texts)


def _build_hidden_map(
    texts: list[str],
    all_hidden: np.ndarray,
) -> dict[str, np.ndarray]:
    """Map text → hidden states array of shape (n_layers+1, dim)."""
    return {t: all_hidden[i] for i, t in enumerate(texts)}


def _build_task_dataset(
    task_name: str,
    tables: list[Table],
    hidden_map: dict[str, np.ndarray],
    layer: int,
    cfg: DictConfig,
) -> ProbeDataset:
    """Dispatch to the correct task builder with config params."""
    if task_name == "parent_header":
        return build_parent_header_task(
            tables, hidden_map, layer=layer,
            n_classes=cfg.task_params.parent_header.n_classes,
            seed=cfg.seed,
        )
    if task_name == "cell_coord":
        return build_cell_coord_task(
            tables, hidden_map, layer=layer,
            n_row_buckets=cfg.task_params.cell_coord.n_row_buckets,
            n_col_buckets=cfg.task_params.cell_coord.n_col_buckets,
            seed=cfg.seed,
        )
    if task_name == "same_row":
        return build_same_row_task(
            tables, hidden_map, layer=layer,
            max_pairs_per_table=cfg.task_params.same_row.max_pairs_per_table,
            seed=cfg.seed,
        )
    msg = f"Unknown task: {task_name}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="configs", config_name="probing_default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the layer-wise probing experiment."""
    set_seed(cfg.seed)
    log.info("Starting exp02_layer_probing")

    project_root = _PROJECT_ROOT
    raw_dir = project_root / cfg.data.raw_dir
    if cfg.data.table_subdir:
        raw_dir = raw_dir / cfg.data.table_subdir

    # ---- Load & sample tables ----
    all_tables = load_tables(raw_dir)
    log.info("Loaded %d tables", len(all_tables))
    if not all_tables:
        log.error("No tables found in %s", raw_dir)
        sys.exit(1)

    n = cfg.smoke_test_n if cfg.smoke_test else cfg.n_tables
    n = min(n, len(all_tables))
    rng = np.random.default_rng(cfg.seed)
    indices = rng.choice(len(all_tables), size=n, replace=False)
    tables = [all_tables[i] for i in indices]
    log.info("Sampled %d tables (smoke_test=%s)", n, cfg.smoke_test)

    # ---- Collect unique cell texts ----
    cell_texts = _collect_cell_texts(tables)
    log.info("Unique cell texts: %d", len(cell_texts))

    results: list[dict[str, Any]] = []

    for model_key in cfg.models:
        log.info("=== Model: %s ===", model_key)

        # ---- Extract hidden states ----
        extractor = HiddenStateExtractor(
            model_key=model_key,
            pooling=cfg.pooling,
            device="cpu",
        )
        n_layers = extractor.n_layers  # transformer layers (excl. embedding)
        log.info("Extracting hidden states for %d texts (%d layers)...", len(cell_texts), n_layers)

        all_hidden = extractor.extract(cell_texts, batch_size=cfg.batch_size)
        # all_hidden: (n_texts, n_layers+1, hidden_dim)
        hidden_map = _build_hidden_map(cell_texts, all_hidden)
        log.info("Hidden states shape: %s", all_hidden.shape)

        # ---- Probe grid: layer × task × probe_type ----
        total_layers = n_layers + 1  # 0=embedding, 1..n_layers=transformer
        for layer in tqdm(range(total_layers), desc=f"{model_key} layers"):
            for task_name in cfg.tasks:
                ds = _build_task_dataset(task_name, tables, hidden_map, layer, cfg)
                if ds.X.shape[0] < 10:
                    log.warning(
                        "Skipping %s layer %d: only %d samples",
                        task_name, layer, ds.X.shape[0],
                    )
                    continue

                for probe_type in cfg.probe_types:
                    result = train_probe(
                        ds,
                        layer=layer,
                        probe_type=probe_type,
                        test_size=cfg.probe_params.test_size,
                        seed=cfg.seed,
                    )
                    results.append({
                        "model": model_key,
                        "layer": layer,
                        "task": task_name,
                        "probe_type": probe_type,
                        "accuracy": round(result.accuracy, 4),
                        "control_accuracy": round(result.control_accuracy, 4),
                        "selectivity": round(result.selectivity, 4),
                        "n_train": result.n_train,
                        "n_test": result.n_test,
                    })

        # Free memory.
        del extractor, all_hidden, hidden_map

    # ---- Save CSV ----
    df = pd.DataFrame(results)
    csv_path = project_root / cfg.output.csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    log.info("Results saved to %s", csv_path)

    # ---- Tenney-style layer curve plot ----
    _plot_layer_curves(df, project_root / cfg.output.layer_curve_path)
    log.info("Layer curve plot saved.")
    log.info("exp02 complete.")


def _plot_layer_curves(df: pd.DataFrame, save_path: Path) -> None:
    """Draw Tenney-2019 style accuracy-vs-layer line plots."""
    if df.empty:
        return

    models = df["model"].unique()
    tasks = df["task"].unique()
    probe_types = df["probe_type"].unique()

    n_tasks = len(tasks)
    fig, axes = plt.subplots(
        1, n_tasks,
        figsize=(6 * n_tasks, 5),
        sharey=True,
    )
    if n_tasks == 1:
        axes = [axes]  # type: ignore[list-item]

    n_series = len(models) * len(probe_types)
    palette = sns.color_palette("tab10", n_colors=max(n_series, 1))

    for ax, task in zip(axes, tasks):
        color_idx = 0
        for model in models:
            for pt in probe_types:
                sub = df[(df["model"] == model) & (df["task"] == task) & (df["probe_type"] == pt)]
                if sub.empty:
                    continue
                sub_sorted = sub.sort_values("layer")
                label = f"{model} / {pt}"
                c = palette[color_idx % len(palette)]
                ax.plot(
                    sub_sorted["layer"],
                    sub_sorted["accuracy"],
                    marker="o",
                    markersize=4,
                    label=label,
                    color=c,
                )
                # Selectivity as dashed line.
                ax.plot(
                    sub_sorted["layer"],
                    sub_sorted["selectivity"],
                    linestyle="--",
                    alpha=0.5,
                    color=c,
                )
                color_idx += 1

        ax.set_title(task, fontsize=12)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Score")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Layer-wise Probing — Accuracy (solid) & Selectivity (dashed)",
        fontsize=14,
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
