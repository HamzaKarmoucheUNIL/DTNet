"""robustness_viz.py — Two thesis figures from results/robustness_results.json.

Fig 1: grouped bar chart (networked vs isolated, mean±std per metric).
Fig 2: per-seed line plots showing metric stability across 5 random seeds.
Usage: python -m src.viz.robustness_viz
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

BG: str = "white"
C_NET: str = "#4A9EFF"
C_ISO: str = "#888888"
METRICS: List[str] = ["mae", "rmse", "f1", "precision", "recall"]
METRIC_LABELS: List[str] = ["MAE", "RMSE", "F1", "Precision", "Recall"]
JSON_PATH: Path = Path("results/robustness_results.json")
FIG1_PATH: Path = Path("results/fig_robustness_comparison.png")
FIG2_PATH: Path = Path("results/fig_seeds_stability.png")

matplotlib.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": "#CCCCCC", "axes.labelcolor": "#333333",
    "xtick.color": "#333333", "ytick.color": "#333333",
    "text.color": "#333333", "grid.color": "#EEEEEE",
    "grid.alpha": 0.8, "legend.facecolor": "#F8F8F8",
    "legend.edgecolor": "#CCCCCC",
})


def _err_kw() -> Dict[str, Any]:
    """Return shared error-bar style kwargs."""
    return dict(ecolor="#555555", elinewidth=1.2, capsize=4, capthick=1.2)


def plot_comparison(summary: Dict[str, Any]) -> None:
    """Figure 1: grouped bar chart of mean±std for networked vs isolated.

    Args:
        summary: ``data["summary"]`` from robustness_results.json.
    """
    net_means = np.array([summary[m]["networked_mean"] for m in METRICS])
    net_stds = np.array([summary[m]["networked_std"] for m in METRICS])
    iso_means = np.array([summary[m]["isolated_mean"] for m in METRICS])
    iso_stds = np.array([summary[m]["isolated_std"] for m in METRICS])

    x = np.arange(len(METRICS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, net_means, w, yerr=net_stds,
           label="Networked (DTNetGNN)", color=C_NET, **_err_kw())
    ax.bar(x + w / 2, iso_means, w, yerr=iso_stds,
           label="Isolated Baseline", color=C_ISO, **_err_kw())

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS, fontsize=12)
    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("DTNet Networked vs Isolated — Performance Comparison\n(mean ± std over 5 random seeds)", fontsize=13, pad=14)
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    FIG1_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG1_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[robustness_viz] Saved Figure 1 → {FIG1_PATH}")


def plot_seeds_stability(seeds: List[int], per_seed: List[Dict[str, Any]]) -> None:
    """Figure 2: 2×3 subplot grid — one metric per cell, last cell = legend.

    Args:
        seeds: Seed values used as x-axis tick labels.
        per_seed: ``data["per_seed"]`` from robustness_results.json.
    """
    x = np.arange(len(seeds))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        "Performance Across Random Seeds — DTNet Networked vs Isolated",
        fontsize=13, y=1.01,
    )

    net_lines, iso_lines = None, None
    for i, (m, label) in enumerate(zip(METRICS, METRIC_LABELS)):
        ax = axes.flat[i]
        net_vals = [r["networked"][m] for r in per_seed]
        iso_vals = [r["isolated"][m]  for r in per_seed]

        net_lines, = ax.plot(
            x, net_vals, color=C_NET, marker="o", linewidth=1.8,
            markersize=5, label="Networked (DTNetGNN)",
        )
        iso_lines, = ax.plot(
            x, iso_vals, color=C_ISO, marker="s", linestyle="--",
            linewidth=1.8, markersize=5, label="Isolated Baseline",
        )
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Seed", fontsize=10)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in seeds], fontsize=8, rotation=15)
        ax.yaxis.grid(True, linestyle="--")
        ax.set_axisbelow(True)

    # Use the 6th cell for the shared legend
    legend_ax = axes.flat[5]
    legend_ax.set_visible(False)
    if net_lines and iso_lines:
        fig.legend(
            [net_lines, iso_lines],
            ["Networked (DTNetGNN)", "Isolated Baseline"],
            loc="center",
            bbox_to_anchor=(0.83, 0.28),
            fontsize=11,
            framealpha=0.8,
        )

    fig.tight_layout()
    FIG2_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG2_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[robustness_viz] Saved Figure 2 → {FIG2_PATH}")


def main() -> None:
    """Load robustness_results.json and produce both thesis figures."""
    if not JSON_PATH.exists():
        raise FileNotFoundError(
            f"[robustness_viz] {JSON_PATH} not found. "
            "Run python -m src.gnn.evaluate_robust first."
        )
    with open(JSON_PATH) as fh:
        data: Dict[str, Any] = json.load(fh)

    print(f"[robustness_viz] Loaded {JSON_PATH}  (seeds={data['seeds']})")
    plot_comparison(data["summary"])
    plot_seeds_stability(data["seeds"], data["per_seed"])
    print("[robustness_viz] Done.")

if __name__ == "__main__":
    main()
