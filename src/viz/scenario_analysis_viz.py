"""scenario_analysis_viz.py — Cascade comparison across 3 disruption scenarios.

Runs the 3 new Phase-6 scenarios and produces a 1×3 subplot figure + summary table.
Usage: python -m src.viz.scenario_analysis_viz
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from src.simulation.model import DTNetModel
from src.simulation.scenarios import (
    scenario_bottleneck_plant,
    scenario_critical_hub_failure,
    scenario_supplier_cascade,
)

np.random.seed(42)
torch.manual_seed(42)

BG: str = "white"
C_DISRUPTED: str = "#e74c3c"
C_HEALTH: str = "#2ecc71"
CSV_FILENAME: str = "updated_data.csv"
N_STEPS: int = 15
SAVE_PATH: Path = Path("results/fig_scenario_comparison.png")

matplotlib.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": "#CCCCCC", "axes.labelcolor": "#333333",
    "xtick.color": "#333333", "ytick.color": "#333333",
    "text.color": "#333333", "grid.color": "#EEEEEE",
    "grid.alpha": 0.8, "legend.facecolor": "#F8F8F8",
    "legend.edgecolor": "#CCCCCC",
})


def _build_graph() -> nx.DiGraph:
    """Load raw CSV and build the DTNet DiGraph (twin + layer attrs on every node)."""
    from src.data.loader import load_csv
    from src.data.preprocess import preprocess
    from src.data.entity_mapping import build_entity_mappings
    from src.graph.topology import infer_topology
    from src.graph.builder import build_graph

    with contextlib.redirect_stdout(io.StringIO()):
        df_raw = load_csv(CSV_FILENAME)
        df_clean, _ = preprocess(df_raw)
        em = build_entity_mappings(df_raw)
        nodes, edges = infer_topology(em)
        G = build_graph(nodes, edges, df_clean)
    print(f"[scenario_viz] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def _reset_twins(G: nx.DiGraph) -> None:
    """Reset every twin agent to baseline state before a new run (COMMON_MISTAKES #10)."""
    for _, data in G.nodes(data=True):
        data["twin"].reset()


def _run_scenario(G: nx.DiGraph, scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Reset twins, create a fresh DTNetModel, inject disruption, run, return history.

    Args:
        G: DTNet DiGraph; twin states reset in-place before each run (COMMON_MISTAKES #10).
        scenario: Dict with ``disrupted_nodes`` and ``severity``.
    """
    _reset_twins(G)
    model = DTNetModel(G)
    for node_id in scenario["disrupted_nodes"]:
        model.inject_disruption(node_id, scenario["severity"])
    for _ in range(N_STEPS):
        model.step()
    return model.get_history()


def _time_to_50pct(history: List[Dict[str, Any]], n_nodes: int) -> Optional[int]:
    """Return first timestep where disrupted nodes ≥ 50% of total, else None."""
    threshold: float = 0.5 * n_nodes
    for record in history:
        if len(record["total_disrupted"]) >= threshold:
            return record["timestep"]
    return None


def plot_comparison(
    scenarios: List[Dict[str, Any]],
    histories: List[List[Dict[str, Any]]],
) -> None:
    """Figure: 1×3 subplots — disrupted count (left y) + network health (right y) per scenario."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Cascade Propagation Under Different Disruption Scenarios",
        fontsize=14, y=1.02,
    )

    for ax, scenario, history in zip(axes, scenarios, histories):
        timesteps = [r["timestep"] for r in history]
        disrupted = [len(r["total_disrupted"]) for r in history]
        health = [r["network_health"] for r in history]

        ax.plot(timesteps, disrupted, color=C_DISRUPTED, linewidth=2, label="Disrupted nodes")
        ax.set_xlabel("Timestep", fontsize=10)
        ax.set_ylabel("Cumulative disrupted nodes", fontsize=10, color=C_DISRUPTED)
        ax.tick_params(axis="y", labelcolor=C_DISRUPTED)
        ax.set_title(scenario["name"].replace("_", " ").title(), fontsize=11, pad=8)
        ax.yaxis.grid(True, linestyle="--")
        ax.set_axisbelow(True)

        ax2 = ax.twinx()
        ax2.plot(timesteps, health, color=C_HEALTH, linewidth=2,
                 linestyle="--", label="Network health")
        ax2.set_ylabel("Network health", fontsize=10, color=C_HEALTH)
        ax2.tick_params(axis="y", labelcolor=C_HEALTH, colors=C_HEALTH)
        ax2.set_ylim(0.0, 1.0)
        ax2.spines["right"].set_edgecolor(C_HEALTH)

        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper right")

    fig.tight_layout()
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[scenario_viz] Saved -> {SAVE_PATH}")


def print_summary_table(
    scenarios: List[Dict[str, Any]],
    histories: List[List[Dict[str, Any]]],
    n_nodes: int,
) -> None:
    """Print a pipe-formatted summary table of scenario outcomes to stdout."""
    cols = ("Scenario", "Initial disrupted", "Final disrupted", "Time to 50% cascade", "Final health")
    rows = []
    for sc, hist in zip(scenarios, histories):
        t50 = _time_to_50pct(hist, n_nodes)
        rows.append((
            sc["name"],
            str(len(sc["disrupted_nodes"])),
            str(len(hist[-1]["total_disrupted"])),
            str(t50) if t50 is not None else "N/A",
            f"{hist[-1]['network_health']:.4f}",
        ))
    widths = [max(len(c), max(len(r[i]) for r in rows)) for i, c in enumerate(cols)]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in widths) + " |"
    print(f"\n[scenario_viz] {sep}")
    print(f"[scenario_viz] {fmt.format(*cols)}")
    print(f"[scenario_viz] {sep}")
    for row in rows:
        print(f"[scenario_viz] {fmt.format(*row)}")
    print(f"[scenario_viz] {sep}\n")


def main() -> None:
    """Build graph, run 3 new scenarios, produce figure and print summary."""
    print("[scenario_viz] Building graph...")
    G = _build_graph()
    n_nodes: int = G.number_of_nodes()

    scenarios = [
        scenario_critical_hub_failure(G),
        scenario_supplier_cascade(G),
        scenario_bottleneck_plant(G),
    ]

    histories: List[List[Dict[str, Any]]] = []
    for sc in scenarios:
        print(f"[scenario_viz] Running '{sc['name']}' "
              f"(nodes={len(sc['disrupted_nodes'])}, severity={sc['severity']}) ...")
        histories.append(_run_scenario(G, sc))

    plot_comparison(scenarios, histories)
    print_summary_table(scenarios, histories, n_nodes)
    print("[scenario_viz] Done.")


if __name__ == "__main__":
    main()
