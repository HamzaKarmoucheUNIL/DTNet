"""architecture_viz.py — DTNet 3-layer architecture diagram (thesis Figure 1).

Usage: python -m src.viz.architecture_viz
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

np.random.seed(42)

BG: str = "white"
SAVE_PATH: Path = Path("results/fig_architecture.png")
C: dict = {
    "supplier": "#4A9EFF", "logistics": "#9B59B6", "plant": "#2ECC71",
    "machine": "#F39C12", "distribution": "#E74C3C",
}


def _band(ax, y0: float, y1: float, color: str) -> None:
    """Translucent horizontal band spanning the full axes width for one layer."""
    ax.add_patch(mpatches.Rectangle(
        (0, y0), 1, y1 - y0, transform=ax.transAxes,
        facecolor=color, edgecolor="#BBBBCC", linewidth=1.0, alpha=0.11, clip_on=False,
    ))


def _box(ax, cx: float, cy: float, w: float, h: float,
         color: str, title: str, sub: str = "") -> None:
    """Rounded rectangle centered at (cx, cy) with bold title and optional subtitle."""
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.012",
        facecolor=color, edgecolor="white", linewidth=1.8, alpha=0.92,
    ))
    dy = 0.020 if sub else 0
    ax.text(cx, cy + dy, title, ha="center", va="center",
            fontsize=10, fontweight="bold", color="white")
    if sub:
        ax.text(cx, cy - dy, sub, ha="center", va="center",
                fontsize=7.5, color="white", alpha=0.85)


def main() -> None:
    """Build and save the DTNet 3-layer architecture figure at 300 DPI."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    _band(ax, 0.02, 0.30, "#4A9EFF")
    _band(ax, 0.33, 0.63, "#9B59B6")
    _band(ax, 0.66, 0.96, "#2ECC71")

    # RQ badges — right margin, connect layers to thesis research questions
    rq_kw = dict(ha="center", va="center", fontsize=7.5, fontweight="bold", color="#555577",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#EEEEF8", edgecolor="#555577", lw=1.2))
    ax.text(0.958, 0.160, "RQ1", **rq_kw)
    ax.text(0.958, 0.490, "RQ1\nRQ2", **rq_kw)
    ax.text(0.958, 0.810, "RQ2\nRQ3", **rq_kw)

    # ── NODE LAYER ───────────────────────────────────────────────────────────
    ax.text(0.46, 0.272, "LAYER 1  —  NODE LAYER  ·  Digital Twin Agents",
            ha="center", fontsize=12, fontweight="bold", color="#1A1A2E")
    ax.text(0.46, 0.243, "Each node models a real supply-chain entity with its own state, attributes, and health score",
            ha="center", fontsize=8.5, color="#444466", style="italic")
    nodes = [
        ("supplier",     "Supplier",      "delivery_reliability\nlead_time_days"),
        ("logistics",    "Logistics",     "transit_time\nroute_reliability"),
        ("plant",        "Plant",         "production_rate\nquality_rate"),
        ("machine",      "Machine",       "temp · vibration\nrpm · load_pct"),
        ("distribution", "Distribution",  "fulfillment_rate\nstock_level"),
    ]
    for i, (key, label, attrs) in enumerate(nodes):
        _box(ax, 0.10 + i * 0.185, 0.132, 0.158, 0.087, C[key], label, attrs)

    # Inter-layer arrow 1
    for fx in [0.28, 0.50, 0.70]:
        ax.annotate("", xy=(fx, 0.332), xytext=(fx, 0.232),
                    arrowprops=dict(arrowstyle="-|>", color="#AAAACC", lw=1.8, mutation_scale=14))
    ax.text(0.46, 0.303, "node attributes   →   graph nodes & edges",
            ha="center", fontsize=8.5, color="#666688")

    # ── GRAPH LAYER ──────────────────────────────────────────────────────────
    ax.text(0.46, 0.604, "LAYER 2  —  GRAPH LAYER  ·  Directed Graph  G = (V, E)  via NetworkX",
            ha="center", fontsize=12, fontweight="bold", color="#1A1A2E")
    ax.text(0.46, 0.574, "Nodes = digital twins  ·  Edges encode flow type & criticality weight  ·  4 edge types",
            ha="center", fontsize=8.5, color="#444466", style="italic")
    gpos = {"S": (0.12, 0.482), "L": (0.29, 0.492), "P": (0.46, 0.458),
            "M": (0.63, 0.492), "D": (0.80, 0.482)}
    gcolors = {k: C[v] for k, v in
               zip("SLPMD", ["supplier", "logistics", "plant", "machine", "distribution"])}
    for src, tgt in [("S", "L"), ("L", "P"), ("P", "M"), ("M", "D"), ("S", "P"), ("L", "M")]:
        x1, y1 = gpos[src]; x2, y2 = gpos[tgt]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#888899", lw=1.5, mutation_scale=13))
    for k, (gx, gy) in gpos.items():
        ax.add_patch(plt.Circle((gx, gy), 0.033, color=gcolors[k], zorder=5))
        ax.text(gx, gy, k, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white", zorder=6)
    etypes = [("material_flow", "#4A9EFF"), ("operational", "#9B59B6"),
              ("process_chain", "#F39C12"), ("shared_part_dep.", "#E74C3C")]
    for i, (et, ec) in enumerate(etypes):
        x = 0.07 + (i % 2) * 0.24; y = 0.372 - (i // 2) * 0.022
        ax.add_patch(plt.Circle((x, y), 0.007, color=ec, zorder=4))
        ax.text(x + 0.013, y, et, va="center", fontsize=7.8, color="#555566")

    # Inter-layer arrow 2
    for fx in [0.28, 0.50, 0.70]:
        ax.annotate("", xy=(fx, 0.658), xytext=(fx, 0.558),
                    arrowprops=dict(arrowstyle="-|>", color="#AAAACC", lw=1.8, mutation_scale=14))
    ax.text(0.46, 0.629, "graph topology  +  node features   →   GNN input tensors",
            ha="center", fontsize=8.5, color="#666688")

    # ── INTELLIGENCE LAYER ───────────────────────────────────────────────────
    ax.text(0.46, 0.932, "LAYER 3  —  INTELLIGENCE LAYER  ·  GNN + Agent-Based Simulation",
            ha="center", fontsize=12, fontweight="bold", color="#1A1A2E")
    ax.text(0.46, 0.903, "Learns propagation patterns from graph structure  ·  Predicts & simulates cascading failures",
            ha="center", fontsize=8.5, color="#444466", style="italic")
    ax.add_patch(FancyBboxPatch((0.055, 0.725), 0.340, 0.140, boxstyle="round,pad=0.015",
                                facecolor="#1B4F72", edgecolor="#4A9EFF", linewidth=2.0))
    ax.text(0.225, 0.818, "GNN", ha="center", fontsize=16, fontweight="bold", color="#4A9EFF")
    ax.text(0.225, 0.784, "PyTorch Geometric  ·  GATConv", ha="center", fontsize=8.5, color="#AACCFF")
    ax.text(0.225, 0.760, "Disruption severity prediction", ha="center", fontsize=8.0, color="#7799BB")
    ax.add_patch(FancyBboxPatch((0.550, 0.725), 0.340, 0.140, boxstyle="round,pad=0.015",
                                facecolor="#0D3B1E", edgecolor="#2ECC71", linewidth=2.0))
    ax.text(0.720, 0.818, "ABS", ha="center", fontsize=16, fontweight="bold", color="#2ECC71")
    ax.text(0.720, 0.784, "Mesa  ·  SimultaneousActivation", ha="center", fontsize=8.5, color="#AAFFCC")
    ax.text(0.720, 0.760, "Cascade propagation simulation", ha="center", fontsize=8.0, color="#66BB88")
    # Arrow above both boxes with label clearly in the gap
    ax.annotate("", xy=(0.550, 0.875), xytext=(0.395, 0.875),
                arrowprops=dict(arrowstyle="<->", color="#888899", lw=2.0, mutation_scale=16))
    ax.text(0.472, 0.888, "feeds predictions", ha="center", fontsize=8, color="#666677")
    ax.text(0.46, 0.706,
            "Output:  disrupted node set  ·  cascade severity scores  ·  network health trajectory",
            ha="center", fontsize=8.5, color="#333355", style="italic")

    fig.suptitle(
        "DTNet Architecture — Three-Layer Framework for Supply Chain Disruption Prediction",
        fontsize=13, fontweight="bold", color="#1A1A2E", y=0.995,
    )
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_PATH, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[architecture_viz] Saved -> {SAVE_PATH}")


if __name__ == "__main__":
    main()
