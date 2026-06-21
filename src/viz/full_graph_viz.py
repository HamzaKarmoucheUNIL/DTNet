"""full_graph_viz.py — DTNet supply-chain graph topology figure (thesis).

Usage: python -m src.viz.full_graph_viz
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.viz.colors import LAYER_COLORS, LAYER_ORDER

np.random.seed(42)

BG: str = "white"
SAVE_PATH: Path = Path("results/fig_full_graph.png")
C: Dict[str, str] = LAYER_COLORS  # single source of truth (src/viz/colors.py)
EDGE_STYLES: Dict[str, dict] = {
    "material_flow":          {"color": "#5599EE", "ls": "solid",   "lw": 1.4, "alpha": 0.70},
    "operational":            {"color": "#EE9955", "ls": "dashed",  "lw": 0.9, "alpha": 0.55},
    "process_chain":          {"color": "#55DD99", "ls": "dotted",  "lw": 0.9, "alpha": 0.55},
    "shared_part_dependency": {"color": "#CC77CC", "ls": "dashdot", "lw": 0.8, "alpha": 0.50},
}

matplotlib.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "text.color": "#333333",
})


def _build_graph() -> nx.DiGraph:
    """Load raw data and build the DTNet DiGraph; suppresses print output."""
    from src.data.loader import load_csv
    from src.data.preprocess import preprocess
    from src.data.entity_mapping import build_entity_mappings
    from src.graph.topology import infer_topology
    from src.graph.builder import build_graph
    with contextlib.redirect_stdout(io.StringIO()):
        df_raw = load_csv("updated_data.csv")
        df_clean, _ = preprocess(df_raw)
        em = build_entity_mappings(df_raw)
        nodes, edges = infer_topology(em)
        G = build_graph(nodes, edges, df_clean)
    return G


def _hierarchical_pos(G: nx.DiGraph) -> Dict[str, tuple]:
    """Left-to-right layout; machine layer uses 3 sub-columns to avoid crowding."""
    LAYER_X: Dict[str, float] = {
        "supplier": 0.05, "logistics": 0.22, "plant": 0.40,
        "machine": 0.59, "distribution": 0.96,
    }
    NCOLS: int = 3
    buckets: Dict[str, List[str]] = {l: [] for l in LAYER_ORDER}
    for nid, data in G.nodes(data=True):
        buckets.setdefault(data.get("layer", "machine"), []).append(nid)
    pos: Dict[str, tuple] = {}
    for layer, nids in buckets.items():
        nids_s = sorted(nids)
        n = len(nids_s)
        x0 = LAYER_X.get(layer, 0.5)
        if layer == "machine":
            per_col = (n + NCOLS - 1) // NCOLS
            for i, nid in enumerate(nids_s):
                pos[nid] = (x0 + (i // per_col) * 0.13, ((i % per_col) + 0.5) / per_col)
        else:
            for i, nid in enumerate(nids_s):
                pos[nid] = (x0, (i + 0.5) / n)
    return pos


def main() -> None:
    """Build and save the DTNet full graph topology figure at 300 DPI."""
    print("[full_graph_viz] Building graph...")
    G = _build_graph()
    pos = _hierarchical_pos(G)

    deg = nx.degree_centrality(G)
    sizes = [70 + deg[n] * 2200 for n in G.nodes()]
    colors = [C.get(G.nodes[n].get("layer", "machine"), "#888888") for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(22, 13))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis("off")

    # Draw each edge type with its own colour + line style
    for etype, sty in EDGE_STYLES.items():
        edgelist = [(u, v) for u, v, d in G.edges(data=True)
                    if d.get("edge_type") == etype]
        if not edgelist:
            continue
        nx.draw_networkx_edges(
            G, pos, edgelist=edgelist, ax=ax,
            edge_color=sty["color"], style=sty["ls"],
            alpha=sty["alpha"], width=sty["lw"],
            arrows=True, arrowsize=7,
            min_source_margin=5, min_target_margin=5,
        )

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes,
                           alpha=0.93, edgecolors="#555555", linewidths=0.5)

    # Layer column labels at top
    col_label_x = {"supplier": 0.05, "logistics": 0.22, "plant": 0.40,
                   "machine": 0.72, "distribution": 0.96}
    for layer, x in col_label_x.items():
        count = sum(1 for _, d in G.nodes(data=True) if d.get("layer") == layer)
        ax.text(x, 1.012, f"{layer.capitalize()}\n(n={count})",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color=C[layer], transform=ax.transAxes)

    # Legend: node layer colours + edge type styles
    node_handles = [mpatches.Patch(color=C[l], label=l.capitalize()) for l in LAYER_ORDER]
    edge_handles = [
        mlines.Line2D([], [], color=s["color"], ls=s["ls"], lw=1.8,
                      label=et.replace("_", " "))
        for et, s in EDGE_STYLES.items()
    ]
    ax.legend(
        handles=node_handles + edge_handles,
        loc="lower center", ncol=9,
        facecolor="#F8F8F8", labelcolor="#333333", fontsize=9,
        edgecolor="#CCCCCC", bbox_to_anchor=(0.5, -0.04),
    )

    n_n, n_e = G.number_of_nodes(), G.number_of_edges()
    ax.set_title(
        f"DTNet Supply Chain Graph Topology\n"
        f"{n_n} nodes  ·  {n_e} edges  ·  5 node types  ·  4 edge types",
        color="#333333", fontsize=14, fontweight="bold", pad=16,
    )

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_PATH, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[full_graph_viz] Saved -> {SAVE_PATH}")


if __name__ == "__main__":
    main()
