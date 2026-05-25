# pip install streamlit
"""app.py — DTNet Streamlit dashboard for thesis defense demo.

Single-page interactive dashboard: build graph, pick a disruption scenario,
run the cascading-failure simulation, visualise before/after network state,
and explore the cascade step-by-step on a timeline slider.

Usage: streamlit run dashboard/app.py
"""

from __future__ import annotations

import contextlib
import io
import random
from typing import Any, Dict, List, Set

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import streamlit as st
import torch

from src.simulation.model import DTNetModel
from src.simulation.scenarios import (
    scenario_bottleneck_plant,
    scenario_critical_hub_failure,
    scenario_supplier_cascade,
)

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BG: str = "#0a0e17"
LAYER_COLORS: Dict[str, str] = {
    "supplier": "#4c7aff", "logistics": "#9b59b6", "plant": "#2ecc71",
    "machine": "#f39c12", "distribution": "#e74c3c",
}
DISRUPTED_COLOR: str = "#ff3333"
N_STEPS: int = 15
SCENARIOS: List[str] = [
    "Random single node",
    "Critical hub (highest betweenness)",
    "All suppliers",
    "Bottleneck plant",
]


# ---------------------------------------------------------------------------
# Cached graph loader
# ---------------------------------------------------------------------------

@st.cache_resource
def load_graph():
    """Build the DTNet graph and spring layout once; share across reruns."""
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
    pos = nx.spring_layout(G, seed=42, k=0.6)
    return G, pos


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _node_colors(G: nx.DiGraph, disrupted: Set[str]) -> List[str]:
    """Return per-node color list; disrupted nodes override their layer color."""
    return [
        DISRUPTED_COLOR if nid in disrupted
        else LAYER_COLORS.get(G.nodes[nid].get("layer", "machine"), "#888888")
        for nid in G.nodes
    ]


def _draw_graph(G: nx.DiGraph, pos: Dict, disrupted: Set[str], title: str, ax) -> None:
    """Draw the supply-chain graph on ax with layer-coloured nodes."""
    nx.draw_networkx(
        G, pos=pos, ax=ax, node_color=_node_colors(G, disrupted),
        node_size=25, with_labels=False, arrows=True,
        arrowsize=5, edge_color="#2a2f3e", width=0.5,
    )
    ax.set_facecolor(BG)
    ax.set_title(title, color="white", fontsize=10, pad=6)
    ax.axis("off")


def _legend_patches() -> List[mpatches.Patch]:
    """Build matplotlib legend patches for all layers + disrupted marker."""
    patches = [mpatches.Patch(color=c, label=l.capitalize()) for l, c in LAYER_COLORS.items()]
    patches.append(mpatches.Patch(color=DISRUPTED_COLOR, label="Disrupted"))
    return patches


# ---------------------------------------------------------------------------
# Section 1 — Page config & header
# ---------------------------------------------------------------------------

st.set_page_config(page_title="DTNet Dashboard", layout="wide")
st.title("DTNet — Supply Chain Digital Twin Network Simulator")
st.subheader("Interactive Disruption Cascade Simulation")
st.caption(
    "Select a disruption scenario and parameters, then click **Run Simulation** "
    "to observe how failures cascade through the interconnected supply chain."
)

# ---------------------------------------------------------------------------
# Section 2 — Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Simulation Controls")
    propagation_decay = st.slider("Propagation decay", 0.1, 1.0, 0.6, 0.05)
    threshold        = st.slider("Propagation threshold", 0.05, 0.5, 0.15, 0.05)
    severity         = st.slider("Disruption severity", 0.3, 1.0, 0.7, 0.1)
    scenario_choice  = st.selectbox("Scenario", SCENARIOS)
    run_clicked      = st.button("▶  Run Simulation", type="primary", use_container_width=True)
    st.divider()
    st.caption("**Layer colours**")
    for layer, color in LAYER_COLORS.items():
        st.markdown(f'<span style="color:{color}">■</span> {layer.capitalize()}',
                    unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load graph (cached)
# ---------------------------------------------------------------------------

with st.spinner("Loading DTNet graph…"):
    G, pos = load_graph()

# ---------------------------------------------------------------------------
# Run simulation when button is clicked
# ---------------------------------------------------------------------------

if run_clicked:
    np.random.seed(42); random.seed(42)

    if scenario_choice == "Random single node":
        seed_nodes: List[str] = [random.choice(list(G.nodes()))]
    elif scenario_choice == "Critical hub (highest betweenness)":
        seed_nodes = scenario_critical_hub_failure(G)["disrupted_nodes"]
    elif scenario_choice == "All suppliers":
        seed_nodes = scenario_supplier_cascade(G)["disrupted_nodes"]
    else:
        seed_nodes = scenario_bottleneck_plant(G)["disrupted_nodes"]

    for _, d in G.nodes(data=True):
        d["twin"].reset()

    baseline_health    = float(np.mean([d["twin"].compute_health_score() for _, d in G.nodes(data=True)]))
    baseline_capacity  = float(np.mean([d["twin"].capacity              for _, d in G.nodes(data=True)]))

    model = DTNetModel(G, propagation_decay=propagation_decay, threshold=threshold)
    for nid in seed_nodes:
        model.inject_disruption(nid, severity)
    for _ in range(N_STEPS):
        model.step()

    st.session_state.update({
        "history":           model.get_history(),
        "seed_nodes":        seed_nodes,
        "baseline_health":   baseline_health,
        "baseline_capacity": baseline_capacity,
        "scenario_label":    scenario_choice,
    })

# ---------------------------------------------------------------------------
# Section 3 — Results (shown after simulation)
# ---------------------------------------------------------------------------

if "history" in st.session_state:
    history          = st.session_state["history"]
    seed_nodes       = st.session_state["seed_nodes"]
    baseline_health  = st.session_state["baseline_health"]
    baseline_cap     = st.session_state["baseline_capacity"]
    final            = history[-1]
    final_disrupted  = set(final["total_disrupted"])
    n_nodes          = G.number_of_nodes()

    st.divider()
    col_graph, col_metrics = st.columns([0.6, 0.4])
    with col_graph:
        st.subheader("Network State: Before vs After")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), facecolor=BG)
        _draw_graph(G, pos, set(seed_nodes),    f"Before  ({len(seed_nodes)} seed node(s))", ax1)
        _draw_graph(G, pos, final_disrupted,    f"After   ({len(final_disrupted)} disrupted)", ax2)
        fig.legend(handles=_legend_patches(), loc="lower center", ncol=6,
                   facecolor="#0d1117", labelcolor="white", fontsize=8,
                   bbox_to_anchor=(0.5, -0.06))
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_metrics:
        st.subheader("Summary Metrics")
        st.metric("Total nodes disrupted", f"{len(final_disrupted)} / {n_nodes}",
                  delta=f"+{len(final_disrupted)}")
        st.metric("Network health",
                  f"{final['network_health'] * 100:.1f}%",
                  delta=f"{(final['network_health'] - baseline_health) * 100:.1f}%",
                  delta_color="inverse")
        st.metric("Avg capacity",
                  f"{final['total_capacity'] * 100:.1f}%",
                  delta=f"{(final['total_capacity'] - baseline_cap) * 100:.1f}%",
                  delta_color="inverse")
        # Last step at which new nodes were disrupted
        cascade_end = max((r["timestep"] for r in history if r["newly_disrupted"]), default=0)
        st.metric("Time to full cascade", f"{cascade_end} steps")
        st.caption(f"**Scenario:** {st.session_state['scenario_label']}")
        seed_preview = ", ".join(seed_nodes[:4]) + (f" +{len(seed_nodes)-4} more" if len(seed_nodes) > 4 else "")
        st.caption(f"**Seeds:** {seed_preview}")

# ---------------------------------------------------------------------------
# Section 4 — Cascade timeline
# ---------------------------------------------------------------------------

if "history" in st.session_state:
    history = st.session_state["history"]
    st.divider()
    st.subheader("Cascade Timeline")
    t = st.slider("Timestep", 0, len(history) - 1, 0, key="timeline_t")
    record = history[t]

    c1, c2, c3 = st.columns(3)
    c1.metric("New disruptions at step", len(record["newly_disrupted"]))
    c2.metric("Total disrupted so far",  len(record["total_disrupted"]))
    c3.metric("Network health",          f"{record['network_health'] * 100:.1f}%")

    if record["newly_disrupted"]:
        st.info(f"**Newly disrupted at step {t}:** {', '.join(record['newly_disrupted'])}")
    else:
        st.success(f"No new disruptions at step {t}.")
    fig_t, ax_t = plt.subplots(figsize=(9, 4), facecolor=BG)
    _draw_graph(G, pos, set(record["total_disrupted"]), f"Network at timestep {t}", ax_t)
    st.pyplot(fig_t, use_container_width=True)
    plt.close(fig_t)
