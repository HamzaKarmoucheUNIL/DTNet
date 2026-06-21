# pip install streamlit
"""app.py — DTNet Streamlit dashboard for thesis defense demo.

Single-page interactive dashboard: build graph, pick a disruption scenario,
run the cascading-failure simulation, visualise before/after network state,
and explore the cascade step-by-step on a timeline slider.

Usage: streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import contextlib
import io
import random
from typing import Any, Dict, List, Set

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from src.simulation.model import DTNetModel
from src.simulation.scenarios import (
    scenario_bottleneck_plant,
    scenario_critical_hub_failure,
    scenario_supplier_cascade,
)
from src.viz.colors import LAYER_COLORS, DISRUPTED_COLOR

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Force a white background on all dashboard figures (light theme).
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BG: str = "white"
LOGO: Path = Path(__file__).resolve().parent / "DTNET_logo.png"
# LAYER_COLORS and DISRUPTED_COLOR come from the single source of truth in
# src/viz/colors.py. Seed nodes use gold so they stay distinct from the
# logistics layer (orange) in the canonical palette.
SEVERITY_COLOR: str = "#FFD700"    # gold — seed nodes where disruption severity is injected
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

def _node_colors(G: nx.DiGraph, disrupted: Set[str], seeds: Set[str] = frozenset()) -> List[str]:
    """Return per-node color list.

    Seed nodes (where disruption severity is injected) take the severity colour
    (orange); other disrupted nodes take the disrupted colour (red); the rest
    keep their layer colour.
    """
    colors: List[str] = []
    for nid in G.nodes:
        if nid in seeds:
            colors.append(SEVERITY_COLOR)
        elif nid in disrupted:
            colors.append(DISRUPTED_COLOR)
        else:
            colors.append(LAYER_COLORS.get(G.nodes[nid].get("layer", "machine"), "#888888"))
    return colors


def _draw_graph(G: nx.DiGraph, pos: Dict, disrupted: Set[str], title: str, ax,
                seeds: Set[str] = frozenset()) -> None:
    """Draw the supply-chain graph on ax with layer-coloured nodes."""
    nx.draw_networkx(
        G, pos=pos, ax=ax, node_color=_node_colors(G, disrupted, seeds),
        node_size=25, with_labels=False, arrows=True,
        arrowsize=5, edge_color="#cccccc", width=0.5,
    )
    ax.set_facecolor(BG)
    ax.set_title(title, color="#1a1d24", fontsize=10, pad=6)
    ax.axis("off")


def _legend_patches() -> List[mpatches.Patch]:
    """Build matplotlib legend patches for all layers + severity/disrupted markers."""
    patches = [mpatches.Patch(color=c, label=l.capitalize()) for l, c in LAYER_COLORS.items()]
    patches.append(mpatches.Patch(color=SEVERITY_COLOR, label="Disruption severity (seed)"))
    patches.append(mpatches.Patch(color=DISRUPTED_COLOR, label="Disrupted"))
    return patches


def _build_step_figure(G: nx.DiGraph, pos: Dict, record: Dict[str, Any]) -> go.Figure:
    """Build a single-step Plotly figure of the cascade for one history record.

    Shows only the current step (no animation frames); the Streamlit slider
    redraws it on change. A node is red (#D62728) if disrupted at this step
    (member of ``record['total_disrupted']``) and green (#2CA02C) otherwise.
    Reuses the precomputed spring layout `pos`; reads existing run history only.

    Args:
        G: The DTNet supply-chain graph.
        pos: Precomputed node positions (computed once with seed=42).
        record: One per-step dict from ``DTNetModel.get_history()``.

    Returns:
        A static (frame-less) Plotly Figure for the given step.
    """
    disrupted: Set[str] = set(record["total_disrupted"])

    # Edges are static.
    edge_x: List[float] = []
    edge_y: List[float] = []
    for u, v in G.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.6, color="#CCCCCC"), hoverinfo="none",
    )

    colors = [DISRUPTED_COLOR if n in disrupted else "#2CA02C" for n in G.nodes()]
    node_trace = go.Scatter(
        x=[pos[n][0] for n in G.nodes()], y=[pos[n][1] for n in G.nodes()],
        mode="markers",
        marker=dict(size=14, color=colors, line=dict(width=0.5, color="#FFFFFF")),
        text=list(G.nodes()), hoverinfo="text",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white", showlegend=False,
        height=480, margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text=f"Step {record['timestep']}, {len(disrupted)} disrupted",
                   x=0.5, xanchor="center"),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


def _render_disrupted_badges(record: Dict[str, Any]) -> None:
    """Render the current step's disrupted nodes as red pill badges (or None)."""
    names: List[str] = record["total_disrupted"]
    st.markdown(f"**Disrupted nodes (step {record['timestep']})**")
    if names:
        pills = " ".join(
            "<span style='background:#FDECEC; color:#D62728; border-radius:12px; "
            "padding:3px 12px; margin:2px; display:inline-block; font-size:0.85rem;'>"
            f"{name}</span>"
            for name in names
        )
        st.markdown(pills, unsafe_allow_html=True)
    else:
        st.markdown("None")


# ---------------------------------------------------------------------------
# Section 1 — Page config & header
# ---------------------------------------------------------------------------

st.set_page_config(page_title="DTNet", layout="wide")

# Cosmetic-only: round metric cards and add breathing room around the layout.
st.markdown(
    """
    <style>
        .block-container { padding-top: 2.5rem; padding-bottom: 3rem; }
        div[data-testid="stMetric"] {
            background-color: #f7f9fc;
            border: 1px solid #e2e6ee;
            border-radius: 14px;
            padding: 16px 20px;
        }
        div[data-testid="stMetric"] label { opacity: 0.75; }
        section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
        hr { margin: 0.8rem 0; }

        /* Play / Pause timeline buttons (targeted via their st-key-* classes). */
        .st-key-play_btn button {
            background-color: #2CA02C !important;
            color: #ffffff !important;
            border: 0 !important;
            border-radius: 8px !important;
        }
        .st-key-pause_btn button {
            background-color: #D62728 !important;
            color: #ffffff !important;
            border: 0 !important;
            border-radius: 8px !important;
        }
        .st-key-play_btn button:disabled,
        .st-key-pause_btn button:disabled {
            background-color: #c7ccd4 !important;
            color: #ffffff !important;
            opacity: 0.7 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.image(LOGO, width=300)
st.markdown(
    "<p style='color:#9AA0A6; font-size:1.1rem; margin-top:-0.4rem;'>"
    "Supply Chain Digital Twin Network Simulator</p>",
    unsafe_allow_html=True,
)
st.subheader("Interactive Disruption Cascade Simulation")
st.caption(
    "Select a disruption scenario and parameters, then click **Run Simulation** "
    "to observe how failures cascade through the interconnected supply chain."
)

# ---------------------------------------------------------------------------
# Section 2 — Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Scenario")
    scenario_choice  = st.selectbox("Scenario", SCENARIOS)
    run_clicked      = st.button("▶  Run Simulation", type="primary", use_container_width=True)
    st.divider()

    st.header("Parameters")
    propagation_decay = st.slider("Propagation decay", 0.1, 1.0, 0.6, 0.05)
    threshold        = st.slider("Propagation threshold", 0.05, 0.5, 0.15, 0.05)
    severity         = st.slider("Disruption severity", 0.3, 1.0, 0.7, 0.1)
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
# Custom scenario controls (need the loaded graph for the node list)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.divider()
    st.header("Custom Scenario")
    custom_node = st.selectbox("Initial disrupted node", sorted(G.nodes()), key="custom_node")
    custom_run_clicked = st.button("Run custom scenario", use_container_width=True, key="custom_run")

# ---------------------------------------------------------------------------
# Run simulation when a Run button is clicked
# ---------------------------------------------------------------------------

if run_clicked or custom_run_clicked:
    np.random.seed(42); random.seed(42)

    if custom_run_clicked:
        # Custom scenario: disrupt the chosen node at t=0, then reuse the exact
        # same simulation code path below (no new simulation logic).
        seed_nodes: List[str] = [custom_node]
        scenario_label: str = f"Custom simulated cascade (seed: {custom_node})"
    elif scenario_choice == "Random single node":
        seed_nodes = [random.choice(list(G.nodes()))]
        scenario_label = scenario_choice
    elif scenario_choice == "Critical hub (highest betweenness)":
        seed_nodes = scenario_critical_hub_failure(G)["disrupted_nodes"]
        scenario_label = scenario_choice
    elif scenario_choice == "All suppliers":
        seed_nodes = scenario_supplier_cascade(G)["disrupted_nodes"]
        scenario_label = scenario_choice
    else:
        seed_nodes = scenario_bottleneck_plant(G)["disrupted_nodes"]
        scenario_label = scenario_choice

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
        "scenario_label":    scenario_label,
    })
    # Reset the timeline controls for the fresh run (set before the Step slider
    # widget is instantiated later in this run).
    st.session_state["step"] = 0
    st.session_state["is_playing"] = False

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
        _draw_graph(G, pos, set(seed_nodes),    f"Before  ({len(seed_nodes)} seed node(s))", ax1,
                    seeds=set(seed_nodes))
        _draw_graph(G, pos, final_disrupted,    f"After   ({len(final_disrupted)} disrupted)", ax2,
                    seeds=set(seed_nodes))
        fig.legend(handles=_legend_patches(), loc="lower center", ncol=7,
                   facecolor="white", labelcolor="#1a1d24", fontsize=8,
                   bbox_to_anchor=(0.5, -0.06))
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_metrics:
        st.subheader("Summary Metrics")
        # Last step at which new nodes were disrupted
        cascade_end = max((r["timestep"] for r in history if r["newly_disrupted"]), default=0)
        mcol1, mcol2 = st.columns(2)
        mcol1.metric("Total nodes disrupted", f"{len(final_disrupted)} / {n_nodes}",
                     delta=f"+{len(final_disrupted)}")
        mcol2.metric("Network health",
                     f"{final['network_health'] * 100:.1f}%",
                     delta=f"{(final['network_health'] - baseline_health) * 100:.1f}%",
                     delta_color="inverse")
        mcol3, mcol4 = st.columns(2)
        mcol3.metric("Avg capacity",
                     f"{final['total_capacity'] * 100:.1f}%",
                     delta=f"{(final['total_capacity'] - baseline_cap) * 100:.1f}%",
                     delta_color="inverse")
        mcol4.metric("Time to full cascade", f"{cascade_end} steps")
        st.caption(f"**Scenario:** {st.session_state['scenario_label']}")
        seed_preview = ", ".join(seed_nodes[:4]) + (f" +{len(seed_nodes)-4} more" if len(seed_nodes) > 4 else "")
        st.caption(f"**Seeds:** {seed_preview}")

# ---------------------------------------------------------------------------
# Section 4 — Cascade timeline (Streamlit-piloted)
# ---------------------------------------------------------------------------

if "history" in st.session_state:
    history = st.session_state["history"]
    st.divider()
    st.subheader("Cascade Timeline")
    st.caption("Use the slider, or press Play, to step through the simulated "
               "cascade (red = disrupted, green = operational).")

    st.session_state.setdefault("is_playing", False)
    st.session_state.setdefault("step", 0)
    # Keep the step within bounds (e.g. after a new, shorter run).
    st.session_state["step"] = min(st.session_state["step"], len(history) - 1)

    # Play / Pause: real Streamlit buttons in narrow left columns. They live
    # OUTSIDE the fragment so a click triggers a full rerun and re-evaluates the
    # fragment's run_every (start/stop the autoplay timer).
    play_col, pause_col, _spacer = st.columns([1, 1, 6])
    with play_col:
        if st.button("▶ Play", key="play_btn", use_container_width=True,
                     disabled=st.session_state["is_playing"]):
            if st.session_state["step"] >= len(history) - 1:
                st.session_state["step"] = 0  # restart if at the end
            st.session_state["is_playing"] = True
            st.rerun()
    with pause_col:
        if st.button("⏸ Pause", key="pause_btn", use_container_width=True,
                     disabled=not st.session_state["is_playing"]):
            st.session_state["is_playing"] = False
            st.rerun()

    # Autoplay via a fragment that reruns every ~0.6s while playing (no blocking
    # loop). run_every is evaluated at full-app-run time, which is why the
    # buttons above force a full rerun.
    _play_interval = 0.6 if st.session_state["is_playing"] else None

    @st.fragment(run_every=_play_interval)
    def _timeline_fragment() -> None:
        """Render the current-step graph, slider, badges and per-step metrics.

        Advances the step while playing and reruns the whole app to stop the
        timer at the end. Reads existing run history only — no simulation.
        """
        hist = st.session_state["history"]
        n_steps = len(hist)

        # Advance before the slider is instantiated this run (allowed pattern).
        if st.session_state["is_playing"]:
            if st.session_state["step"] < n_steps - 1:
                st.session_state["step"] += 1
            else:
                st.session_state["is_playing"] = False
                st.rerun()

        step = st.slider("Step", 0, n_steps - 1, key="step")
        record = hist[step]

        st.plotly_chart(_build_step_figure(G, pos, record), use_container_width=True)
        _render_disrupted_badges(record)

        c1, c2, c3 = st.columns(3)
        c1.metric("New disruptions at step", len(record["newly_disrupted"]))
        c2.metric("Total disrupted so far",  len(record["total_disrupted"]))
        c3.metric("Network health",          f"{record['network_health'] * 100:.1f}%")

    _timeline_fragment()
