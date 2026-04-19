"""builder.py — Assemble the DTNet nx.DiGraph from topology specs and dataset.

Responsibility: given the node/edge specs from ``topology.infer_topology``
and a preprocessed DataFrame, instantiate the correct agent subclass for
every node (using real per-machine sensor averages where available, calibrated
synthetic values for all other node types) and wire up all edges with their
attributes.

Note on sensor values
---------------------
``preprocess.py`` min-max normalises sensor columns to [0, 1].  MachineAgent
stores these normalised values directly (fields ``temp_bearing``, etc.).
The health-score thresholds in MachineAgent are defined in physical units (°C,
bar), so health scores will be near-perfect for normalised inputs; this is a
known trade-off when scaler parameters are not passed.  If raw-unit accuracy
is required, inverse-transform the sensor columns using the scalers dict
returned by ``preprocess.normalise_sensors`` before calling ``build_graph``.

Public API
----------
- ``build_graph(nodes, edges, df)``  → ``nx.DiGraph``
- ``print_graph_summary(G)``         → None
"""

from __future__ import annotations

import numpy as np
import torch
import networkx as nx
import pandas as pd
from typing import Dict, List, Optional

from src.graph.topology_specs import EdgeSpec, NodeSpec
from src.graph.builder_agents import make_agent
from src.agents.base_agent import DigitalTwinAgent

np.random.seed(42)
torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Sensor column name mapping: processed-df column → MachineAgent field name
# ---------------------------------------------------------------------------

SENSOR_COL_MAP: Dict[str, str] = {
    "temp_bearing_degC":    "temp_bearing",
    "temp_motor_degC":      "temp_motor",
    "vibration_h_mms":      "vibration_h",
    "vibration_v_mms":      "vibration_v",
    "oil_pressure_bar":     "oil_pressure",
    "load_pct":             "load_pct",
    "power_consumption_kw": "power_kw",
    "shaft_rpm":            "rpm",
}

# Column used to group sensor rows by machine identity
COL_ASSET: str = "asset_tag"


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _extract_machine_sensor_means(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute per-machine mean sensor values from the processed DataFrame.

    Groups rows by ``asset_tag`` and takes the column mean for every sensor
    column that is present in both ``SENSOR_COL_MAP`` and ``df``.  Columns
    absent from ``df`` are silently skipped; callers fall back to NodeSpec
    defaults for those fields.

    Args:
        df: Preprocessed DataFrame as returned by ``preprocess.preprocess``.
            Sensor columns are normalised to [0, 1].

    Returns:
        Mapping ``{asset_tag: {agent_field: mean_value}}``.  Only assets that
        appear in ``df`` are included.
    """
    if COL_ASSET not in df.columns:
        return {}

    present_cols: List[str] = [c for c in SENSOR_COL_MAP if c in df.columns]
    if not present_cols:
        return {}

    grouped: pd.DataFrame = (
        df[[COL_ASSET] + present_cols]
        .groupby(COL_ASSET, sort=False)[present_cols]
        .mean()
    )

    return {
        str(asset_tag): {
            SENSOR_COL_MAP[col]: float(row[col])
            for col in present_cols
        }
        for asset_tag, row in grouped.iterrows()
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_graph(
    nodes: List[NodeSpec],
    edges: List[EdgeSpec],
    df: pd.DataFrame,
) -> nx.DiGraph:
    """Build the DTNet supply-chain graph from topology specs and dataset.

    For each NodeSpec a DigitalTwinAgent subclass is instantiated and stored
    under ``G.nodes[node_id]["twin"]``.  Machine nodes are initialised with
    real per-asset_tag sensor means from ``df``; all other node types use the
    calibrated synthetic values carried in the NodeSpec attributes.

    For each EdgeSpec a directed edge is added with ``edge_type``,
    ``flow_capacity``, ``criticality_weight``, and ``latency_days`` as edge
    attributes, matching the pattern from CODING_PATTERNS.md.

    Args:
        nodes: Node specs as returned by ``topology.infer_topology``.
        edges: Edge specs as returned by ``topology.infer_topology``.
        df: Preprocessed DataFrame from ``preprocess.preprocess``.  Used
            exclusively to derive per-machine sensor averages.

    Returns:
        A ``nx.DiGraph`` where every node carries ``{"twin": agent, "layer":
        str}`` and every edge carries ``{"edge_type": str, "flow_capacity":
        float, "criticality_weight": float, "latency_days": int}``.
    """
    np.random.seed(42)

    G: nx.DiGraph = nx.DiGraph()

    # Pre-compute sensor means once — O(N) scan of the dataframe
    sensor_lookup: Dict[str, Dict[str, float]] = _extract_machine_sensor_means(df)

    # --- Add nodes -----------------------------------------------------------
    for spec in nodes:
        agent: DigitalTwinAgent = make_agent(spec, sensor_lookup)
        G.add_node(spec.node_id, twin=agent, layer=spec.layer)

    # --- Add edges -----------------------------------------------------------
    for spec in edges:
        G.add_edge(
            spec.source,
            spec.target,
            edge_type=spec.edge_type,
            flow_capacity=spec.flow_capacity,
            criticality_weight=spec.criticality_weight,
            latency_days=spec.latency_days,
            shared_parts_count=spec.shared_parts_count,
        )

    print(
        f"[builder] Graph built — "
        f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges."
    )
    return G


def print_graph_summary(G: nx.DiGraph) -> None:
    """Print a structured summary of the built DTNet graph.

    Reports node counts and mean health scores by layer, edge counts by
    type, and overall graph connectivity statistics.

    Args:
        G: Completed DTNet DiGraph as returned by ``build_graph``.
    """
    layer_counts: Dict[str, int] = {}
    layer_health: Dict[str, List[float]] = {}

    for _, data in G.nodes(data=True):
        layer: str = data.get("layer", "unknown")
        twin: Optional[DigitalTwinAgent] = data.get("twin")
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
        if twin is not None:
            layer_health.setdefault(layer, []).append(twin.compute_health_score())

    edge_type_counts: Dict[str, int] = {}
    for _, _, data in G.edges(data=True):
        etype: str = data.get("edge_type", "unknown")
        edge_type_counts[etype] = edge_type_counts.get(etype, 0) + 1

    sep: str = "=" * 60
    print(f"\n{sep}")
    print("  DTNET GRAPH SUMMARY")
    print(sep)
    print(f"  Total nodes : {G.number_of_nodes():,}")
    for layer in ["supplier", "logistics", "plant", "machine", "distribution"]:
        count: int = layer_counts.get(layer, 0)
        scores: List[float] = layer_health.get(layer, [])
        avg_h: str = f"{sum(scores) / len(scores):.3f}" if scores else "n/a"
        print(f"    {layer:<20}: {count:>5,}  avg health = {avg_h}")

    print(f"\n  Total edges : {G.number_of_edges():,}")
    for etype in [
        "material_flow",
        "operational",
        "process_chain",
        "shared_part_dependency",
    ]:
        print(f"    {etype:<30}: {edge_type_counts.get(etype, 0):>6,}")

    print(f"\n  Connectivity")
    print(f"    Is weakly connected : {nx.is_weakly_connected(G)}")
    print(f"    Is DAG              : {nx.is_directed_acyclic_graph(G)}")
    n: int = max(G.number_of_nodes(), 1)
    avg_out: float = sum(d for _, d in G.out_degree()) / n
    print(f"    Avg out-degree      : {avg_out:.2f}")
    print(sep + "\n")
