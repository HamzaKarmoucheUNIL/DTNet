"""metrics.py — Graph-level metrics and vulnerability analysis for DTNet.

All functions are pure: they take a ``nx.DiGraph`` and return data structures.
No graph mutation, no side effects, no printing.

Public API
----------
- ``degree_centrality(G)``        → ``DegreeCentrality``
- ``betweenness_centrality(G)``   → ``Dict[str, float]``
- ``critical_nodes(G, top_n)``    → ``List[str]``
- ``vulnerable_edges(G, top_n)``  → ``List[Tuple[str, str, float]]``
- ``network_stats(G)``            → ``NetworkStats``
- ``vulnerability_analysis(G)``   → ``List[NodeVulnerability]``

``NodeVulnerability`` is re-exported from ``metrics_vulnerability`` so callers
only need to import from this module.
"""

from __future__ import annotations

import numpy as np
import torch
import networkx as nx
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.graph.metrics_vulnerability import (
    NodeVulnerability,
    count_dependents,
    compose_reason,
    composite_vulnerability_score,
)

np.random.seed(42)
torch.manual_seed(42)

# Default number of top results returned by ranking functions
DEFAULT_TOP_N: int = 5

# Edge attribute used as a weight proxy for betweenness computation.
# Inverted so high-criticality edges attract shortest paths.
_WEIGHT_ATTR: str = "criticality_weight"

# ---------------------------------------------------------------------------
# Return-type dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DegreeCentrality:
    """Normalised in- and out-degree centrality for every node.

    Attributes:
        in_degree: ``{node_id: normalised_in_degree}`` in [0, 1].
        out_degree: ``{node_id: normalised_out_degree}`` in [0, 1].
    """

    in_degree: Dict[str, float] = field(default_factory=dict)
    out_degree: Dict[str, float] = field(default_factory=dict)


@dataclass
class NetworkStats:
    """Scalar graph-level statistics.

    Attributes:
        density: Edge density (actual edges / possible edges).
        avg_path_length: Mean shortest-path length over the largest weakly
            connected component (undirected); ``None`` if the component has
            fewer than 2 nodes.
        avg_clustering: Mean clustering coefficient (undirected projection).
        num_weakly_connected_components: Count of weakly connected components.
        is_dag: True if the graph is acyclic.
    """

    density: float = 0.0
    avg_path_length: Optional[float] = None
    avg_clustering: float = 0.0
    num_weakly_connected_components: int = 0
    is_dag: bool = False


# ---------------------------------------------------------------------------
# Centrality
# ---------------------------------------------------------------------------

def degree_centrality(G: nx.DiGraph) -> DegreeCentrality:
    """Compute normalised in- and out-degree centrality for every node.

    Normalisation divides each raw degree by ``N - 1`` (NetworkX default).

    Args:
        G: DTNet DiGraph as built by ``builder.build_graph``.

    Returns:
        ``DegreeCentrality`` with both in- and out-degree mappings.
    """
    return DegreeCentrality(
        in_degree=nx.in_degree_centrality(G),
        out_degree=nx.out_degree_centrality(G),
    )


def betweenness_centrality(G: nx.DiGraph) -> Dict[str, float]:
    """Compute normalised betweenness centrality for every node.

    High-criticality edges are treated as "shorter" paths (via inverse weight)
    so nodes on many high-criticality shortest paths score higher.  For large
    graphs (> 500 nodes) an approximate k-sample variant is used.

    Args:
        G: DTNet DiGraph.

    Returns:
        ``{node_id: score}`` normalised to [0, 1].
    """
    n: int = G.number_of_nodes()
    if n == 0:
        return {}

    # Build weight-transformed copy: weight = 1 / criticality_weight
    H: nx.DiGraph = nx.DiGraph()
    H.add_nodes_from(G.nodes())
    for u, v, data in G.edges(data=True):
        cw: float = float(data.get(_WEIGHT_ATTR, 0.5))
        H.add_edge(u, v, weight=1.0 / cw if cw > 0.0 else 1e6)

    if n > 500:
        return nx.betweenness_centrality(
            H, normalized=True, weight="weight", seed=42, k=min(500, n)
        )
    return nx.betweenness_centrality(H, normalized=True, weight="weight")


# ---------------------------------------------------------------------------
# Critical nodes and vulnerable edges
# ---------------------------------------------------------------------------

def critical_nodes(G: nx.DiGraph, top_n: int = DEFAULT_TOP_N) -> List[str]:
    """Return the ``top_n`` nodes with the highest betweenness centrality.

    These are bottleneck nodes whose removal would most disrupt shortest paths
    — prime candidates for disruption scenarios.

    Args:
        G: DTNet DiGraph.
        top_n: Number of node IDs to return.

    Returns:
        List of node IDs sorted by descending betweenness centrality.
    """
    bc: Dict[str, float] = betweenness_centrality(G)
    return sorted(bc, key=lambda node: bc[node], reverse=True)[:top_n]


def vulnerable_edges(
    G: nx.DiGraph,
    top_n: int = DEFAULT_TOP_N,
) -> List[Tuple[str, str, float]]:
    """Return the ``top_n`` edges with the highest ``criticality_weight``.

    These are supply-chain connections whose failure propagates most
    aggressively to downstream nodes.

    Args:
        G: DTNet DiGraph.
        top_n: Number of edges to return.

    Returns:
        ``[(source, target, criticality_weight), …]`` sorted by descending
        criticality weight.
    """
    scored: List[Tuple[str, str, float]] = [
        (u, v, float(data.get(_WEIGHT_ATTR, 0.0)))
        for u, v, data in G.edges(data=True)
    ]
    scored.sort(key=lambda t: t[2], reverse=True)
    return scored[:top_n]


# ---------------------------------------------------------------------------
# Network-level statistics
# ---------------------------------------------------------------------------

def network_stats(G: nx.DiGraph) -> NetworkStats:
    """Compute scalar graph-level statistics.

    Average path length is measured on the largest weakly connected component
    treated as undirected (avoids ``∞`` for disconnected graphs).  Clustering
    uses the undirected projection of G.

    Args:
        G: DTNet DiGraph.

    Returns:
        Populated ``NetworkStats`` dataclass.
    """
    if G.number_of_nodes() == 0:
        return NetworkStats()

    density: float = nx.density(G)
    is_dag: bool = nx.is_directed_acyclic_graph(G)
    wcc: List[Any] = list(nx.weakly_connected_components(G))
    num_wcc: int = len(wcc)

    avg_path: Optional[float] = None
    if num_wcc > 0:
        largest: Any = max(wcc, key=len)
        sub: nx.Graph = G.subgraph(largest).to_undirected()
        if sub.number_of_nodes() > 1:
            avg_path = nx.average_shortest_path_length(sub)

    avg_clustering: float = nx.average_clustering(G.to_undirected())

    return NetworkStats(
        density=density,
        avg_path_length=avg_path,
        avg_clustering=avg_clustering,
        num_weakly_connected_components=num_wcc,
        is_dag=is_dag,
    )


# ---------------------------------------------------------------------------
# Composite vulnerability analysis
# ---------------------------------------------------------------------------

def vulnerability_analysis(
    G: nx.DiGraph,
    top_n: int = DEFAULT_TOP_N,
) -> List[NodeVulnerability]:
    """Identify the most critical nodes and explain their vulnerability.

    Composite score weights (defined in ``metrics_vulnerability.py``):
    - Betweenness centrality  40 % — bottleneck on critical paths
    - Out-degree centrality   25 % — upstream fan-out
    - In-degree centrality    15 % — input concentration
    - Normalised dependents   15 % — total downstream reachability
    - Inverse health score     5 % — already-degraded nodes are higher risk

    Args:
        G: DTNet DiGraph as built by ``builder.build_graph``.
        top_n: Number of top critical nodes to return (default 5).

    Returns:
        List of ``NodeVulnerability`` sorted by descending
        ``vulnerability_score``, length ``min(top_n, |V|)``.
    """
    n: int = G.number_of_nodes()
    if n == 0:
        return []

    bc: Dict[str, float] = betweenness_centrality(G)
    dc: DegreeCentrality = degree_centrality(G)

    results: List[NodeVulnerability] = []

    for node_id in G.nodes():
        data: Dict = G.nodes[node_id]
        layer: str = data.get("layer", "unknown")
        twin: Any = data.get("twin")
        health: Optional[float] = (
            twin.compute_health_score() if twin is not None else None
        )

        bw: float = bc.get(node_id, 0.0)
        out_dc: float = dc.out_degree.get(node_id, 0.0)
        in_dc: float = dc.in_degree.get(node_id, 0.0)
        num_dep: int = count_dependents(G, node_id)

        vscore: float = composite_vulnerability_score(
            bw, out_dc, in_dc, num_dep, health, n
        )
        reason: str = compose_reason(node_id, layer, bw, out_dc, num_dep, health)

        results.append(NodeVulnerability(
            node_id=node_id,
            layer=layer,
            betweenness=bw,
            in_degree_centrality=in_dc,
            out_degree_centrality=out_dc,
            num_dependents=num_dep,
            health_score=health,
            vulnerability_score=vscore,
            reason=reason,
        ))

    results.sort(key=lambda r: r.vulnerability_score, reverse=True)
    return results[:top_n]
