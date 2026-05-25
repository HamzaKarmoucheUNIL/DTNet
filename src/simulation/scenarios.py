"""scenarios.py — Disruption scenario factories for DTNet simulation.

Each function returns an ``initial_disruption`` dict {node_id: severity} ready
to pass to ``DTNetModel.inject_disruption()``.  Scenarios cover a spectrum from
localised to network-wide shocks, providing diverse training data for the GNN
(COMMON_MISTAKES #8) and covering all three research questions of the thesis.

Scenario → thesis mapping
  single_supplier_failure  RQ1 — localised upstream shock propagation
  logistics_bottleneck     RQ1, RQ2 — mid-chain bottleneck amplification
  multi_supplier_cascade   RQ2 — correlated multi-source simultaneous failure
  targeted_attack          RQ3 — structural vulnerability via centrality
  random_disruption        RQ1–3 — diverse GNN training data generation
"""

from __future__ import annotations

import numpy as np
import torch
import networkx as nx
from typing import Any, Dict, List

np.random.seed(42)
torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LAYER_SUPPLIER: str = "supplier"
LAYER_LOGISTICS: str = "logistics"
LAYER_PLANT: str = "plant"
_BETWEENNESS_NORMALISED: bool = True
SEVERITY_CRITICAL_HUB: float = 0.9
SEVERITY_SUPPLIER_CASCADE: float = 0.7
SEVERITY_BOTTLENECK_PLANT: float = 0.85


# ---------------------------------------------------------------------------
# Scenario 1 — Single supplier failure
# ---------------------------------------------------------------------------


def single_supplier_failure(
    G: nx.DiGraph,
    supplier_id: str,
    severity: float = 0.9,
) -> Dict[str, float]:
    """Simulate sudden failure of one upstream supplier.

    Thesis relevance (RQ1)
    ----------------------
    Models a factory fire, strike, or trade restriction that takes a single
    supplier offline.  A high default severity (0.9) maximises observable
    cascading effects, letting the GNN learn how far a localised upstream shock
    travels before attenuating below the propagation threshold.

    Args:
        G: DTNet DiGraph with "layer" and "twin" attributes on every node.
        supplier_id: Node ID of the supplier to disrupt (must be layer='supplier').
        severity: Disruption magnitude in [0, 1]. Default 0.9.

    Returns:
        Dict mapping supplier_id → severity.

    Raises:
        KeyError: If supplier_id is not in G.
        ValueError: If the node is not a supplier-layer node.
    """
    if supplier_id not in G.nodes:
        raise KeyError(f"Node '{supplier_id}' not found in the graph.")
    node_layer: str = G.nodes[supplier_id].get("layer", "")
    if node_layer != LAYER_SUPPLIER:
        raise ValueError(
            f"Node '{supplier_id}' has layer '{node_layer}', expected 'supplier'."
        )
    return {supplier_id: float(np.clip(severity, 0.0, 1.0))}


# ---------------------------------------------------------------------------
# Scenario 2 — Logistics hub bottleneck
# ---------------------------------------------------------------------------


def logistics_bottleneck(
    G: nx.DiGraph,
    hub_id: str,
    severity: float = 0.7,
) -> Dict[str, float]:
    """Simulate congestion or partial failure of a logistics hub.

    Thesis relevance (RQ1, RQ2)
    ---------------------------
    Logistics hubs route flows from many suppliers to many plants, so their
    failure simultaneously starves multiple downstream nodes — a wider blast
    radius than a single-supplier shock of equal severity.  The lower default
    (0.7) represents partial congestion (port delays, warehouse overflow) rather
    than total shutdown, testing whether the GNN learns that mid-chain position
    amplifies impact beyond raw severity.

    Args:
        G: DTNet DiGraph with "layer" and "twin" attributes on every node.
        hub_id: Node ID of the logistics hub (must be layer='logistics').
        severity: Disruption magnitude in [0, 1]. Default 0.7.

    Returns:
        Dict mapping hub_id → severity.

    Raises:
        KeyError: If hub_id is not in G.
        ValueError: If the node is not a logistics-layer node.
    """
    if hub_id not in G.nodes:
        raise KeyError(f"Node '{hub_id}' not found in the graph.")
    node_layer: str = G.nodes[hub_id].get("layer", "")
    if node_layer != LAYER_LOGISTICS:
        raise ValueError(
            f"Node '{hub_id}' has layer '{node_layer}', expected 'logistics'."
        )
    return {hub_id: float(np.clip(severity, 0.0, 1.0))}


# ---------------------------------------------------------------------------
# Scenario 3 — Multi-supplier cascade
# ---------------------------------------------------------------------------


def multi_supplier_cascade(
    G: nx.DiGraph,
    n_suppliers: int = 2,
    severity: float = 0.8,
    seed: int = 42,
) -> Dict[str, float]:
    """Simulate simultaneous failure of multiple upstream suppliers.

    Thesis relevance (RQ2)
    ----------------------
    Natural disasters and industry-wide shortages (e.g. semiconductor crisis)
    knock out several suppliers concurrently.  All selected suppliers receive
    the same severity so structural position — not asymmetric severity —
    drives the difference in propagation pattern.  Tests whether the GNN
    captures non-linear interaction effects that exceed the sum of independent
    single-supplier shocks.

    Args:
        G: DTNet DiGraph with "layer" and "twin" attributes on every node.
        n_suppliers: Number of suppliers to disrupt (clamped to available count).
        severity: Disruption magnitude applied to all selected suppliers [0,1].
        seed: Random seed for reproducible supplier selection.

    Returns:
        Dict mapping each selected supplier node_id → severity.

    Raises:
        ValueError: If the graph has no supplier-layer nodes.
    """
    supplier_ids: List[str] = [
        nid for nid, d in G.nodes(data=True) if d.get("layer") == LAYER_SUPPLIER
    ]
    if not supplier_ids:
        raise ValueError("Graph contains no nodes with layer='supplier'.")

    rng: np.random.Generator = np.random.default_rng(seed)
    n: int = min(n_suppliers, len(supplier_ids))
    selected: List[str] = list(rng.choice(supplier_ids, size=n, replace=False))

    return {nid: float(np.clip(severity, 0.0, 1.0)) for nid in selected}


# ---------------------------------------------------------------------------
# Scenario 4 — Targeted attack on the most critical node
# ---------------------------------------------------------------------------


def targeted_attack(
    G: nx.DiGraph,
    strategy: str = "highest_betweenness",
    severity: float = 0.9,
) -> Dict[str, float]:
    """Attack the single most structurally critical node in the network.

    Thesis relevance (RQ3)
    ----------------------
    Centrality-based targeting identifies the structurally irreplaceable node
    whose removal maximally fragments flow paths.  This scenario underpins the
    worst-case baseline for RQ3: does the GNN learn to assign high pre-disruption
    vulnerability scores to high-centrality nodes, enabling proactive risk
    prioritisation?  It also stress-tests the isolated-vs-networked comparison
    because isolated twins cannot account for their own centrality.

    Supported strategies
    --------------------
    'highest_betweenness' — most shortest paths pass through this node (default).
    'highest_degree'      — node with the most direct connections.
    'lowest_health'       — currently weakest node (pre-existing degradation).

    Args:
        G: DTNet DiGraph with "layer" and "twin" attributes on every node.
        strategy: Node selection method. One of 'highest_betweenness',
            'highest_degree', 'lowest_health'.
        severity: Disruption magnitude in [0, 1]. Default 0.9.

    Returns:
        Dict mapping the selected node_id → severity.

    Raises:
        ValueError: If G is empty or strategy is not recognised.
    """
    if G.number_of_nodes() == 0:
        raise ValueError("Graph is empty — no nodes to attack.")

    valid: List[str] = ["highest_betweenness", "highest_degree", "lowest_health"]
    if strategy not in valid:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {valid}.")

    severity = float(np.clip(severity, 0.0, 1.0))

    if strategy == "highest_betweenness":
        centrality: Dict[str, float] = nx.betweenness_centrality(
            G, normalized=_BETWEENNESS_NORMALISED
        )
        target: str = max(centrality, key=centrality.__getitem__)
    elif strategy == "highest_degree":
        target = max(G.nodes, key=lambda n: G.degree(n))
    else:  # lowest_health
        target = min(G.nodes, key=lambda n: G.nodes[n]["twin"].compute_health_score())

    return {target: severity}


# ---------------------------------------------------------------------------
# Scenario 5 — Random disruption
# ---------------------------------------------------------------------------


def random_disruption(
    G: nx.DiGraph,
    n_nodes: int = 3,
    min_severity: float = 0.3,
    max_severity: float = 0.9,
    seed: int = 42,
) -> Dict[str, float]:
    """Disrupt a random selection of nodes at random severities.

    Thesis relevance (RQ1–3)
    ------------------------
    GNN training requires thousands of diverse snapshots with varied disruption
    locations and magnitudes (COMMON_MISTAKES #8).  Sweeping seed across many
    calls generates the distributional breadth needed to prevent overfitting to
    any single failure pattern.  It also mirrors real-world disruption dynamics:
    most failures are neither perfectly targeted nor fully random, making this
    scenario the most ecologically valid baseline for the thesis.

    Args:
        G: DTNet DiGraph with "layer" and "twin" attributes on every node.
        n_nodes: Number of nodes to disrupt (clamped to graph size).
        min_severity: Lower bound of the uniform severity distribution [0, 1].
        max_severity: Upper bound; must be >= min_severity.
        seed: Random seed for reproducible node selection and severity sampling.

    Returns:
        Dict mapping each selected node_id → sampled severity.

    Raises:
        ValueError: If G is empty or min_severity > max_severity.
    """
    if G.number_of_nodes() == 0:
        raise ValueError("Graph is empty — no nodes to disrupt.")
    if min_severity > max_severity:
        raise ValueError(
            f"min_severity ({min_severity}) must be <= max_severity ({max_severity})."
        )

    rng: np.random.Generator = np.random.default_rng(seed)
    all_nodes: List[str] = list(G.nodes)
    n: int = min(n_nodes, len(all_nodes))
    selected: List[str] = list(rng.choice(all_nodes, size=n, replace=False))
    severities: np.ndarray = rng.uniform(min_severity, max_severity, size=n)

    return {
        nid: float(np.clip(sev, 0.0, 1.0))
        for nid, sev in zip(selected, severities)
    }


# ---------------------------------------------------------------------------
# Scenario 6 — Critical hub failure (highest betweenness centrality)
# ---------------------------------------------------------------------------


def scenario_critical_hub_failure(G: nx.DiGraph) -> Dict[str, Any]:
    """Disrupt the single node with the highest betweenness centrality at severity 0.9.

    Simulates a targeted attack on the most connected/critical point in the
    supply chain.  Betweenness centrality identifies the node through which the
    most shortest paths flow; its removal maximally disrupts information and
    material routing across the entire network.  G is not modified.

    Args:
        G: DTNet DiGraph with ``layer`` and ``twin`` attributes on every node.

    Returns:
        Dict with keys ``name`` (str), ``description`` (str),
        ``disrupted_nodes`` (list of one node ID), ``severity`` (float 0.9).

    Raises:
        ValueError: If G contains no nodes.
    """
    if G.number_of_nodes() == 0:
        raise ValueError("Graph is empty — no nodes to attack.")
    centrality: Dict[str, float] = nx.betweenness_centrality(
        G, normalized=_BETWEENNESS_NORMALISED
    )
    target: str = max(centrality, key=centrality.__getitem__)
    return {
        "name": "critical_hub_failure",
        "description": (
            f"Targeted attack on node '{target}', the highest-betweenness hub "
            f"(centrality={centrality[target]:.4f}) in the supply chain."
        ),
        "disrupted_nodes": [target],
        "severity": SEVERITY_CRITICAL_HUB,
    }


# ---------------------------------------------------------------------------
# Scenario 7 — Global supplier cascade (all suppliers simultaneously)
# ---------------------------------------------------------------------------


def scenario_supplier_cascade(G: nx.DiGraph) -> Dict[str, Any]:
    """Disrupt ALL supplier-layer nodes simultaneously at severity 0.7.

    Simulates a global raw material shortage affecting the entire upstream of
    the supply chain — e.g. a commodity crisis or simultaneous export
    restrictions across multiple source regions.  G is not modified.

    Args:
        G: DTNet DiGraph with ``layer`` and ``twin`` attributes on every node.

    Returns:
        Dict with keys ``name`` (str), ``description`` (str),
        ``disrupted_nodes`` (sorted list of all supplier node IDs),
        ``severity`` (float 0.7).

    Raises:
        ValueError: If the graph has no supplier-layer nodes.
    """
    supplier_nodes: List[str] = [
        nid for nid, d in G.nodes(data=True) if d.get("layer") == LAYER_SUPPLIER
    ]
    if not supplier_nodes:
        raise ValueError("Graph contains no nodes with layer='supplier'.")
    return {
        "name": "supplier_cascade",
        "description": (
            f"Global raw-material shortage: all {len(supplier_nodes)} supplier "
            "nodes disrupted simultaneously."
        ),
        "disrupted_nodes": sorted(supplier_nodes),
        "severity": SEVERITY_SUPPLIER_CASCADE,
    }


# ---------------------------------------------------------------------------
# Scenario 8 — Bottleneck plant failure (highest in-degree among plants)
# ---------------------------------------------------------------------------


def scenario_bottleneck_plant(G: nx.DiGraph) -> Dict[str, Any]:
    """Disrupt the plant node with the highest in-degree among plant-layer nodes.

    Simulates failure of the most depended-upon production facility — the plant
    that receives inputs from the largest number of upstream nodes, and whose
    shutdown therefore starves the widest share of downstream output.
    G is not modified.

    Args:
        G: DTNet DiGraph with ``layer`` and ``twin`` attributes on every node.

    Returns:
        Dict with keys ``name`` (str), ``description`` (str),
        ``disrupted_nodes`` (single-element list with the plant node ID),
        ``severity`` (float 0.85).

    Raises:
        ValueError: If the graph has no plant-layer nodes.
    """
    plant_nodes: List[str] = [
        nid for nid, d in G.nodes(data=True) if d.get("layer") == LAYER_PLANT
    ]
    if not plant_nodes:
        raise ValueError("Graph contains no nodes with layer='plant'.")
    target: str = max(plant_nodes, key=lambda nid: G.in_degree(nid))
    return {
        "name": "bottleneck_plant",
        "description": (
            f"Failure of plant '{target}', the most depended-upon production "
            f"facility (in-degree={G.in_degree(target)})."
        ),
        "disrupted_nodes": [target],
        "severity": SEVERITY_BOTTLENECK_PLANT,
    }
