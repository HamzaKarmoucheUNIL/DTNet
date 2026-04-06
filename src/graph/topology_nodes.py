"""topology_nodes.py — Node specification builders for the DTNet topology.

Internal module used by ``topology.py``. Do not import directly from outside
``src/graph``; use ``infer_topology`` from ``topology.py`` instead.
"""

from __future__ import annotations

from typing import List

from src.data.entity_mapping import EntityMappings
from src.graph.topology_specs import NodeSpec

# Fixed node IDs for logistics and distribution layers
HUB_IDS: List[str] = ["HUB_A", "HUB_B"]
DIST_IDS: List[str] = ["DIST_MAIN", "DIST_EXPORT"]


def build_supplier_nodes(em: EntityMappings) -> List[NodeSpec]:
    """Create one supplier node per part family.

    Node ID format: ``SUP_<part_family>``.

    Args:
        em: Populated EntityMappings from entity_mapping.py.

    Returns:
        List of NodeSpec objects sorted by family name for determinism.
    """
    nodes: List[NodeSpec] = []
    for family in em.part_families:
        parts: List[str] = em.family_to_parts.get(family, [])
        nodes.append(NodeSpec(
            node_id=f"SUP_{family}",
            layer="supplier",
            attributes={
                "delivery_reliability": 0.95,
                "lead_time_days": 7.0,
                "defect_rate": 0.02,
                "parts_supplied": parts,
                "cost_per_unit": 10.0,
                "capacity": 1.0,
                "throughput": 1.0,
                "failure_prob": 0.05,
            },
        ))
    return nodes


def build_logistics_nodes() -> List[NodeSpec]:
    """Create the two fixed logistics hub nodes (HUB_A, HUB_B).

    Returns:
        List of two NodeSpec objects for HUB_A and HUB_B.
    """
    nodes: List[NodeSpec] = []
    for hub_id in HUB_IDS:
        nodes.append(NodeSpec(
            node_id=hub_id,
            layer="logistics",
            attributes={
                "transit_time_days": 3.0,
                "warehouse_capacity": 0.8,
                "route_reliability": 0.95,
                "backlog": 0.0,
                "capacity": 1.0,
                "throughput": 1.0,
                "failure_prob": 0.03,
            },
        ))
    return nodes


def build_plant_nodes(em: EntityMappings) -> List[NodeSpec]:
    """Create one plant node per plant_code in the dataset.

    Args:
        em: Populated EntityMappings.

    Returns:
        List of NodeSpec objects sorted by plant_code for determinism.
    """
    nodes: List[NodeSpec] = []
    for plant_code in em.plant_codes:
        machines: List[str] = em.plant_to_machines.get(plant_code, [])
        nodes.append(NodeSpec(
            node_id=plant_code,
            layer="plant",
            attributes={
                "production_rate": 0.9,
                "quality_rate": 0.95,
                "num_active_machines": len(machines),
                "machines": machines,
                "capacity": 1.0,
                "throughput": 1.0,
                "failure_prob": 0.04,
            },
        ))
    return nodes


def build_machine_nodes(em: EntityMappings) -> List[NodeSpec]:
    """Create one machine node per asset_tag in the dataset.

    Args:
        em: Populated EntityMappings.

    Returns:
        List of NodeSpec objects sorted by asset_tag for determinism.
    """
    nodes: List[NodeSpec] = []
    for asset_tag in em.asset_tags:
        parts: List[str] = em.machine_to_parts.get(asset_tag, [])
        nodes.append(NodeSpec(
            node_id=asset_tag,
            layer="machine",
            attributes={
                "temp_bearing": 60.0,
                "temp_motor": 70.0,
                "vibration_h": 1.0,
                "vibration_v": 1.0,
                "oil_pressure": 50.0,
                "load_pct": 0.8,
                "power_kw": 50.0,
                "rpm": 1500.0,
                "breakdown_flag": False,
                "parts_required": parts,
                "capacity": 1.0,
                "throughput": 1.0,
                "failure_prob": 0.08,
            },
        ))
    return nodes


def build_distribution_nodes() -> List[NodeSpec]:
    """Create the two fixed distribution nodes (DIST_MAIN, DIST_EXPORT).

    Returns:
        List of two NodeSpec objects.
    """
    nodes: List[NodeSpec] = []
    for dist_id in DIST_IDS:
        nodes.append(NodeSpec(
            node_id=dist_id,
            layer="distribution",
            attributes={
                "demand_variability": 0.2,
                "fulfillment_rate": 0.95,
                "stock_level": 0.8,
                "delivery_delay_days": 1.0,
                "capacity": 1.0,
                "throughput": 1.0,
                "failure_prob": 0.03,
            },
        ))
    return nodes
