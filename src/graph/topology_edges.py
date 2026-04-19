"""topology_edges.py — Edge specification builders for the DTNet topology.

Internal module used by ``topology.py``. Do not import directly from outside
``src/graph``; use ``infer_topology`` from ``topology.py`` instead.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from src.data.entity_mapping import EntityMappings
from src.graph.topology_specs import EdgeSpec

# Hub IDs must stay consistent with topology_nodes.py
HUB_IDS: List[str] = ["HUB_A", "HUB_B"]

# Index into sorted plant_codes that feeds DIST_EXPORT
EXPORT_PLANT_INDEX: int = 0

# Criticality weight per edge type
CRITICALITY: Dict[str, float] = {
    "material_flow": 0.8,
    "operational": 0.9,
    "process_chain": 0.7,
    "shared_part_dependency": 0.6,
}

# Propagation latency (days) per edge type
LATENCY: Dict[str, int] = {
    "material_flow": 2,
    "operational": 1,
    "process_chain": 1,
    "shared_part_dependency": 3,
}


def build_supplier_to_hub_edges(em: EntityMappings) -> List[EdgeSpec]:
    """Wire each supplier to a logistics hub using round-robin assignment.

    Suppliers are iterated in sorted part_family order (deterministic) and
    assigned to HUB_A or HUB_B alternately.

    Args:
        em: Populated EntityMappings.

    Returns:
        List of EdgeSpec objects (material_flow), one per part family.
    """
    edges: List[EdgeSpec] = []
    for idx, family in enumerate(em.part_families):
        hub_id: str = HUB_IDS[idx % len(HUB_IDS)]
        edges.append(EdgeSpec(
            source=f"SUP_{family}",
            target=hub_id,
            edge_type="material_flow",
            flow_capacity=0.9,
            criticality_weight=CRITICALITY["material_flow"],
            latency_days=LATENCY["material_flow"],
        ))
    return edges


def build_hub_to_plant_edges(em: EntityMappings) -> List[EdgeSpec]:
    """Connect every logistics hub to every plant (full fan-out).

    HUB_A is the primary hub and gets a slightly higher criticality weight
    than HUB_B to reflect its priority role.

    Args:
        em: Populated EntityMappings.

    Returns:
        List of EdgeSpec objects (material_flow).
    """
    edges: List[EdgeSpec] = []
    for hub_idx, hub_id in enumerate(HUB_IDS):
        criticality: float = 0.85 if hub_idx == 0 else 0.75
        for plant_code in em.plant_codes:
            edges.append(EdgeSpec(
                source=hub_id,
                target=plant_code,
                edge_type="material_flow",
                flow_capacity=0.9,
                criticality_weight=criticality,
                latency_days=LATENCY["material_flow"],
            ))
    return edges


def build_plant_to_machine_edges(em: EntityMappings) -> List[EdgeSpec]:
    """Connect each plant to all machines located inside it (operational edges).

    Args:
        em: Populated EntityMappings.

    Returns:
        List of EdgeSpec objects (operational).
    """
    edges: List[EdgeSpec] = []
    for plant_code, machines in em.plant_to_machines.items():
        for asset_tag in machines:
            edges.append(EdgeSpec(
                source=plant_code,
                target=asset_tag,
                edge_type="operational",
                flow_capacity=1.0,
                criticality_weight=CRITICALITY["operational"],
                latency_days=LATENCY["operational"],
            ))
    return edges


def build_process_chain_edges(em: EntityMappings) -> List[EdgeSpec]:
    """Chain machines within each plant sequentially (process_chain edges).

    Machines are sorted by asset_tag within each plant and linked in order:
    machine[0] → machine[1] → … → machine[n-1].

    Args:
        em: Populated EntityMappings.

    Returns:
        List of EdgeSpec objects (process_chain). Plants with fewer than 2
        machines produce no edges.
    """
    edges: List[EdgeSpec] = []
    for machines in em.plant_to_machines.values():
        if len(machines) < 2:
            continue
        for i in range(len(machines) - 1):
            edges.append(EdgeSpec(
                source=machines[i],
                target=machines[i + 1],
                edge_type="process_chain",
                flow_capacity=0.85,
                criticality_weight=CRITICALITY["process_chain"],
                latency_days=LATENCY["process_chain"],
            ))
    return edges


def build_cross_plant_edges(em: EntityMappings) -> List[EdgeSpec]:
    """Add unidirectional shared_part_dependency edges for cross-plant parts.

    Tries part_no-level sharing first (``em.cross_plant_parts``). If that
    mapping is empty (part_nos are plant-specific, as in updated_data.csv),
    falls back to part_family-level sharing (``em.cross_plant_families``).

    For each (key, plant_pair): one representative machine is selected per
    plant — the machine that uses the most parts belonging to that key —
    then a single directed edge is emitted from the representative in the
    alphabetically smaller plant to the one in the larger plant.

    Multiple shared keys between the same representative pair are collapsed
    into a single edge whose ``shared_parts_count`` equals the number of
    shared keys and whose ``criticality_weight`` equals
    ``shared_parts_count / max_shared_parts`` across all pairs.

    Args:
        em: Populated EntityMappings.

    Returns:
        List of EdgeSpec objects (shared_part_dependency), one per
        unordered representative machine pair.
    """
    # Choose the sharing index: prefer part_no, fall back to part_family.
    if em.cross_plant_parts:
        sharing_index: Dict[str, List[str]] = em.cross_plant_parts

        def pick_rep(plant_code: str, key: str) -> Optional[str]:
            """Machine in plant_code that uses part_no key (most total parts)."""
            candidates: List[str] = [
                m for m in em.plant_to_machines.get(plant_code, [])
                if key in em.machine_to_parts.get(m, [])
            ]
            if not candidates:
                return None
            return max(candidates, key=lambda m: len(em.machine_to_parts.get(m, [])))
    else:
        sharing_index = em.cross_plant_families

        def pick_rep(plant_code: str, key: str) -> Optional[str]:  # type: ignore[misc]
            """Machine in plant_code with the most parts in part_family key."""
            candidates: List[str] = [
                m for m in em.plant_to_machines.get(plant_code, [])
                if key in em.machine_to_families.get(m, [])
            ]
            if not candidates:
                return None
            family_parts: Set[str] = set(em.family_to_parts.get(key, []))
            return max(
                candidates,
                key=lambda m: len(set(em.machine_to_parts.get(m, [])) & family_parts),
            )

    # Pass 1 — count shared keys per unordered representative pair.
    # Direction: representative in smaller plant_code → larger plant_code.
    pair_counts: Dict[Tuple[str, str], int] = {}

    for key, plants in sharing_index.items():
        plant_reps: Dict[str, str] = {}
        for plant_code in plants:
            rep: Optional[str] = pick_rep(plant_code, key)
            if rep is not None:
                plant_reps[plant_code] = rep

        plant_list: List[str] = sorted(plant_reps.keys())
        for i in range(len(plant_list)):
            for j in range(i + 1, len(plant_list)):
                src: str = plant_reps[plant_list[i]]
                tgt: str = plant_reps[plant_list[j]]
                pair_counts[(src, tgt)] = pair_counts.get((src, tgt), 0) + 1

    if not pair_counts:
        return []

    # Pass 2 — emit one EdgeSpec per pair; scale criticality by shared count.
    max_shared: int = max(pair_counts.values())
    edges: List[EdgeSpec] = []
    for (s, t), count in pair_counts.items():
        edges.append(EdgeSpec(
            source=s,
            target=t,
            edge_type="shared_part_dependency",
            flow_capacity=0.7,
            criticality_weight=count / max_shared,
            latency_days=LATENCY["shared_part_dependency"],
            shared_parts_count=count,
        ))
    return edges


def build_plant_to_distribution_edges(em: EntityMappings) -> List[EdgeSpec]:
    """Connect plants to distribution nodes.

    All plants feed DIST_MAIN. The first plant (alphabetically) also feeds
    DIST_EXPORT to model an export channel.

    Args:
        em: Populated EntityMappings.

    Returns:
        List of EdgeSpec objects (material_flow).
    """
    edges: List[EdgeSpec] = []
    export_plant: str = em.plant_codes[EXPORT_PLANT_INDEX]

    for plant_code in em.plant_codes:
        edges.append(EdgeSpec(
            source=plant_code,
            target="DIST_MAIN",
            edge_type="material_flow",
            flow_capacity=0.9,
            criticality_weight=CRITICALITY["material_flow"],
            latency_days=LATENCY["material_flow"],
        ))
        if plant_code == export_plant:
            edges.append(EdgeSpec(
                source=plant_code,
                target="DIST_EXPORT",
                edge_type="material_flow",
                flow_capacity=0.8,
                criticality_weight=0.65,
                latency_days=LATENCY["material_flow"],
            ))
    return edges
