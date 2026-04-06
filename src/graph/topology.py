"""topology.py — Public API for DTNet supply-chain topology inference.

This module is PURE DATA COMPUTATION. It reads an ``EntityMappings`` instance
and returns two plain lists:

- ``nodes``: one ``NodeSpec`` per node, carrying its ID, layer, and initial
  attribute values.
- ``edges``: one ``EdgeSpec`` per directed edge, carrying source, target,
  edge type, and weight attributes.

No NetworkX objects are created here. ``builder.py`` assembles those from
the specs produced by ``infer_topology``.

Node layers and counts
----------------------
- Supplier     : one per ``part_family``
- Logistics    : 2 hubs (HUB_A, HUB_B) — suppliers split round-robin
- Plant        : one per ``plant_code``
- Machine      : one per ``asset_tag``, linked to its plant
- Distribution : 2 nodes (DIST_MAIN for all plants, DIST_EXPORT for one)

Edge types (directed)
---------------------
- ``material_flow``           : supplier→hub, hub→plant, plant→dist
- ``operational``             : plant→machine
- ``process_chain``           : machine→next machine in same plant
- ``shared_part_dependency``  : bidirectional between cross-plant machine pairs

Implementation is split across three internal modules:
- ``topology_specs.py``  — NodeSpec and EdgeSpec dataclasses
- ``topology_nodes.py``  — node builder functions
- ``topology_edges.py``  — edge builder functions
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List, Tuple

from src.data.entity_mapping import EntityMappings
from src.graph.topology_specs import EdgeSpec, NodeSpec
from src.graph.topology_nodes import (
    build_supplier_nodes,
    build_logistics_nodes,
    build_plant_nodes,
    build_machine_nodes,
    build_distribution_nodes,
)
from src.graph.topology_edges import (
    build_supplier_to_hub_edges,
    build_hub_to_plant_edges,
    build_plant_to_machine_edges,
    build_process_chain_edges,
    build_cross_plant_edges,
    build_plant_to_distribution_edges,
)

np.random.seed(42)
torch.manual_seed(42)

# Re-export specs so callers only need to import from topology.py
__all__ = ["NodeSpec", "EdgeSpec", "infer_topology", "print_topology_summary"]


def infer_topology(
    em: EntityMappings,
) -> Tuple[List[NodeSpec], List[EdgeSpec]]:
    """Infer the full DTNet supply-chain topology from entity mappings.

    Single entry point for this module. Assembles all node and edge
    specifications without constructing any NetworkX objects. Pass the
    returned specs to ``builder.build_graph`` to get a ``nx.DiGraph``.

    Args:
        em: Fully populated ``EntityMappings`` as returned by
            ``build_entity_mappings``.

    Returns:
        A tuple ``(nodes, edges)`` where:

        - ``nodes`` is a flat list of ``NodeSpec`` in layer order:
          suppliers → logistics → plants → machines → distribution.
        - ``edges`` is a flat list of ``EdgeSpec`` covering all six edge
          sets in the order listed above.
    """
    nodes: List[NodeSpec] = (
        build_supplier_nodes(em)
        + build_logistics_nodes()
        + build_plant_nodes(em)
        + build_machine_nodes(em)
        + build_distribution_nodes()
    )

    edges: List[EdgeSpec] = (
        build_supplier_to_hub_edges(em)
        + build_hub_to_plant_edges(em)
        + build_plant_to_machine_edges(em)
        + build_process_chain_edges(em)
        + build_cross_plant_edges(em)
        + build_plant_to_distribution_edges(em)
    )

    return nodes, edges


def print_topology_summary(nodes: List[NodeSpec], edges: List[EdgeSpec]) -> None:
    """Print a structured summary of the inferred topology.

    Args:
        nodes: Node list as returned by ``infer_topology``.
        edges: Edge list as returned by ``infer_topology``.
    """
    layer_counts: Dict[str, int] = {}
    for n in nodes:
        layer_counts[n.layer] = layer_counts.get(n.layer, 0) + 1

    edge_type_counts: Dict[str, int] = {}
    for e in edges:
        edge_type_counts[e.edge_type] = edge_type_counts.get(e.edge_type, 0) + 1

    sep: str = "=" * 55
    print(f"\n{sep}")
    print("  TOPOLOGY SUMMARY")
    print(sep)
    print(f"  Total nodes : {len(nodes):,}")
    for layer in ["supplier", "logistics", "plant", "machine", "distribution"]:
        print(f"    {layer:<20}: {layer_counts.get(layer, 0):>6,}")
    print(f"\n  Total edges : {len(edges):,}")
    for etype in ["material_flow", "operational", "process_chain", "shared_part_dependency"]:
        print(f"    {etype:<30}: {edge_type_counts.get(etype, 0):>6,}")
    print(sep + "\n")
