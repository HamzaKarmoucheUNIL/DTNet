"""topology_specs.py — NodeSpec and EdgeSpec data containers.

Shared between topology_nodes.py, topology_edges.py, and topology.py.
Import these from here, not from topology.py, to avoid circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

# Default edge attribute values
DEFAULT_FLOW_CAPACITY: float = 0.9
DEFAULT_CRITICALITY: float = 0.7
DEFAULT_LATENCY_DAYS: int = 2


@dataclass
class NodeSpec:
    """Specification for a single node in the supply-chain graph.

    Attributes:
        node_id: Unique string identifier used as the NetworkX node key.
        layer: One of 'supplier' | 'logistics' | 'plant' | 'machine' |
            'distribution'.
        attributes: Dict of initial attribute values for this node.
            Keys align with the corresponding agent dataclass fields.
    """

    node_id: str
    layer: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeSpec:
    """Specification for a single directed edge in the supply-chain graph.

    Attributes:
        source: Node ID of the edge origin.
        target: Node ID of the edge destination.
        edge_type: One of 'material_flow' | 'operational' | 'process_chain' |
            'shared_part_dependency'.
        flow_capacity: Normalised flow capacity of this edge [0, 1].
        criticality_weight: How critical this connection is [0, 1]. Used by
            the cascading failure propagation formula.
        latency_days: Timesteps before a disruption propagates along this edge.
    """

    source: str
    target: str
    edge_type: str
    flow_capacity: float = DEFAULT_FLOW_CAPACITY
    criticality_weight: float = DEFAULT_CRITICALITY
    latency_days: int = DEFAULT_LATENCY_DAYS
    shared_parts_count: int = 0
