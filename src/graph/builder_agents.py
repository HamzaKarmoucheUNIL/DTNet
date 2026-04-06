"""builder_agents.py — Agent factory functions for the graph builder.

Internal module used exclusively by ``builder.py``. Do not import from
outside ``src/graph``; use ``build_graph`` from ``builder.py`` instead.

Each ``_make_*_agent`` function instantiates the correct DigitalTwinAgent
subclass from a NodeSpec, using real sensor data for machines and calibrated
synthetic values for all other node types.
"""

from __future__ import annotations

from typing import Dict

from src.graph.topology_specs import NodeSpec
from src.agents.base_agent import DigitalTwinAgent
from src.agents.machine_agent import MachineAgent
from src.agents.supplier_agent import SupplierAgent
from src.agents.plant_agent import PlantAgent
from src.agents.logistics_agent import LogisticsAgent
from src.agents.distribution_agent import DistributionAgent


def _make_machine_agent(
    spec: NodeSpec,
    sensor_lookup: Dict[str, Dict[str, float]],
) -> MachineAgent:
    """Instantiate a MachineAgent, overriding sensor defaults with dataset means.

    NodeSpec attributes provide the baseline; any sensor field present in
    ``sensor_lookup`` for this asset_tag replaces the corresponding default.

    Args:
        spec: NodeSpec with ``layer == "machine"``.
        sensor_lookup: Per-asset_tag sensor mean dict from
            ``_extract_machine_sensor_means`` in builder.py.

    Returns:
        Fully initialised MachineAgent.
    """
    attrs: Dict = dict(spec.attributes)
    for field_name, value in sensor_lookup.get(spec.node_id, {}).items():
        attrs[field_name] = value

    return MachineAgent(
        node_id=spec.node_id,
        node_type="machine",
        capacity=float(attrs.get("capacity", 1.0)),
        throughput=float(attrs.get("throughput", 1.0)),
        failure_prob=float(attrs.get("failure_prob", 0.08)),
        temp_bearing=float(attrs.get("temp_bearing", 60.0)),
        temp_motor=float(attrs.get("temp_motor", 70.0)),
        vibration_h=float(attrs.get("vibration_h", 1.0)),
        vibration_v=float(attrs.get("vibration_v", 1.0)),
        oil_pressure=float(attrs.get("oil_pressure", 50.0)),
        load_pct=float(attrs.get("load_pct", 0.8)),
        power_kw=float(attrs.get("power_kw", 50.0)),
        rpm=float(attrs.get("rpm", 1500.0)),
        breakdown_flag=bool(attrs.get("breakdown_flag", False)),
        parts_required=list(attrs.get("parts_required", [])),
    )


def _make_supplier_agent(spec: NodeSpec) -> SupplierAgent:
    """Instantiate a SupplierAgent from NodeSpec attributes.

    Args:
        spec: NodeSpec with ``layer == "supplier"``.

    Returns:
        Fully initialised SupplierAgent with calibrated synthetic values.
    """
    attrs: Dict = spec.attributes
    return SupplierAgent(
        node_id=spec.node_id,
        node_type="supplier",
        capacity=float(attrs.get("capacity", 1.0)),
        throughput=float(attrs.get("throughput", 1.0)),
        failure_prob=float(attrs.get("failure_prob", 0.05)),
        delivery_reliability=float(attrs.get("delivery_reliability", 0.95)),
        lead_time_days=float(attrs.get("lead_time_days", 7.0)),
        defect_rate=float(attrs.get("defect_rate", 0.02)),
        parts_supplied=list(attrs.get("parts_supplied", [])),
        cost_per_unit=float(attrs.get("cost_per_unit", 10.0)),
    )


def _make_plant_agent(spec: NodeSpec) -> PlantAgent:
    """Instantiate a PlantAgent from NodeSpec attributes.

    Args:
        spec: NodeSpec with ``layer == "plant"``.

    Returns:
        Fully initialised PlantAgent.
    """
    attrs: Dict = spec.attributes
    return PlantAgent(
        node_id=spec.node_id,
        node_type="plant",
        capacity=float(attrs.get("capacity", 1.0)),
        throughput=float(attrs.get("throughput", 1.0)),
        failure_prob=float(attrs.get("failure_prob", 0.04)),
        production_rate=float(attrs.get("production_rate", 0.9)),
        quality_rate=float(attrs.get("quality_rate", 0.95)),
        num_active_machines=int(attrs.get("num_active_machines", 0)),
        machines=list(attrs.get("machines", [])),
    )


def _make_logistics_agent(spec: NodeSpec) -> LogisticsAgent:
    """Instantiate a LogisticsAgent from NodeSpec attributes.

    Args:
        spec: NodeSpec with ``layer == "logistics"``.

    Returns:
        Fully initialised LogisticsAgent with calibrated synthetic values.
    """
    attrs: Dict = spec.attributes
    return LogisticsAgent(
        node_id=spec.node_id,
        node_type="logistics",
        capacity=float(attrs.get("capacity", 1.0)),
        throughput=float(attrs.get("throughput", 1.0)),
        failure_prob=float(attrs.get("failure_prob", 0.03)),
        transit_time_days=float(attrs.get("transit_time_days", 3.0)),
        warehouse_capacity=float(attrs.get("warehouse_capacity", 0.8)),
        route_reliability=float(attrs.get("route_reliability", 0.95)),
        backlog=float(attrs.get("backlog", 0.0)),
    )


def _make_distribution_agent(spec: NodeSpec) -> DistributionAgent:
    """Instantiate a DistributionAgent from NodeSpec attributes.

    Args:
        spec: NodeSpec with ``layer == "distribution"``.

    Returns:
        Fully initialised DistributionAgent with calibrated synthetic values.
    """
    attrs: Dict = spec.attributes
    return DistributionAgent(
        node_id=spec.node_id,
        node_type="distribution",
        capacity=float(attrs.get("capacity", 1.0)),
        throughput=float(attrs.get("throughput", 1.0)),
        failure_prob=float(attrs.get("failure_prob", 0.03)),
        demand_variability=float(attrs.get("demand_variability", 0.2)),
        fulfillment_rate=float(attrs.get("fulfillment_rate", 0.95)),
        stock_level=float(attrs.get("stock_level", 0.8)),
        delivery_delay_days=float(attrs.get("delivery_delay_days", 1.0)),
    )


def make_agent(
    spec: NodeSpec,
    sensor_lookup: Dict[str, Dict[str, float]],
) -> DigitalTwinAgent:
    """Dispatch to the correct agent factory based on NodeSpec layer.

    Args:
        spec: NodeSpec describing the node to create.
        sensor_lookup: Pre-computed per-machine sensor means; only consulted
            for machine-layer nodes.

    Returns:
        The appropriate DigitalTwinAgent subclass instance.

    Raises:
        ValueError: If ``spec.layer`` is not a recognised node type.
    """
    if spec.layer == "machine":
        return _make_machine_agent(spec, sensor_lookup)
    if spec.layer == "supplier":
        return _make_supplier_agent(spec)
    if spec.layer == "plant":
        return _make_plant_agent(spec)
    if spec.layer == "logistics":
        return _make_logistics_agent(spec)
    if spec.layer == "distribution":
        return _make_distribution_agent(spec)
    raise ValueError(
        f"[builder] Unknown node layer '{spec.layer}' for node '{spec.node_id}'. "
        f"Expected one of: machine, supplier, plant, logistics, distribution."
    )
