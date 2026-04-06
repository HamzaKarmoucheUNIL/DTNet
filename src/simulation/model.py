"""model.py — DTNetModel: Mesa simulation model for cascading disruption.

Wraps every NetworkX node (DigitalTwinAgent) in a TwinMesaAgent and runs a
SimultaneousActivation scheduler.  Disruptions propagate through the network
each step using the criticality-weighted formula from CODING_PATTERNS.md.
"""

from __future__ import annotations

import numpy as np
import torch
import networkx as nx
import mesa
from mesa import DataCollector
from mesa.time import SimultaneousActivation
from typing import Any, Dict, List, Set

from src.agents.base_agent import DigitalTwinAgent

np.random.seed(42)
torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PROPAGATION_DECAY: float = 0.6
DEFAULT_THRESHOLD: float = 0.15


# ---------------------------------------------------------------------------
# Mesa wrapper agent
# ---------------------------------------------------------------------------


class TwinMesaAgent(mesa.Agent):
    """Mesa SimultaneousActivation wrapper for a single DigitalTwinAgent node.

    step() reads predecessor disruption state and stores a pending severity;
    advance() applies it if above the model threshold, then ticks the twin.

    Attributes:
        node_id: NetworkX node key this wrapper represents.
        twin: Underlying DigitalTwinAgent dataclass instance.
        _pending_severity: Severity computed in step(), applied in advance().
    """

    def __init__(
        self,
        unique_id: int,
        model: "DTNetModel",
        node_id: str,
        twin: DigitalTwinAgent,
    ) -> None:
        """Initialise the Mesa wrapper for one digital twin node.

        Args:
            unique_id: Integer identifier required by Mesa.
            model: Parent DTNetModel instance.
            node_id: NetworkX node key matching G.nodes.
            twin: The DigitalTwinAgent backing this node.
        """
        super().__init__(unique_id, model)
        self.node_id: str = node_id
        self.twin: DigitalTwinAgent = twin
        self._pending_severity: float = 0.0

    def step(self) -> None:
        """Compute max incoming disruption severity from disrupted predecessors.

        Uses criticality_weight (COMMON_MISTAKES #11) and propagation_decay
        per the formula in CODING_PATTERNS.md.  Stores result in
        _pending_severity for advance() to act on.
        """
        G: nx.DiGraph = self.model.G
        max_incoming: float = 0.0

        for pred_id in G.predecessors(self.node_id):
            pred_twin: DigitalTwinAgent = G.nodes[pred_id]["twin"]
            if not pred_twin.is_disrupted:
                continue

            edge_data: Dict[str, Any] = G.edges[pred_id, self.node_id]
            criticality: float = float(edge_data.get("criticality_weight", 1.0))

            # Propagation formula — CODING_PATTERNS.md §Simulation Patterns
            incoming: float = (
                pred_twin.disruption_severity
                * criticality
                * self.model.propagation_decay
            )
            vulnerability: float = 1.0 - self.twin.compute_health_score()
            adjusted: float = incoming * (1.0 + vulnerability * 0.5)

            if adjusted > max_incoming:
                max_incoming = adjusted

        self._pending_severity = max_incoming

    def advance(self) -> None:
        """Apply pending disruption if above threshold; tick the twin's counter."""
        current_step: int = self.model.schedule.steps
        if self._pending_severity > self.model.threshold:
            self.twin.apply_disruption(self._pending_severity, current_step)
        self.twin.step()
        self._pending_severity = 0.0


# ---------------------------------------------------------------------------
# Main simulation model
# ---------------------------------------------------------------------------


class DTNetModel(mesa.Model):
    """Mesa simulation model for DTNet cascading disruption scenarios.

    Wraps a NetworkX DiGraph and orchestrates SimultaneousActivation.
    Full per-step state is recorded by a DataCollector (COMMON_MISTAKES #6).
    Call reset() between runs to prevent state leakage (COMMON_MISTAKES #10).

    Attributes:
        G: DiGraph where every node carries {"twin": DigitalTwinAgent}.
        propagation_decay: Decay factor applied to propagating severity.
        threshold: Minimum adjusted severity to trigger propagation.
        schedule: SimultaneousActivation scheduler.
        datacollector: Records model- and agent-level stats each step.
        _agent_map: Mapping node_id → TwinMesaAgent.
        _disrupted_prev: Disrupted node IDs before the current step.
        _newly_disrupted: Node IDs first disrupted in the current step.
    """

    def __init__(
        self,
        G: nx.DiGraph,
        propagation_decay: float = DEFAULT_PROPAGATION_DECAY,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        """Initialise the DTNet simulation model.

        Creates one TwinMesaAgent per graph node, registers them with the
        scheduler, and sets up the DataCollector.

        Args:
            G: NetworkX DiGraph with "twin" attribute on every node.
            propagation_decay: Decay multiplier for propagating severity [0,1].
            threshold: Minimum adjusted severity to disrupt a successor [0,1].
        """
        super().__init__()
        np.random.seed(42)

        self.G: nx.DiGraph = G
        self.propagation_decay: float = propagation_decay
        self.threshold: float = threshold

        self.schedule: SimultaneousActivation = SimultaneousActivation(self)
        self._agent_map: Dict[str, TwinMesaAgent] = {}
        self._disrupted_prev: Set[str] = set()
        self._newly_disrupted: List[str] = []

        for uid, (node_id, data) in enumerate(G.nodes(data=True)):
            twin: DigitalTwinAgent = data["twin"]
            agent: TwinMesaAgent = TwinMesaAgent(uid, self, node_id, twin)
            self.schedule.add(agent)
            self._agent_map[node_id] = agent

        self.datacollector: DataCollector = self._make_datacollector()

    def _make_datacollector(self) -> DataCollector:
        """Build a DataCollector with all required model and agent reporters.

        Returns:
            Configured mesa.DataCollector instance.
        """
        return DataCollector(
            model_reporters={
                "num_disrupted": lambda m: sum(
                    1 for _, d in m.G.nodes(data=True) if d["twin"].is_disrupted
                ),
                "avg_health": lambda m: float(
                    np.mean(
                        [d["twin"].compute_health_score() for _, d in m.G.nodes(data=True)]
                    )
                ),
                "avg_capacity": lambda m: float(
                    np.mean([d["twin"].capacity for _, d in m.G.nodes(data=True)])
                ),
                "newly_disrupted": lambda m: list(m._newly_disrupted),
            },
            agent_reporters={
                "disruption_severity": lambda a: a.twin.disruption_severity,
                "health_score": lambda a: a.twin.compute_health_score(),
                "capacity": lambda a: a.twin.capacity,
            },
        )

    def step(self) -> None:
        """Advance the simulation by one timestep.

        Snapshots pre-step disruption state, runs the scheduler (propagating
        disruptions), detects newly disrupted nodes, and records all stats.
        """
        self._disrupted_prev = {
            node_id
            for node_id, data in self.G.nodes(data=True)
            if data["twin"].is_disrupted
        }

        self.schedule.step()

        disrupted_now: Set[str] = {
            node_id
            for node_id, data in self.G.nodes(data=True)
            if data["twin"].is_disrupted
        }
        self._newly_disrupted = sorted(disrupted_now - self._disrupted_prev)

        self.datacollector.collect(self)

    def inject_disruption(self, node_id: str, severity: float) -> None:
        """Inject an initial disruption at a specific graph node.

        Call before step() to seed the simulation with a starting failure.

        Args:
            node_id: NetworkX node key of the node to disrupt.
            severity: Disruption magnitude in [0, 1].

        Raises:
            KeyError: If node_id is not present in the graph.
        """
        if node_id not in self.G.nodes:
            raise KeyError(f"Node '{node_id}' not found in the graph.")
        twin: DigitalTwinAgent = self.G.nodes[node_id]["twin"]
        twin.apply_disruption(severity, self.schedule.steps)

    def reset(self) -> None:
        """Restore all agents to baseline; clear history (COMMON_MISTAKES #10)."""
        for _, data in self.G.nodes(data=True):
            data["twin"].reset()

        self._disrupted_prev = set()
        self._newly_disrupted = []
        self.schedule.steps = 0
        self.schedule.time = 0
        self.datacollector = self._make_datacollector()

    def get_history(self) -> List[Dict[str, Any]]:
        """Return simulation history as a list of per-timestep dicts.

        Merges model-level aggregates with per-node snapshots (format from
        CODING_PATTERNS.md §Simulation Patterns).

        Returns:
            List of dicts per recorded timestep with keys: 'timestep',
            'newly_disrupted', 'total_disrupted', 'network_health',
            'total_capacity', 'node_states' (node_id → {disruption_severity,
            health_score, capacity}).
        """
        model_df = self.datacollector.get_model_vars_dataframe()
        agent_df = self.datacollector.get_agent_vars_dataframe()

        agent_step_levels: set = set(agent_df.index.get_level_values(0).unique())
        id_to_node: Dict[int, str] = {
            a.unique_id: a.node_id for a in self.schedule.agents
        }

        history: List[Dict[str, Any]] = []

        for step_idx in model_df.index:
            node_states: Dict[str, Dict[str, float]] = {}

            if step_idx in agent_step_levels:
                step_df = agent_df.loc[step_idx]
                for agent_id, node_id in id_to_node.items():
                    if agent_id in step_df.index:
                        row = step_df.loc[agent_id]
                        node_states[node_id] = {
                            "disruption_severity": float(row["disruption_severity"]),
                            "health_score": float(row["health_score"]),
                            "capacity": float(row["capacity"]),
                        }

            total_disrupted: List[str] = sorted(
                nid
                for nid, state in node_states.items()
                if state["disruption_severity"] > 0.0
            )

            history.append(
                {
                    "timestep": int(step_idx),
                    "newly_disrupted": list(model_df.loc[step_idx, "newly_disrupted"]),
                    "total_disrupted": total_disrupted,
                    "network_health": float(model_df.loc[step_idx, "avg_health"]),
                    "total_capacity": float(model_df.loc[step_idx, "avg_capacity"]),
                    "node_states": node_states,
                }
            )

        return history
