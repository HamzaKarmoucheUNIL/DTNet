"""Base digital twin agent for DTNet simulation.

Defines the abstract DigitalTwinAgent dataclass that all node-type-specific
agents must inherit from. Node-type-specific attributes (e.g. sensor readings
for machines, lead times for suppliers) belong only in subclasses.
"""

from __future__ import annotations

import numpy as np
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

np.random.seed(42)
torch.manual_seed(42)

# Health score weights — shared across all node types
CAPACITY_WEIGHT: float = 0.4
THROUGHPUT_WEIGHT: float = 0.4
FAILURE_PROB_WEIGHT: float = 0.2


@dataclass
class DigitalTwinAgent(ABC):
    """Abstract base class for all digital twin agents in the DTNet simulation.

    Every node in the supply-chain graph is backed by a subclass of this
    dataclass. Common disruption state and health logic live here; attributes
    that are specific to a node type (sensor readings, lead times, etc.) must
    be defined only in the corresponding subclass — never here.

    Attributes:
        node_id: Unique string identifier matching the NetworkX node key.
        node_type: One of 'supplier' | 'plant' | 'machine' | 'logistics' |
            'distribution'.
        capacity: Normalised production/flow capacity in [0, 1].
        throughput: Normalised actual throughput in [0, 1].
        failure_prob: Probability of spontaneous failure in [0, 1].
        is_disrupted: Whether the agent is currently in a disrupted state.
        disruption_severity: Severity of the active disruption in [0, 1];
            0.0 when not disrupted.
        time_disrupted: Number of simulation timesteps the agent has been
            continuously disrupted. Reset to 0 on recovery.
    """

    node_id: str
    node_type: str

    # Operational state — all normalised to [0, 1]
    capacity: float
    throughput: float
    failure_prob: float

    # Disruption state
    is_disrupted: bool = field(default=False)
    disruption_severity: float = field(default=0.0)
    time_disrupted: int = field(default=0)

    # --- private baseline values for reset() --------------------------------
    _init_capacity: float = field(init=False, repr=False)
    _init_throughput: float = field(init=False, repr=False)
    _init_failure_prob: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Cache baseline operational values used by reset()."""
        self._init_capacity = self.capacity
        self._init_throughput = self.throughput
        self._init_failure_prob = self.failure_prob

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def step(self) -> None:
        """Advance the agent by one simulation timestep (called by Mesa)."""
        ...

    # ------------------------------------------------------------------
    # Concrete shared methods
    # ------------------------------------------------------------------

    def compute_health_score(self) -> float:
        """Compute a scalar health score for this agent.

        The score is a weighted combination of capacity, throughput, and the
        complement of failure probability, then penalised by current disruption
        severity.

        Returns:
            A float in [0, 1] where 1.0 represents a fully healthy agent and
            0.0 represents a completely failed / severely disrupted agent.
        """
        base: float = (
            CAPACITY_WEIGHT * self.capacity
            + THROUGHPUT_WEIGHT * self.throughput
            + FAILURE_PROB_WEIGHT * (1.0 - self.failure_prob)
        )
        # Disruption degrades health linearly with severity
        health: float = base * (1.0 - self.disruption_severity)
        return float(np.clip(health, 0.0, 1.0))

    def apply_disruption(self, severity: float, timestep: int) -> None:
        """Apply (or worsen) a disruption on this agent.

        Sets disruption flags, degrades capacity and throughput proportionally
        to severity, and records the timestep at which disruption was applied.
        If the agent is already disrupted, only escalates if the new severity
        is higher than the current one.

        Args:
            severity: Disruption severity in [0, 1]; higher values cause more
                degradation.
            timestep: The current simulation timestep (stored for auditing and
                propagation latency calculations).
        """
        severity = float(np.clip(severity, 0.0, 1.0))

        if self.is_disrupted and severity <= self.disruption_severity:
            # No escalation — existing disruption is already at least as bad
            return

        self.is_disrupted = True
        self.disruption_severity = severity
        self.time_disrupted = timestep

        # Degrade operational state proportionally to severity
        self.capacity = float(np.clip(self._init_capacity * (1.0 - severity), 0.0, 1.0))
        self.throughput = float(np.clip(self._init_throughput * (1.0 - severity), 0.0, 1.0))

    def reset(self) -> None:
        """Restore the agent to its pre-disruption baseline state.

        Clears all disruption flags and resets operational values to the
        initial values captured at construction time. Call this before starting
        a new simulation run to prevent state leakage across runs.
        """
        self.is_disrupted = False
        self.disruption_severity = 0.0
        self.time_disrupted = 0

        self.capacity = self._init_capacity
        self.throughput = self._init_throughput
        self.failure_prob = self._init_failure_prob
