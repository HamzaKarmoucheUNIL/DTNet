"""PlantAgent — digital twin for a manufacturing plant node.

Tracks plant-level production aggregates. Individual machine sensor readings
belong in MachineAgent, not here. The plant health reflects how well machines
inside the plant are running, expressed through production and quality rates.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List

from src.agents.base_agent import DigitalTwinAgent

np.random.seed(42)
torch.manual_seed(42)

# --- Health weight distribution ------------------------------------------
W_PRODUCTION: float = 0.45
W_QUALITY: float = 0.35
W_MACHINE_AVAIL: float = 0.20   # fraction of machines still active


@dataclass
class PlantAgent(DigitalTwinAgent):
    """Digital twin for an entire manufacturing plant.

    Aggregates the state of the machines inside the plant into plant-level KPIs.
    Disruption reduces production rate, degrades quality, and takes machines
    offline.

    Attributes:
        production_rate: Normalised actual vs. target production rate [0, 1].
        quality_rate: Fraction of output meeting quality standards [0, 1].
        num_active_machines: Number of machines currently operational (≥ 0).
        machines: List of machine node IDs that belong to this plant.
    """

    production_rate: float = field(default=0.9)
    quality_rate: float = field(default=0.95)
    num_active_machines: int = field(default=10)
    machines: List[str] = field(default_factory=list)

    # Private baselines for reset()
    _init_production_rate: float = field(init=False, repr=False)
    _init_quality_rate: float = field(init=False, repr=False)
    _init_num_active_machines: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Cache plant-level attribute baselines."""
        super().__post_init__()
        self._init_production_rate = self.production_rate
        self._init_quality_rate = self.quality_rate
        self._init_num_active_machines = self.num_active_machines

    # ------------------------------------------------------------------
    # Health scoring
    # ------------------------------------------------------------------

    def compute_health_score(self) -> float:
        """Compute plant health from production rate, quality rate, and machine availability.

        Machine availability is expressed as the fraction of machines still
        active relative to the initial count. A plant with no active machines
        scores 0 on that component.

        Returns:
            A float in [0, 1]. 1.0 = full production at target quality with
            all machines online.
        """
        production_score: float = float(np.clip(self.production_rate, 0.0, 1.0))
        quality_score: float = float(np.clip(self.quality_rate, 0.0, 1.0))

        if self._init_num_active_machines > 0:
            machine_avail: float = float(
                np.clip(self.num_active_machines / self._init_num_active_machines, 0.0, 1.0)
            )
        else:
            machine_avail = 1.0  # no machines tracked — treat as neutral

        health: float = (
            W_PRODUCTION * production_score
            + W_QUALITY * quality_score
            + W_MACHINE_AVAIL * machine_avail
        )
        return float(np.clip(health, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Disruption
    # ------------------------------------------------------------------

    def apply_disruption(self, severity: float, timestep: int) -> None:
        """Degrade plant KPIs by disruption severity.

        Production rate and quality rate decrease; machines are taken offline
        proportionally. Capacity and throughput are also degraded.

        Args:
            severity: Disruption magnitude in [0, 1].
            timestep: Current simulation timestep.
        """
        severity = float(np.clip(severity, 0.0, 1.0))

        if self.is_disrupted and severity <= self.disruption_severity:
            return

        self.is_disrupted = True
        self.disruption_severity = severity
        self.time_disrupted = timestep

        # Degrade operational state
        self.capacity = float(np.clip(self._init_capacity * (1.0 - severity), 0.0, 1.0))
        self.throughput = float(np.clip(self._init_throughput * (1.0 - severity), 0.0, 1.0))

        # Degrade plant KPIs
        self.production_rate = float(
            np.clip(self._init_production_rate * (1.0 - severity), 0.0, 1.0)
        )
        self.quality_rate = float(
            np.clip(self._init_quality_rate * (1.0 - severity * 0.7), 0.0, 1.0)
        )
        machines_lost: int = int(np.floor(self._init_num_active_machines * severity))
        self.num_active_machines = max(self._init_num_active_machines - machines_lost, 0)

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance plant state by one simulation timestep.

        Increments time_disrupted counter while disrupted.
        """
        if self.is_disrupted:
            self.time_disrupted += 1

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Restore plant to pre-disruption baseline."""
        super().reset()
        self.production_rate = self._init_production_rate
        self.quality_rate = self._init_quality_rate
        self.num_active_machines = self._init_num_active_machines
