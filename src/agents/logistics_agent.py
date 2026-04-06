"""LogisticsAgent — digital twin for a logistics / warehouse node.

Tracks transit times, warehouse capacity, route reliability, and backlog.
No sensor readings and no production KPIs — those belong in other node types.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field

from src.agents.base_agent import DigitalTwinAgent

np.random.seed(42)
torch.manual_seed(42)

# --- Normalisation constants for health scoring --------------------------
TRANSIT_TIME_NOMINAL: float = 3.0    # days — baseline "good" transit time
TRANSIT_TIME_MAX: float = 14.0       # days — at or above this → contribution = 0
BACKLOG_MAX: float = 500.0           # units — at or above this → contribution = 0

# Health weight distribution
W_TRANSIT: float = 0.30
W_WAREHOUSE: float = 0.25
W_RELIABILITY: float = 0.30
W_BACKLOG: float = 0.15


@dataclass
class LogisticsAgent(DigitalTwinAgent):
    """Digital twin for a logistics hub or warehouse node.

    Monitors transit performance and storage capacity. Disruption increases
    transit times and backlog while reducing route reliability and warehouse
    throughput.

    Attributes:
        transit_time_days: Current average transit time in days (≥ 0).
        warehouse_capacity: Fraction of warehouse space available [0, 1].
        route_reliability: Fraction of shipments that arrive undamaged and
            on route [0, 1].
        backlog: Number of outstanding unshipped orders (≥ 0).
    """

    transit_time_days: float = field(default=3.0)
    warehouse_capacity: float = field(default=0.8)
    route_reliability: float = field(default=0.95)
    backlog: float = field(default=0.0)

    # Private baselines for reset()
    _init_transit_time_days: float = field(init=False, repr=False)
    _init_warehouse_capacity: float = field(init=False, repr=False)
    _init_route_reliability: float = field(init=False, repr=False)
    _init_backlog: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Cache logistics attribute baselines."""
        super().__post_init__()
        self._init_transit_time_days = self.transit_time_days
        self._init_warehouse_capacity = self.warehouse_capacity
        self._init_route_reliability = self.route_reliability
        self._init_backlog = self.backlog

    # ------------------------------------------------------------------
    # Health scoring
    # ------------------------------------------------------------------

    def compute_health_score(self) -> float:
        """Compute logistics node health from transit time, capacity, reliability, and backlog.

        Long transit times, full warehouses, unreliable routes, and large
        backlogs all reduce health.

        Returns:
            A float in [0, 1]. 1.0 = fast transit, available capacity, perfect
            reliability, no backlog.
        """
        # Transit time score — penalise beyond nominal
        transit_excess: float = max(self.transit_time_days - TRANSIT_TIME_NOMINAL, 0.0)
        transit_range: float = TRANSIT_TIME_MAX - TRANSIT_TIME_NOMINAL
        transit_score: float = float(np.clip(1.0 - transit_excess / transit_range, 0.0, 1.0))

        # Warehouse capacity score — available space is healthy
        warehouse_score: float = float(np.clip(self.warehouse_capacity, 0.0, 1.0))

        # Route reliability score — direct mapping
        reliability_score: float = float(np.clip(self.route_reliability, 0.0, 1.0))

        # Backlog score — penalise growing backlogs
        backlog_score: float = float(np.clip(1.0 - self.backlog / BACKLOG_MAX, 0.0, 1.0))

        health: float = (
            W_TRANSIT * transit_score
            + W_WAREHOUSE * warehouse_score
            + W_RELIABILITY * reliability_score
            + W_BACKLOG * backlog_score
        )
        return float(np.clip(health, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Disruption
    # ------------------------------------------------------------------

    def apply_disruption(self, severity: float, timestep: int) -> None:
        """Degrade logistics performance by disruption severity.

        Transit times increase, warehouse capacity decreases (blocked stock),
        route reliability drops, and backlog accumulates. Capacity and
        throughput are also degraded.

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

        # Degrade logistics attributes
        self.transit_time_days = self._init_transit_time_days * (1.0 + severity * 3.0)
        self.warehouse_capacity = float(
            np.clip(self._init_warehouse_capacity * (1.0 - severity * 0.8), 0.0, 1.0)
        )
        self.route_reliability = float(
            np.clip(self._init_route_reliability * (1.0 - severity * 0.9), 0.0, 1.0)
        )
        self.backlog = self._init_backlog + severity * BACKLOG_MAX * 0.6

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance logistics state by one simulation timestep.

        Increments time_disrupted counter while disrupted.
        """
        if self.is_disrupted:
            self.time_disrupted += 1

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Restore logistics node to pre-disruption baseline."""
        super().reset()
        self.transit_time_days = self._init_transit_time_days
        self.warehouse_capacity = self._init_warehouse_capacity
        self.route_reliability = self._init_route_reliability
        self.backlog = self._init_backlog
