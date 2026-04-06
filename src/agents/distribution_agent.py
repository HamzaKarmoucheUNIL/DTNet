"""DistributionAgent — digital twin for a distribution / retail node.

Tracks demand-side metrics: fulfillment rate, stock level, delivery delays,
and demand variability. No sensor readings, no production KPIs, no transit
attributes — those belong in other node types.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field

from src.agents.base_agent import DigitalTwinAgent

np.random.seed(42)
torch.manual_seed(42)

# --- Normalisation constants for health scoring --------------------------
DELIVERY_DELAY_MAX: float = 14.0    # days — at or above this → contribution = 0
DEMAND_VARIABILITY_MAX: float = 1.0  # already [0, 1]; 1.0 = maximally volatile

# Health weight distribution
W_FULFILLMENT: float = 0.40
W_STOCK: float = 0.35
W_DELAY: float = 0.25


@dataclass
class DistributionAgent(DigitalTwinAgent):
    """Digital twin for a distribution centre or retail endpoint.

    Monitors customer-facing performance: whether orders are fulfilled on time,
    how much stock is on hand, and how long deliveries are delayed. Disruption
    reduces fulfillment rates, depletes stock, and inflates delivery delays.

    Attributes:
        demand_variability: Coefficient of variation of demand [0, 1]. Higher
            values indicate more volatile / unpredictable demand.
        fulfillment_rate: Fraction of customer orders fulfilled on time [0, 1].
        stock_level: Normalised inventory level as a fraction of target [0, 1].
        delivery_delay_days: Current average delivery delay in days (≥ 0).
    """

    demand_variability: float = field(default=0.2)
    fulfillment_rate: float = field(default=0.95)
    stock_level: float = field(default=0.8)
    delivery_delay_days: float = field(default=1.0)

    # Private baselines for reset()
    _init_demand_variability: float = field(init=False, repr=False)
    _init_fulfillment_rate: float = field(init=False, repr=False)
    _init_stock_level: float = field(init=False, repr=False)
    _init_delivery_delay_days: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Cache distribution attribute baselines."""
        super().__post_init__()
        self._init_demand_variability = self.demand_variability
        self._init_fulfillment_rate = self.fulfillment_rate
        self._init_stock_level = self.stock_level
        self._init_delivery_delay_days = self.delivery_delay_days

    # ------------------------------------------------------------------
    # Health scoring
    # ------------------------------------------------------------------

    def compute_health_score(self) -> float:
        """Compute distribution node health from fulfillment rate, stock level, and delivery delay.

        Demand variability is not scored directly (it is an exogenous input),
        but it is used in apply_disruption to scale the impact of disruptions.

        Returns:
            A float in [0, 1]. 1.0 = perfect fulfillment, full stock, no delay.
        """
        fulfillment_score: float = float(np.clip(self.fulfillment_rate, 0.0, 1.0))
        stock_score: float = float(np.clip(self.stock_level, 0.0, 1.0))
        delay_score: float = float(
            np.clip(1.0 - self.delivery_delay_days / DELIVERY_DELAY_MAX, 0.0, 1.0)
        )

        health: float = (
            W_FULFILLMENT * fulfillment_score
            + W_STOCK * stock_score
            + W_DELAY * delay_score
        )
        return float(np.clip(health, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Disruption
    # ------------------------------------------------------------------

    def apply_disruption(self, severity: float, timestep: int) -> None:
        """Degrade distribution performance by disruption severity.

        Fulfillment rate and stock level fall; delivery delays rise. Demand
        variability increases as uncertainty in the supply signal grows. High
        demand variability amplifies the effective impact of disruptions.

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

        # High demand variability amplifies disruption impact on stock
        variability_amplifier: float = 1.0 + self.demand_variability * 0.5

        # Degrade operational state
        self.capacity = float(np.clip(self._init_capacity * (1.0 - severity), 0.0, 1.0))
        self.throughput = float(np.clip(self._init_throughput * (1.0 - severity), 0.0, 1.0))

        # Degrade distribution attributes
        self.fulfillment_rate = float(
            np.clip(self._init_fulfillment_rate * (1.0 - severity * 0.85), 0.0, 1.0)
        )
        self.stock_level = float(
            np.clip(
                self._init_stock_level * (1.0 - severity * variability_amplifier),
                0.0,
                1.0,
            )
        )
        self.delivery_delay_days = (
            self._init_delivery_delay_days + severity * DELIVERY_DELAY_MAX * 0.7
        )
        self.demand_variability = float(
            np.clip(self._init_demand_variability + severity * 0.4, 0.0, DEMAND_VARIABILITY_MAX)
        )

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance distribution state by one simulation timestep.

        Increments time_disrupted counter while disrupted.
        """
        if self.is_disrupted:
            self.time_disrupted += 1

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Restore distribution node to pre-disruption baseline."""
        super().reset()
        self.demand_variability = self._init_demand_variability
        self.fulfillment_rate = self._init_fulfillment_rate
        self.stock_level = self._init_stock_level
        self.delivery_delay_days = self._init_delivery_delay_days
