"""SupplierAgent — digital twin for a raw-material or component supplier node.

Supply-chain attributes only (delivery reliability, lead time, defect rate).
No sensor readings — those belong exclusively to MachineAgent.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List

from src.agents.base_agent import DigitalTwinAgent

np.random.seed(42)
torch.manual_seed(42)

# --- Normalisation constants for health scoring --------------------------
LEAD_TIME_NOMINAL: float = 7.0    # days — baseline "good" lead time
LEAD_TIME_MAX: float = 30.0       # days — at or above this → health contribution = 0
DEFECT_RATE_MAX: float = 0.10     # 10 % defect rate → health contribution = 0

# Health weight distribution
W_RELIABILITY: float = 0.40
W_LEAD_TIME: float = 0.35
W_DEFECT: float = 0.25


@dataclass
class SupplierAgent(DigitalTwinAgent):
    """Digital twin for a supplier node in the supply-chain network.

    Tracks delivery performance, lead times, and product quality. Disruption
    increases lead times and defect rates while reducing delivery reliability.

    Attributes:
        delivery_reliability: Fraction of orders delivered on time [0, 1].
        lead_time_days: Average order-to-delivery lead time in days (≥ 0).
        defect_rate: Fraction of delivered units that are defective [0, 1].
        parts_supplied: List of part IDs that this supplier provides.
        cost_per_unit: Unit cost in arbitrary currency units (≥ 0).
    """

    delivery_reliability: float = field(default=0.95)
    lead_time_days: float = field(default=7.0)
    defect_rate: float = field(default=0.02)
    parts_supplied: List[str] = field(default_factory=list)
    cost_per_unit: float = field(default=10.0)

    # Private baselines for reset()
    _init_delivery_reliability: float = field(init=False, repr=False)
    _init_lead_time_days: float = field(init=False, repr=False)
    _init_defect_rate: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Cache supply-chain attribute baselines."""
        super().__post_init__()
        self._init_delivery_reliability = self.delivery_reliability
        self._init_lead_time_days = self.lead_time_days
        self._init_defect_rate = self.defect_rate

    # ------------------------------------------------------------------
    # Health scoring
    # ------------------------------------------------------------------

    def compute_health_score(self) -> float:
        """Compute supplier health from delivery reliability, lead time, and defect rate.

        Reliability directly maps to health; lead time is penalised when it
        exceeds the nominal baseline; high defect rates strongly reduce health.

        Returns:
            A float in [0, 1]. 1.0 = fully reliable, on-time, zero defects.
        """
        # Reliability score — direct mapping
        reliability_score: float = float(np.clip(self.delivery_reliability, 0.0, 1.0))

        # Lead time score — penalise proportionally beyond nominal
        lead_excess: float = max(self.lead_time_days - LEAD_TIME_NOMINAL, 0.0)
        lead_range: float = LEAD_TIME_MAX - LEAD_TIME_NOMINAL
        lead_score: float = float(np.clip(1.0 - lead_excess / lead_range, 0.0, 1.0))

        # Defect rate score — higher defects → lower score
        defect_score: float = float(np.clip(1.0 - self.defect_rate / DEFECT_RATE_MAX, 0.0, 1.0))

        health: float = (
            W_RELIABILITY * reliability_score
            + W_LEAD_TIME * lead_score
            + W_DEFECT * defect_score
        )
        return float(np.clip(health, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Disruption
    # ------------------------------------------------------------------

    def apply_disruption(self, severity: float, timestep: int) -> None:
        """Degrade supplier performance attributes by disruption severity.

        Delivery reliability drops, lead times increase, and defect rate rises.
        Capacity and throughput are degraded via the base formula.

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

        # Degrade supply-chain attributes
        self.delivery_reliability = float(
            np.clip(self._init_delivery_reliability * (1.0 - severity * 0.9), 0.0, 1.0)
        )
        self.lead_time_days = self._init_lead_time_days * (1.0 + severity * 3.0)
        self.defect_rate = float(
            np.clip(self._init_defect_rate + severity * (DEFECT_RATE_MAX - self._init_defect_rate), 0.0, 1.0)
        )

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance supplier state by one simulation timestep.

        Increments time_disrupted counter while disrupted.
        """
        if self.is_disrupted:
            self.time_disrupted += 1

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Restore supplier to pre-disruption baseline."""
        super().reset()
        self.delivery_reliability = self._init_delivery_reliability
        self.lead_time_days = self._init_lead_time_days
        self.defect_rate = self._init_defect_rate
