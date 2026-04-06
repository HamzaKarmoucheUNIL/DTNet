"""MachineAgent — digital twin for a physical machine node.

Sensor attributes come from the Kaggle predictive-maintenance dataset.
No supply-chain-level attributes (lead times, reliability scores, etc.)
belong here — those live in SupplierAgent / LogisticsAgent.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List

from src.agents.base_agent import DigitalTwinAgent

np.random.seed(42)
torch.manual_seed(42)

# --- Sensor thresholds used in health scoring ----------------------------
TEMP_BEARING_MAX: float = 100.0   # °C — above this → severe degradation
TEMP_MOTOR_MAX: float = 120.0     # °C
VIBRATION_MAX: float = 10.0       # mm/s — combined h+v
OIL_PRESSURE_MIN: float = 20.0    # bar — below this → critical
OIL_PRESSURE_MAX: float = 80.0    # bar — nominal upper bound

# Health weight distribution across sensor groups
W_TEMP: float = 0.25
W_VIBRATION: float = 0.25
W_OIL: float = 0.20
W_LOAD: float = 0.15
W_BASE: float = 0.15   # capacity / throughput / failure_prob from base


@dataclass
class MachineAgent(DigitalTwinAgent):
    """Digital twin for a single physical machine on the production floor.

    Sensor readings mirror the Kaggle predictive-maintenance feature set.
    Health is computed from real-time sensor values; disruption degrades both
    the operational state (capacity, throughput) and the sensor readings.

    Attributes:
        temp_bearing: Bearing temperature in °C.
        temp_motor: Motor temperature in °C.
        vibration_h: Horizontal vibration in mm/s.
        vibration_v: Vertical vibration in mm/s.
        oil_pressure: Lubrication oil pressure in bar.
        load_pct: Current load as a fraction of rated capacity [0, 1].
        power_kw: Active power draw in kW.
        rpm: Spindle / motor speed in revolutions per minute.
        breakdown_flag: True when the machine has failed completely.
        parts_required: List of part IDs that this machine consumes.
    """

    # Sensor readings
    temp_bearing: float = field(default=60.0)
    temp_motor: float = field(default=70.0)
    vibration_h: float = field(default=1.0)
    vibration_v: float = field(default=1.0)
    oil_pressure: float = field(default=50.0)
    load_pct: float = field(default=0.8)
    power_kw: float = field(default=50.0)
    rpm: float = field(default=1500.0)
    breakdown_flag: bool = field(default=False)
    parts_required: List[str] = field(default_factory=list)

    # Private sensor baselines for reset()
    _init_temp_bearing: float = field(init=False, repr=False)
    _init_temp_motor: float = field(init=False, repr=False)
    _init_vibration_h: float = field(init=False, repr=False)
    _init_vibration_v: float = field(init=False, repr=False)
    _init_oil_pressure: float = field(init=False, repr=False)
    _init_load_pct: float = field(init=False, repr=False)
    _init_power_kw: float = field(init=False, repr=False)
    _init_rpm: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Cache sensor baselines after parent initialisation."""
        super().__post_init__()
        self._init_temp_bearing = self.temp_bearing
        self._init_temp_motor = self.temp_motor
        self._init_vibration_h = self.vibration_h
        self._init_vibration_v = self.vibration_v
        self._init_oil_pressure = self.oil_pressure
        self._init_load_pct = self.load_pct
        self._init_power_kw = self.power_kw
        self._init_rpm = self.rpm

    # ------------------------------------------------------------------
    # Health scoring
    # ------------------------------------------------------------------

    def compute_health_score(self) -> float:
        """Compute machine health from sensor readings and base state.

        Each sensor group is normalised to [0, 1] where 1.0 is healthy.
        High temperatures, high vibration, and abnormal oil pressure each
        reduce the score. A complete breakdown forces the score to 0.

        Returns:
            A float in [0, 1]. Returns 0.0 immediately if breakdown_flag
            is set.
        """
        if self.breakdown_flag:
            return 0.0

        # Temperature health — higher temp → lower score
        temp_b_score: float = float(
            np.clip(1.0 - self.temp_bearing / TEMP_BEARING_MAX, 0.0, 1.0)
        )
        temp_m_score: float = float(
            np.clip(1.0 - self.temp_motor / TEMP_MOTOR_MAX, 0.0, 1.0)
        )
        temp_score: float = (temp_b_score + temp_m_score) / 2.0

        # Vibration health — combined magnitude; higher → worse
        vib_combined: float = self.vibration_h + self.vibration_v
        vib_score: float = float(np.clip(1.0 - vib_combined / VIBRATION_MAX, 0.0, 1.0))

        # Oil pressure health — penalise both too-low and too-high
        oil_mid: float = (OIL_PRESSURE_MIN + OIL_PRESSURE_MAX) / 2.0
        oil_range: float = (OIL_PRESSURE_MAX - OIL_PRESSURE_MIN) / 2.0
        oil_score: float = float(
            np.clip(1.0 - abs(self.oil_pressure - oil_mid) / oil_range, 0.0, 1.0)
        )

        # Load health — very high load accelerates wear
        load_score: float = float(np.clip(1.0 - self.load_pct, 0.0, 1.0))

        # Base operational health (capacity / throughput / failure_prob)
        base_score: float = (
            0.4 * self.capacity
            + 0.4 * self.throughput
            + 0.2 * (1.0 - self.failure_prob)
        )

        health: float = (
            W_TEMP * temp_score
            + W_VIBRATION * vib_score
            + W_OIL * oil_score
            + W_LOAD * load_score
            + W_BASE * base_score
        )
        return float(np.clip(health, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Disruption
    # ------------------------------------------------------------------

    def apply_disruption(self, severity: float, timestep: int) -> None:
        """Degrade sensor readings and operational state by disruption severity.

        Temperatures and vibration increase; oil pressure drops toward the
        unsafe lower bound; capacity and throughput decrease. A severity of
        1.0 triggers a full breakdown.

        Args:
            severity: Disruption magnitude in [0, 1].
            timestep: Current simulation timestep.
        """
        severity = float(np.clip(severity, 0.0, 1.0))

        if self.is_disrupted and severity <= self.disruption_severity:
            return

        # Update base disruption state
        self.is_disrupted = True
        self.disruption_severity = severity
        self.time_disrupted = timestep

        # Degrade operational state
        self.capacity = float(np.clip(self._init_capacity * (1.0 - severity), 0.0, 1.0))
        self.throughput = float(np.clip(self._init_throughput * (1.0 - severity), 0.0, 1.0))

        # Degrade sensor readings
        self.temp_bearing = self._init_temp_bearing * (1.0 + severity * 0.8)
        self.temp_motor = self._init_temp_motor * (1.0 + severity * 0.7)
        self.vibration_h = self._init_vibration_h * (1.0 + severity * 3.0)
        self.vibration_v = self._init_vibration_v * (1.0 + severity * 3.0)
        self.oil_pressure = float(
            np.clip(self._init_oil_pressure * (1.0 - severity * 0.6), OIL_PRESSURE_MIN * 0.5, None)
        )
        self.load_pct = float(np.clip(self._init_load_pct * (1.0 + severity * 0.3), 0.0, 1.0))

        if severity >= 1.0:
            self.breakdown_flag = True

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance machine state by one simulation timestep.

        Increments time_disrupted counter when active. Subclasses or the
        simulation engine may add spontaneous failure checks here.
        """
        if self.is_disrupted:
            self.time_disrupted += 1

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Restore machine to pre-disruption baseline including sensor values."""
        super().reset()
        self.temp_bearing = self._init_temp_bearing
        self.temp_motor = self._init_temp_motor
        self.vibration_h = self._init_vibration_h
        self.vibration_v = self._init_vibration_v
        self.oil_pressure = self._init_oil_pressure
        self.load_pct = self._init_load_pct
        self.power_kw = self._init_power_kw
        self.rpm = self._init_rpm
        self.breakdown_flag = False
