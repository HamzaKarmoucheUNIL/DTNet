"""colors.py — single source of truth for DTNet supply-chain layer colours.

Import ``LAYER_COLORS`` from here everywhere instead of hardcoding per-layer hex
values, so the five layers always render with five distinct, consistent colours
across every figure (topology, distributions, dashboard, comparison plots) and
their legends.

Red (``DISRUPTED_COLOR`` = #D62728) is RESERVED for the 'disrupted' state and
must never be assigned to a layer.
"""

from __future__ import annotations

from typing import Dict, List

# Canonical left-to-right supply-chain layer order.
LAYER_ORDER: List[str] = ["supplier", "logistics", "plant", "machine", "distribution"]

# The one and only per-layer colour map — five visually distinct hues.
LAYER_COLORS: Dict[str, str] = {
    "supplier":     "#1F77B4",  # blue
    "logistics":    "#FF7F0E",  # orange
    "plant":        "#2CA02C",  # green
    "machine":      "#9467BD",  # purple
    "distribution": "#17BECF",  # cyan
}

# Reserved status colour — never assign to a layer.
DISRUPTED_COLOR: str = "#D62728"  # red
