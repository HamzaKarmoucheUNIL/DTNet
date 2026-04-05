"""
entity_mapping.py — Extract entity relationships from the raw dataset.

Responsibility: build all lookup dictionaries that Phase 2 (graph
construction) needs to wire up nodes and edges.

Mappings produced
-----------------
- plant_to_machines  : plant_code  → sorted list of asset_tags
- machine_to_parts   : asset_tag   → sorted list of part_nos
- part_to_machines   : part_no     → sorted list of asset_tags
- family_to_parts    : part_family → sorted list of part_nos
- cross_plant_parts  : part_no     → sorted list of plant_codes
                       (only parts that appear in ≥ 2 plants)

All mappings are bundled in the ``EntityMappings`` dataclass.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

np.random.seed(42)

# ── column name constants ──────────────────────────────────────────────────
COL_PLANT:  str = "plant_code"
COL_ASSET:  str = "asset_tag"
COL_PART:   str = "part_no"
COL_FAMILY: str = "part_family"


# ── dataclass ──────────────────────────────────────────────────────────────

@dataclass
class EntityMappings:
    """All entity relationship mappings extracted from the dataset.

    Attributes
    ----------
    plant_codes : list[str]
        All unique plant codes, sorted.
    asset_tags : list[str]
        All unique machine asset tags, sorted.
    part_nos : list[str]
        All unique part numbers, sorted.
    part_families : list[str]
        All unique part families, sorted.
    plant_to_machines : dict[str, list[str]]
        Maps each plant_code to the sorted list of asset_tags located
        in that plant.
    machine_to_parts : dict[str, list[str]]
        Maps each asset_tag to the sorted list of part_nos used by
        that machine.
    part_to_machines : dict[str, list[str]]
        Maps each part_no to the sorted list of asset_tags that use it.
    family_to_parts : dict[str, list[str]]
        Maps each part_family to the sorted list of part_nos in that
        family.
    cross_plant_parts : dict[str, list[str]]
        Maps part_no → sorted list of plant_codes for parts that appear
        in two or more plants. Empty if no shared parts exist.
    """

    plant_codes:       List[str] = field(default_factory=list)
    asset_tags:        List[str] = field(default_factory=list)
    part_nos:          List[str] = field(default_factory=list)
    part_families:     List[str] = field(default_factory=list)

    plant_to_machines: Dict[str, List[str]] = field(default_factory=dict)
    machine_to_parts:  Dict[str, List[str]] = field(default_factory=dict)
    part_to_machines:  Dict[str, List[str]] = field(default_factory=dict)
    family_to_parts:   Dict[str, List[str]] = field(default_factory=dict)
    cross_plant_parts: Dict[str, List[str]] = field(default_factory=dict)


# ── helpers ────────────────────────────────────────────────────────────────

def _require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    """Raise ValueError if any required column is absent from df.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check.
    cols : list[str]
        Column names that must be present.

    Raises
    ------
    ValueError
        Lists every missing column in the error message.
    """
    missing: List[str] = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[entity_mapping] Required columns not found: {missing}. "
            f"Available: {list(df.columns)}"
        )


def _group_sorted(df: pd.DataFrame, by: str, collect: str) -> Dict[str, List[str]]:
    """Group df by ``by`` and collect unique sorted values of ``collect``.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame (must contain both columns).
    by : str
        Column to group on (becomes dict key).
    collect : str
        Column whose unique values are aggregated into the list.

    Returns
    -------
    dict[str, list[str]]
        ``{key: sorted_unique_values}``.
    """
    return (
        df.groupby(by)[collect]
        .apply(lambda s: sorted(s.dropna().unique().tolist()))
        .to_dict()
    )


# ── main public function ───────────────────────────────────────────────────

def build_entity_mappings(df: pd.DataFrame) -> "EntityMappings":
    """Extract all entity relationships from a raw (or preprocessed) DataFrame.

    The function works on both the raw string-valued DataFrame and the
    integer-encoded one from ``preprocess.py``, but the raw form is
    preferred so that keys remain human-readable strings.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing at minimum the columns:
        ``plant_code``, ``asset_tag``, ``part_no``, ``part_family``.

    Returns
    -------
    EntityMappings
        Fully populated dataclass with all five mapping dictionaries.
    """
    required: List[str] = [COL_PLANT, COL_ASSET, COL_PART, COL_FAMILY]
    _require_columns(df, required)

    # Work on a lean copy — only the four columns we need
    cols: pd.DataFrame = df[required].copy().astype(str)
    # Treat literal "nan" strings (from astype(str) on NaN) as missing
    cols.replace("nan", pd.NA, inplace=True)
    cols.dropna(inplace=True)

    # ── unique entity lists ────────────────────────────────────────────────
    plant_codes:   List[str] = sorted(cols[COL_PLANT].unique().tolist())
    asset_tags:    List[str] = sorted(cols[COL_ASSET].unique().tolist())
    part_nos:      List[str] = sorted(cols[COL_PART].unique().tolist())
    part_families: List[str] = sorted(cols[COL_FAMILY].unique().tolist())

    # ── directional mappings ───────────────────────────────────────────────
    plant_to_machines: Dict[str, List[str]] = _group_sorted(cols, COL_PLANT,  COL_ASSET)
    machine_to_parts:  Dict[str, List[str]] = _group_sorted(cols, COL_ASSET,  COL_PART)
    part_to_machines:  Dict[str, List[str]] = _group_sorted(cols, COL_PART,   COL_ASSET)
    family_to_parts:   Dict[str, List[str]] = _group_sorted(cols, COL_FAMILY, COL_PART)

    # ── cross-plant shared parts ───────────────────────────────────────────
    # For each part_no, collect the set of plants that stock it
    part_to_plants: Dict[str, List[str]] = _group_sorted(cols, COL_PART, COL_PLANT)
    cross_plant_parts: Dict[str, List[str]] = {
        part: plants
        for part, plants in part_to_plants.items()
        if len(plants) >= 2
    }

    return EntityMappings(
        plant_codes=plant_codes,
        asset_tags=asset_tags,
        part_nos=part_nos,
        part_families=part_families,
        plant_to_machines=plant_to_machines,
        machine_to_parts=machine_to_parts,
        part_to_machines=part_to_machines,
        family_to_parts=family_to_parts,
        cross_plant_parts=cross_plant_parts,
    )


def print_entity_summary(em: EntityMappings) -> None:
    """Print a structured summary of the entity mappings.

    Parameters
    ----------
    em : EntityMappings
        Populated entity mappings as returned by ``build_entity_mappings``.
    """
    sep: str = "=" * 55

    print(f"\n{sep}")
    print("  ENTITY STRUCTURE SUMMARY")
    print(sep)
    print(f"  {'Unique plant_codes':<35}: {len(em.plant_codes):>6,}")
    print(f"  {'Unique asset_tags (machines)':<35}: {len(em.asset_tags):>6,}")
    print(f"  {'Unique part_nos':<35}: {len(em.part_nos):>6,}")
    print(f"  {'Unique part_families':<35}: {len(em.part_families):>6,}")

    print(f"\n  {'Machines per plant':}")
    for plant, machines in em.plant_to_machines.items():
        print(f"    {plant:<20}: {len(machines):>5,} machines")

    machines_with_parts: int = sum(1 for v in em.machine_to_parts.values() if v)
    avg_parts: float = (
        sum(len(v) for v in em.machine_to_parts.values()) / len(em.machine_to_parts)
        if em.machine_to_parts else 0.0
    )
    print(f"\n  Machines with ≥1 part recorded : {machines_with_parts:,}")
    print(f"  Avg parts per machine          : {avg_parts:.1f}")

    print(f"\n  Part families ({len(em.part_families)}):")
    for family, parts in em.family_to_parts.items():
        print(f"    {family:<30}: {len(parts):>5,} parts")

    print(f"\n  Cross-plant shared parts       : {len(em.cross_plant_parts):,}")
    if em.cross_plant_parts:
        # Show up to 10 examples
        examples = list(em.cross_plant_parts.items())[:10]
        print(f"  (showing first {len(examples)})")
        for part, plants in examples:
            print(f"    {part:<20} shared by: {plants}")

    print(sep + "\n")


# ── CLI entry point ────────────────────────────────────────────────────────

def main() -> None:
    """Run entity mapping standalone from the CLI.

    Usage
    -----
    .. code-block:: bash

        python -m src.data.entity_mapping <filename.csv>

    Example
    -------
    .. code-block:: bash

        python -m src.data.entity_mapping predictive_maintenance.csv
    """
    if len(sys.argv) < 2:
        print("Usage: python -m src.data.entity_mapping <filename.csv>")
        sys.exit(1)

    filename: str = sys.argv[1]

    from src.data.loader import load_csv  # local import to avoid circular dep

    print(f"[entity_mapping] Loading '{filename}'...")
    df: pd.DataFrame = load_csv(filename)

    print("[entity_mapping] Building entity mappings...")
    em: EntityMappings = build_entity_mappings(df)

    print_entity_summary(em)


if __name__ == "__main__":
    main()
