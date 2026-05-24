"""generate_data.py — Standalone entry-point to produce simulation_runs.pkl.

Builds the DTNet graph from data/raw/updated_data.csv, then calls
``generate_simulation_data()`` to run 5,000 diverse cascading-failure
simulations and saves the result to data/processed/simulation_runs.pkl,
which is the path expected by src/gnn/dataset.py (PKL_PATH).

Usage
-----
    python -m src.simulation.generate_data              # 5 000 runs (default)
    python -m src.simulation.generate_data --n-runs 10000

Seeds are fixed (np 42, random 42, torch 42) for reproducibility
(COMMON_MISTAKES #3).  Each run uses a deterministic per-run RNG derived
from BASE_SEED (COMMON_MISTAKES #8 / #10).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_RUNS_DEFAULT: int = 5_000
MAX_TIMESTEPS: int = 10
BASE_SEED: int = 0
CSV_FILENAME: str = "updated_data.csv"
PKL_OUTPUT: Path = Path("data/processed/simulation_runs.pkl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with ``n_runs`` attribute.
    """
    parser = argparse.ArgumentParser(
        description="Generate simulation_runs.pkl for DTNet GNN training."
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=N_RUNS_DEFAULT,
        help=f"Number of simulation runs to generate (default: {N_RUNS_DEFAULT:,}).",
    )
    return parser.parse_args()


def _build_graph():
    """Load data and build the DTNet DiGraph.

    Uses the same import chain as notebooks/03_simulation_runs.ipynb so the
    graph is identical to the one used during Phase 3.

    Returns:
        nx.DiGraph: Fully constructed DTNet graph with twin + layer attributes.
    """
    from src.data.loader import load_csv
    from src.data.preprocess import preprocess
    from src.data.entity_mapping import build_entity_mappings
    from src.graph.topology import infer_topology
    from src.graph.builder import build_graph

    print(f"[generate_data] Loading {CSV_FILENAME} ...")
    df_raw = load_csv(CSV_FILENAME)

    print("[generate_data] Preprocessing ...")
    df_clean, _ = preprocess(df_raw)

    print("[generate_data] Building entity mappings ...")
    em = build_entity_mappings(df_raw)

    print("[generate_data] Inferring topology ...")
    nodes, edges = infer_topology(em)

    print("[generate_data] Building graph ...")
    G = build_graph(nodes, edges, df_clean)
    print(
        f"[generate_data] Graph ready — "
        f"nodes={G.number_of_nodes()}  edges={G.number_of_edges()}"
    )
    return G


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Build the graph, generate simulation runs, and save the PKL.

    Saves to PKL_OUTPUT (data/processed/simulation_runs.pkl), which is the
    path used by src/gnn/dataset.py build_dataloaders().
    """
    args = _parse_args()
    n_runs: int = args.n_runs

    if n_runs < 5_000:
        print(
            f"[generate_data] WARNING: n_runs={n_runs} is below the 5,000 minimum "
            "recommended by COMMON_MISTAKES #8. GNN training may underfit."
        )

    G = _build_graph()

    from src.simulation.data_generator import generate_simulation_data

    print(f"[generate_data] Generating {n_runs:,} runs -> {PKL_OUTPUT}")
    runs: List[Dict[str, Any]] = generate_simulation_data(
        G,
        n_runs=n_runs,
        base_seed=BASE_SEED,
        output_path=PKL_OUTPUT,
        max_timesteps=MAX_TIMESTEPS,
    )

    # Quick sanity-check on the saved file
    n_nodes: int = len(runs[0]["node_order"])
    n_feats: int = len(runs[0]["initial_features"][0])
    n_edges: int = len(runs[0]["edge_index"])
    print(
        f"\n[generate_data] Sanity check:"
        f"\n  runs        : {len(runs):,}"
        f"\n  nodes/run   : {n_nodes}"
        f"\n  features    : {n_feats}  (expected 10)"
        f"\n  edges/run   : {n_edges}"
        f"\n  PKL path    : {PKL_OUTPUT.resolve()}"
        f"\n  PKL size    : {PKL_OUTPUT.stat().st_size / 1024 ** 2:.1f} MB"
    )
    print("[generate_data] Done. Run src/gnn/train.py next.")


if __name__ == "__main__":
    main()
