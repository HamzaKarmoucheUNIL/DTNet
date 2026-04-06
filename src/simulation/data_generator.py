"""data_generator.py — Generate GNN training data via mass simulation runs.

Creates N diverse cascading-failure simulation runs with varied disruption
nodes, counts (1–3), and severities (0.3–1.0).  Each run records the initial
node-feature snapshot and final disruption-severity target per node, producing
the supervised training dataset for the GNN (COMMON_MISTAKES #8).

Public API: ``generate_simulation_data(G, n_runs, ...)`` → list + saved .pkl
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
import torch

from src.agents.base_agent import DigitalTwinAgent
from src.simulation.model import DTNetModel

np.random.seed(42)
torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Constants — configurable defaults (COMMON_MISTAKES #5)
# ---------------------------------------------------------------------------

N_RUNS_DEFAULT: int = 10_000
BASE_SEED: int = 0
MAX_TIMESTEPS_DEFAULT: int = 10
MIN_SEVERITY: float = 0.3
MAX_SEVERITY: float = 1.0
MIN_DISRUPTION_SEEDS: int = 1
MAX_DISRUPTION_SEEDS: int = 3
LOG_EVERY: int = 1_000

# Fixed canonical layer order for one-hot encoding
LAYER_ORDER: List[str] = ["supplier", "logistics", "plant", "machine", "distribution"]

OUTPUT_PATH_DEFAULT: Path = Path("data/processed/simulation_runs.pkl")

# 5 scalars + 5 one-hot layer flags = 10 features per node
N_NODE_FEATURES: int = 10


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _node_feature_vector(twin: DigitalTwinAgent, layer: str) -> List[float]:
    """Build a fixed-length (10-value) feature vector for one node.

    Layout: [capacity, throughput, failure_prob, health_score,
    disruption_severity, is_supplier, is_logistics, is_plant,
    is_machine, is_distribution].

    Args:
        twin: DigitalTwinAgent instance for this node.
        layer: Layer label string (e.g. 'supplier').

    Returns:
        List of 10 floats.
    """
    one_hot: List[float] = [
        1.0 if layer == lbl else 0.0 for lbl in LAYER_ORDER
    ]
    return [
        twin.capacity,
        twin.throughput,
        twin.failure_prob,
        twin.compute_health_score(),
        twin.disruption_severity,
        *one_hot,
    ]


def _build_edge_index(G: nx.DiGraph, node_order: List[str]) -> List[List[int]]:
    """Build the edge index as [src_idx, dst_idx] integer pairs.

    Computed once; shared across all runs (topology never changes).

    Args:
        G: DTNet DiGraph.
        node_order: Canonical node ID ordering.

    Returns:
        List of [src_idx, dst_idx] pairs, one per directed edge.
    """
    idx_map: Dict[str, int] = {nid: i for i, nid in enumerate(node_order)}
    return [
        [idx_map[u], idx_map[v]]
        for u, v in G.edges()
    ]


def _extract_initial_features(
    G: nx.DiGraph,
    node_order: List[str],
) -> List[List[float]]:
    """Extract per-node feature vectors from the current (baseline) twin state.

    Call AFTER ``model.reset()`` and BEFORE ``inject_disruption()`` so that
    disruption_severity is 0.0 for every node.

    Args:
        G: DTNet DiGraph with "twin" and "layer" on every node.
        node_order: Canonical node ID ordering.

    Returns:
        List of N_nodes feature vectors, each of length ``N_NODE_FEATURES``.
    """
    features: List[List[float]] = []
    for nid in node_order:
        data = G.nodes[nid]
        twin: DigitalTwinAgent = data["twin"]
        layer: str = data.get("layer", "machine")
        features.append(_node_feature_vector(twin, layer))
    return features


def _extract_final_severities(
    G: nx.DiGraph,
    node_order: List[str],
) -> List[float]:
    """Read disruption_severity from every twin after the simulation ends.

    Reads directly from twin objects, bypassing ``model.get_history()``'s
    DataFrame overhead — critical for throughput at 10,000 runs.

    Args:
        G: DTNet DiGraph.
        node_order: Canonical node ID ordering.

    Returns:
        List of floats, one per node.
    """
    return [
        float(G.nodes[nid]["twin"].disruption_severity)
        for nid in node_order
    ]


def _sample_disruption(
    rng: np.random.Generator,
    node_order: List[str],
    G: nx.DiGraph,
) -> Dict[str, float]:
    """Sample 1–3 random nodes with Uniform[MIN_SEVERITY, MAX_SEVERITY] severities.

    Args:
        rng: Per-run RNG from ``base_seed + run_index``.
        node_order: Canonical node ID list (sampling pool).
        G: DTNet DiGraph (reserved for future layer-aware sampling).

    Returns:
        Dict mapping selected node IDs → sampled severity.
    """
    n_seeds: int = int(
        rng.integers(MIN_DISRUPTION_SEEDS, MAX_DISRUPTION_SEEDS + 1)
    )
    n_seeds = min(n_seeds, len(node_order))

    selected: List[str] = list(
        rng.choice(node_order, size=n_seeds, replace=False)
    )
    severities: np.ndarray = rng.uniform(MIN_SEVERITY, MAX_SEVERITY, size=n_seeds)

    return {
        nid: float(np.clip(sev, 0.0, 1.0))
        for nid, sev in zip(selected, severities)
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_simulation_data(
    G: nx.DiGraph,
    n_runs: int = N_RUNS_DEFAULT,
    base_seed: int = BASE_SEED,
    output_path: Optional[Path] = None,
    max_timesteps: int = MAX_TIMESTEPS_DEFAULT,
) -> List[Dict[str, Any]]:
    """Generate N diverse simulation runs and save the dataset to a pickle file.

    Builds one ``DTNetModel`` and reuses it across all runs via ``reset()``,
    avoiding the cost of re-creating Mesa agents per run.  Each run uses seed
    ``base_seed + run_index`` for reproducibility.

    Each returned dict has keys:
      ``initial_features``   List[List[float]] (N_nodes, N_NODE_FEATURES) — baseline.
      ``final_severities``   List[float] (N_nodes,) — severity after max_timesteps.
      ``initial_disruption`` Dict[str, float] — injected node_id → severity.
      ``edge_index``         List[[src_idx, dst_idx]] — shared across all runs.
      ``node_order``         List[str] — index → node_id mapping.

    Args:
        G: DTNet DiGraph (must be built via ``build_graph()`` first).
        n_runs: Number of simulation runs. Default 10,000.
        base_seed: Base seed; run i uses ``base_seed + i``. Default 0.
        output_path: Pickle output path. Defaults to
            ``data/processed/simulation_runs.pkl``.
        max_timesteps: Steps per run. Default 10.

    Returns:
        List of run dicts (also persisted to ``output_path``).
    """
    np.random.seed(42)

    if output_path is None:
        output_path = OUTPUT_PATH_DEFAULT
    output_path = Path(output_path)

    # ── pre-compute topology (same for every run) ──────────────────────────
    node_order: List[str] = list(G.nodes())
    edge_index: List[List[int]] = _build_edge_index(G, node_order)
    n_nodes: int = len(node_order)
    n_edges: int = G.number_of_edges()

    print(f"[data_generator] Starting {n_runs:,} runs — "
          f"nodes={n_nodes} edges={n_edges} features={N_NODE_FEATURES} "
          f"steps={max_timesteps}")

    # Build model once; reuse via reset() — avoids re-creating N agents per run
    for _, data in G.nodes(data=True):
        data["twin"].reset()
    model: DTNetModel = DTNetModel(G)

    dataset: List[Dict[str, Any]] = []
    t_start: float = time.perf_counter()

    for run_idx in range(n_runs):
        rng: np.random.Generator = np.random.default_rng(base_seed + run_idx)

        # Reset all twins + scheduler + DataCollector (COMMON_MISTAKES #10)
        model.reset()

        # Baseline snapshot — must precede any inject_disruption() call
        initial_features: List[List[float]] = _extract_initial_features(G, node_order)

        # Sample and inject disruption
        initial_disruption: Dict[str, float] = _sample_disruption(rng, node_order, G)
        for node_id, severity in initial_disruption.items():
            model.inject_disruption(node_id, severity)

        # Run cascade
        for _ in range(max_timesteps):
            model.step()

        # Final severities — read directly from twins (avoids DataFrame overhead)
        final_severities: List[float] = _extract_final_severities(G, node_order)

        dataset.append({
            "initial_features":   initial_features,
            "final_severities":   final_severities,
            "initial_disruption": initial_disruption,
            "edge_index":         edge_index,   # shared object → deduped by pickle
            "node_order":         node_order,   # shared object → deduped by pickle
        })

        # ── progress report ────────────────────────────────────────────────
        completed: int = run_idx + 1
        if completed % LOG_EVERY == 0:
            elapsed: float = time.perf_counter() - t_start
            rate: float = completed / max(elapsed, 1e-9)
            eta: float = (n_runs - completed) / max(rate, 1e-9)
            n_width: int = len(str(n_runs))
            print(
                f"[{completed:{n_width}d}/{n_runs}]  "
                f"elapsed={elapsed:6.0f}s  "
                f"rate={rate:6.1f} runs/s  "
                f"eta={eta:6.0f}s"
            )

    # ── save dataset ───────────────────────────────────────────────────────
    total_elapsed: float = time.perf_counter() - t_start
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as fh:
        pickle.dump(dataset, fh, protocol=pickle.HIGHEST_PROTOCOL)

    file_mb: float = output_path.stat().st_size / (1024 ** 2)
    ms_per_run: float = total_elapsed / max(n_runs, 1) * 1000.0

    print(f"\n[data_generator] Done — {n_runs:,} runs in {total_elapsed:.1f}s "
          f"({ms_per_run:.2f} ms/run)")
    print(f"[data_generator] Saved {file_mb:.1f} MB -> {output_path}")
    print(f"[data_generator] Shape: {n_runs} runs x {n_nodes} nodes x {N_NODE_FEATURES} features")

    return dataset
