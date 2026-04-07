"""dataset.py — Load simulation_runs.pkl and build PyG DataLoader objects.

Reads the pickled dataset produced by ``data_generator.py``, converts each
run to a ``torch_geometric.data.Data`` object, applies column-wise
StandardScaler normalization (fit on train set only to prevent leakage),
splits into train/val/test (70/15/15), and returns three DataLoaders.

Public API: ``build_dataloaders(pkl_path, G, batch_size, ...)``
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

try:
    from torch_geometric.loader import DataLoader
except ImportError:  # older PyG versions
    from torch_geometric.data import DataLoader  # type: ignore[no-redef]

np.random.seed(42)
torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PKL_PATH: Path = Path("data/processed/simulation_runs.pkl")
BATCH_SIZE: int = 32
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
# TEST_RATIO inferred as 1 - TRAIN_RATIO - VAL_RATIO = 0.15
SPLIT_SEED: int = 42

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_edge_index_tensor(edge_pairs: List[List[int]]) -> torch.Tensor:
    """Convert list of [src, dst] pairs to a (2, E) long tensor.

    The run dicts store edge_index as List[[src_idx, dst_idx]].
    PyG expects shape (2, E) — rows are src / dst index vectors.

    Args:
        edge_pairs: List of [src_idx, dst_idx] pairs (E entries).

    Returns:
        Tensor of shape (2, E) with dtype torch.long.
    """
    if not edge_pairs:
        return torch.zeros((2, 0), dtype=torch.long)
    arr: np.ndarray = np.array(edge_pairs, dtype=np.int64)  # (E, 2)
    return torch.from_numpy(arr.T).contiguous()              # (2, E)


def _extract_edge_attrs(
    G: nx.DiGraph,
    node_order: List[str],
) -> torch.Tensor:
    """Extract (criticality_weight, flow_capacity) per directed edge.

    Edges are iterated in ``G.edges()`` order, which matches the order used
    in ``data_generator._build_edge_index``, so indices align with edge_index.

    Args:
        G: DTNet DiGraph with ``criticality_weight`` and ``flow_capacity``
            on every edge.
        node_order: Canonical node ordering (used only for consistency check).

    Returns:
        Tensor of shape (E, 2): columns are [criticality_weight, flow_capacity].
    """
    rows: List[List[float]] = []
    for _u, _v, data in G.edges(data=True):
        cw: float = float(data.get("criticality_weight", 0.5))
        fc: float = float(data.get("flow_capacity", 0.9))
        rows.append([cw, fc])
    return torch.tensor(rows, dtype=torch.float)


def _fit_scaler(runs: List[Dict], train_indices: List[int]) -> StandardScaler:
    """Fit a StandardScaler on the training-set node features.

    Stacks all node feature matrices from training runs into one matrix and
    fits the scaler. Columns with zero variance (e.g., disruption_severity
    is always 0.0 in the initial snapshot) have their scale set to 1.0 to
    prevent division-by-zero during transform.

    Args:
        runs: Full list of simulation run dicts.
        train_indices: Indices of the runs assigned to the training split.

    Returns:
        Fitted ``StandardScaler`` ready for ``transform()``.
    """
    matrices: List[np.ndarray] = [
        np.array(runs[i]["initial_features"], dtype=np.float32)
        for i in train_indices
    ]
    all_features: np.ndarray = np.vstack(matrices)   # (N_train_nodes_total, F)

    scaler: StandardScaler = StandardScaler()
    scaler.fit(all_features)

    # Guard against zero-variance columns (would produce NaN after division)
    scaler.scale_ = np.where(scaler.scale_ == 0.0, 1.0, scaler.scale_)
    return scaler


def _run_to_data(
    run: Dict,
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor],
    scaler: StandardScaler,
) -> Data:
    """Convert a single simulation run dict to a PyG Data object.

    Node features (x) are normalized with the pre-fitted StandardScaler.
    Targets (y) are final disruption severities in [0, 1] — no normalization
    applied so they remain interpretable as probabilities.

    Args:
        run: Run dict with keys ``initial_features``, ``final_severities``,
            ``edge_index``, ``node_order``.
        edge_index: Precomputed (2, E) edge index tensor (shared across runs).
        edge_attr: Optional (E, 2) edge attribute tensor (shared across runs).
        scaler: Fitted StandardScaler for node features.

    Returns:
        ``torch_geometric.data.Data`` with x, edge_index, edge_attr, y.
    """
    # Node features — normalize with training scaler (COMMON_MISTAKES #7)
    x_raw: np.ndarray = np.array(run["initial_features"], dtype=np.float32)
    x_norm: np.ndarray = scaler.transform(x_raw).astype(np.float32)
    x: torch.Tensor = torch.from_numpy(x_norm)                 # (N, F)

    # Targets — final disruption severity per node
    y_np: np.ndarray = np.array(run["final_severities"], dtype=np.float32)
    y: torch.Tensor = torch.from_numpy(y_np)                   # (N,)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def _print_statistics(
    runs: List[Dict],
    train_data: List[Data],
    val_data: List[Data],
    test_data: List[Data],
    edge_attr: Optional[torch.Tensor],
    batch_size: int,
) -> None:
    """Print dataset statistics to stdout.

    Reports graph counts, feature dimensions, and target class balance.

    Args:
        runs: Full list of simulation run dicts.
        train_data: Training Data objects.
        val_data: Validation Data objects.
        test_data: Test Data objects.
        edge_attr: Edge attribute tensor (or None if G was not provided).
        batch_size: Batch size used for the DataLoaders.
    """
    n_features: int = train_data[0].x.shape[1]
    n_edge_features: int = edge_attr.shape[1] if edge_attr is not None else 0

    # Aggregate all target values for class-balance stats
    all_targets: np.ndarray = np.concatenate(
        [np.array(r["final_severities"]) for r in runs]
    )
    frac_disrupted: float = float(np.mean(all_targets > 0.0))
    mean_sev: float = float(np.mean(all_targets))
    max_sev: float = float(np.max(all_targets))

    print(f"\n[dataset] ── Statistics ─────────────────────────────────────")
    print(f"[dataset]  Total graphs : {len(runs):,}  "
          f"(train={len(train_data)}  val={len(val_data)}  test={len(test_data)})")
    print(f"[dataset]  Node features: {n_features}")
    print(f"[dataset]  Edge features: {n_edge_features if n_edge_features else 'N/A (G not provided)'}")
    print(f"[dataset]  Target (y)   : severity in [0, 1]  "
          f"mean={mean_sev:.4f}  max={max_sev:.4f}")
    print(f"[dataset]  Class balance: {frac_disrupted*100:.1f}% nodes disrupted (severity > 0)")
    print(f"[dataset]  Batch size   : {batch_size}")
    print(f"[dataset] ────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_dataloaders(
    pkl_path: Path = PKL_PATH,
    G: Optional[nx.DiGraph] = None,
    batch_size: int = BATCH_SIZE,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = SPLIT_SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load simulation_runs.pkl and return train/val/test PyG DataLoaders.

    Each simulation run becomes one PyG ``Data`` graph:
      - ``x``          : (N, 10) normalized node feature matrix.
      - ``edge_index`` : (2, E) directed edge index.
      - ``edge_attr``  : (E, 2) [criticality_weight, flow_capacity] if G given.
      - ``y``          : (N,)  final disruption severity per node in [0, 1].

    Normalization is fitted on the training split only to prevent leakage.
    Pass the same ``seed`` for reproducible splits.

    Args:
        pkl_path: Path to ``simulation_runs.pkl``. Defaults to
            ``data/processed/simulation_runs.pkl``.
        G: DTNet DiGraph with edge attributes. Required for ``edge_attr``;
            if None, ``edge_attr`` is set to None in all Data objects.
        batch_size: Mini-batch size for DataLoaders. Default 32.
        train_ratio: Fraction of graphs for training. Default 0.70.
        val_ratio: Fraction of graphs for validation. Default 0.15.
            Test fraction = 1 - train_ratio - val_ratio (default 0.15).
        seed: Random seed for reproducible train/val/test split. Default 42.

    Returns:
        Tuple of ``(train_loader, val_loader, test_loader)``.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    pkl_path = Path(pkl_path)
    print(f"[dataset] Loading {pkl_path} ...")
    with open(pkl_path, "rb") as fh:
        runs: List[Dict] = pickle.load(fh)

    n_runs: int = len(runs)
    node_order: List[str] = runs[0]["node_order"]
    n_nodes: int = len(node_order)
    n_features: int = len(runs[0]["initial_features"][0])
    print(f"[dataset] Loaded {n_runs:,} runs — nodes={n_nodes}  features={n_features}")

    # ── Shared topology tensors (identical across all runs) ─────────────────
    edge_index: torch.Tensor = _build_edge_index_tensor(runs[0]["edge_index"])
    n_edges: int = edge_index.shape[1]

    edge_attr: Optional[torch.Tensor] = None
    if G is not None:
        edge_attr = _extract_edge_attrs(G, node_order)
        print(f"[dataset] Edge attrs extracted — edges={n_edges}  edge_features=2")
    else:
        print(f"[dataset] No G provided — edge_attr=None  edges={n_edges}")

    # ── Train / Val / Test split (stratified by index) ──────────────────────
    test_ratio: float = 1.0 - train_ratio - val_ratio
    indices: List[int] = list(range(n_runs))

    train_idx, temp_idx = train_test_split(
        indices, test_size=(val_ratio + test_ratio), random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(test_ratio / (val_ratio + test_ratio)),
        random_state=seed,
    )

    # ── Fit scaler on training features only (prevents leakage) ─────────────
    scaler: StandardScaler = _fit_scaler(runs, train_idx)

    # ── Build Data object lists ──────────────────────────────────────────────
    def _build_list(idx_list: List[int]) -> List[Data]:
        """Convert a list of run indices to PyG Data objects."""
        return [
            _run_to_data(runs[i], edge_index, edge_attr, scaler)
            for i in idx_list
        ]

    train_data: List[Data] = _build_list(train_idx)
    val_data:   List[Data] = _build_list(val_idx)
    test_data:  List[Data] = _build_list(test_idx)

    # ── Wrap in DataLoaders ──────────────────────────────────────────────────
    train_loader: DataLoader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader:   DataLoader = DataLoader(val_data,   batch_size=batch_size, shuffle=False)
    test_loader:  DataLoader = DataLoader(test_data,  batch_size=batch_size, shuffle=False)

    _print_statistics(runs, train_data, val_data, test_data, edge_attr, batch_size)

    return train_loader, val_loader, test_loader
