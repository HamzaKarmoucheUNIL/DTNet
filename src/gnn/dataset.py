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
    """Extract (criticality_weight, flow_capacity, shared_parts_count) per directed edge.

    Edges are iterated in ``G.edges()`` order, which matches the order used
    in ``data_generator._build_edge_index``, so indices align with edge_index.
    ``shared_parts_count`` is normalized by the maximum count across all edges
    to keep it in [0, 1].

    Args:
        G: DTNet DiGraph with ``criticality_weight``, ``flow_capacity``, and
            ``shared_parts_count`` on every edge.
        node_order: Canonical node ordering (used only for consistency check).

    Returns:
        Tensor of shape (E, 3): columns are
        [criticality_weight, flow_capacity, shared_parts_count_normalized].
    """
    edge_data: List[tuple] = list(G.edges(data=True))
    shared_counts: List[float] = [
        float(d.get("shared_parts_count", 0.0)) for _, _, d in edge_data
    ]
    max_shared: float = max(shared_counts) if shared_counts else 1.0
    if max_shared == 0.0:
        max_shared = 1.0

    rows: List[List[float]] = []
    for (_, _, data), sc in zip(edge_data, shared_counts):
        cw: float = float(data.get("criticality_weight", 0.5))
        fc: float = float(data.get("flow_capacity", 0.9))
        rows.append([cw, fc, sc / max_shared])
    return torch.tensor(rows, dtype=torch.float)


def _compute_structural_features(
    G: nx.DiGraph,
    node_order: List[str],
) -> np.ndarray:
    """Compute 6 structural/topological features for every node in node_order.

    All six metrics are computed once on G and aligned to ``node_order``.
    Values are already in [0, 1] (centrality measures are normalised by
    definition; in/out-degree is divided by the graph maximum).

    Feature order (appended after existing twin features):
      [degree_centrality, betweenness_centrality, in_degree_norm,
       out_degree_norm, closeness_centrality, pagerank]

    Args:
        G: DTNet DiGraph (topology must match ``node_order``).
        node_order: Canonical node ID ordering matching simulation run dicts.

    Returns:
        ``np.ndarray`` of shape ``(N, 6)`` with dtype float32, values in [0, 1].
    """
    deg_cen: Dict[str, float] = nx.degree_centrality(G)
    btw_cen: Dict[str, float] = nx.betweenness_centrality(G, normalized=True)
    clo_cen: Dict[str, float] = nx.closeness_centrality(G)
    pgr: Dict[str, float] = nx.pagerank(G)

    in_deg_raw: Dict[str, int] = dict(G.in_degree())
    out_deg_raw: Dict[str, int] = dict(G.out_degree())
    max_in: float = float(max(in_deg_raw.values())) if in_deg_raw else 1.0
    max_out: float = float(max(out_deg_raw.values())) if out_deg_raw else 1.0
    if max_in == 0.0:
        max_in = 1.0
    if max_out == 0.0:
        max_out = 1.0

    rows: List[List[float]] = []
    for nid in node_order:
        rows.append([
            float(deg_cen.get(nid, 0.0)),
            float(btw_cen.get(nid, 0.0)),
            float(in_deg_raw.get(nid, 0)) / max_in,
            float(out_deg_raw.get(nid, 0)) / max_out,
            float(clo_cen.get(nid, 0.0)),
            float(pgr.get(nid, 0.0)),
        ])
    return np.array(rows, dtype=np.float32)   # (N, 6)


def _fit_scaler(
    runs: List[Dict],
    train_indices: List[int],
    struct_feats: Optional[np.ndarray] = None,
) -> StandardScaler:
    """Fit a StandardScaler on the training-set node features.

    Stacks all node feature matrices from training runs into one matrix and
    fits the scaler.  If ``struct_feats`` is provided (shape ``(N, 6)``),
    the structural features are concatenated to each run's feature matrix
    BEFORE fitting so the scaler covers the full augmented feature space.
    Columns with zero variance have their scale set to 1.0 to prevent
    division-by-zero during transform.

    Args:
        runs: Full list of simulation run dicts.
        train_indices: Indices of the runs assigned to the training split.
        struct_feats: Optional ``(N_nodes, 6)`` array of pre-computed
            structural/topological features to append before fitting.

    Returns:
        Fitted ``StandardScaler`` ready for ``transform()``.
    """
    matrices: List[np.ndarray] = []
    for i in train_indices:
        feat: np.ndarray = np.array(runs[i]["initial_features"], dtype=np.float32)
        if struct_feats is not None:
            feat = np.concatenate([feat, struct_feats], axis=1)  # (N, F+6)
        matrices.append(feat)
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
    struct_feats: Optional[np.ndarray] = None,
) -> Data:
    """Convert a single simulation run dict to a PyG Data object.

    Node features (x) are normalized with the pre-fitted StandardScaler.
    If ``struct_feats`` is provided it is concatenated to each run's raw
    feature matrix BEFORE scaling (the scaler was fitted on the same layout).
    Targets (y) are final disruption severities in [0, 1] — no normalization
    applied so they remain interpretable as probabilities.

    Args:
        run: Run dict with keys ``initial_features``, ``final_severities``,
            ``edge_index``, ``node_order``.
        edge_index: Precomputed (2, E) edge index tensor (shared across runs).
        edge_attr: Optional (E, 3) edge attribute tensor (shared across runs).
        scaler: Fitted StandardScaler for node features.
        struct_feats: Optional ``(N, 6)`` structural feature array to append.

    Returns:
        ``torch_geometric.data.Data`` with x, edge_index, edge_attr, y.
    """
    # Node features — normalize with training scaler (COMMON_MISTAKES #7)
    x_raw: np.ndarray = np.array(run["initial_features"], dtype=np.float32)
    if struct_feats is not None:
        x_raw = np.concatenate([x_raw, struct_feats], axis=1)  # (N, F+6)
    x_norm: np.ndarray = scaler.transform(x_raw).astype(np.float32)
    x: torch.Tensor = torch.from_numpy(x_norm)                 # (N, F) or (N, F+6)

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
      - ``x``          : (N, 16) normalized node feature matrix
                         (10 twin features + 6 structural, when G is provided).
      - ``edge_index`` : (2, E) directed edge index.
      - ``edge_attr``  : (E, 3) [criticality_weight, flow_capacity,
                         shared_parts_count_norm] if G given.
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
    struct_feats: Optional[np.ndarray] = None
    if G is not None:
        edge_attr = _extract_edge_attrs(G, node_order)
        print(f"[dataset] Edge attrs extracted — edges={n_edges}  edge_features=3")
        struct_feats = _compute_structural_features(G, node_order)
        print(f"[dataset] Structural features computed — {struct_feats.shape[1]} per node")
    else:
        print(f"[dataset] No G provided — edge_attr=None  struct_feats=None  edges={n_edges}")

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
    scaler: StandardScaler = _fit_scaler(runs, train_idx, struct_feats)

    n_features_new: int = n_features + (struct_feats.shape[1] if struct_feats is not None else 0)
    print(f"[dataset] Feature count: {n_features} (original) → {n_features_new} (with structural)")

    # ── Build Data object lists ──────────────────────────────────────────────
    def _build_list(idx_list: List[int]) -> List[Data]:
        """Convert a list of run indices to PyG Data objects."""
        return [
            _run_to_data(runs[i], edge_index, edge_attr, scaler, struct_feats)
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
