"""evaluate.py — Test-set evaluation, attention analysis, and scenario testing.

Loads saved DTNetGNN and IsolatedBaseline checkpoints, evaluates on the test
set with both regression metrics (MSE, MAE, R²) and classification metrics
(Accuracy, F1, Precision, Recall, AUC-ROC), extracts GATConv attention weights
(which edges matter most?), and runs scenario comparisons.

Public API: ``run_evaluation(test_loader, G, runs, device) -> Dict``
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader  # type: ignore[no-redef]

from src.gnn.dataset import _compute_structural_features, build_dataloaders
from src.gnn.model import DTNetGNN, IsolatedBaseline
from src.gnn.train import (
    BASELINE_SAVE_PATH, DISRUPTION_THRESHOLD, DROPOUT, GNN_SAVE_PATH,
    HIDDEN_CHANNELS, NUM_HEADS,
)

np.random.seed(42)
torch.manual_seed(42)

LAYER_ORDER: List[str] = ["supplier", "logistics", "plant", "machine", "distribution"]
ONE_HOT_START: int = 5         # first one-hot layer flag in the feature vector
TOP_K_EDGES: int = 10          # top attention-weighted edges to report
CLS_DECISION_THRESHOLD: float = 0.5   # sigmoid output threshold for binary prediction


def _compute_metrics(y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
    """Return MSE, MAE, and R² between ``y_pred`` and ``y_true``."""
    mse: float = F.mse_loss(y_pred, y_true).item()
    mae: float = torch.mean(torch.abs(y_pred - y_true)).item()
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
    return {"mse": mse, "mae": mae, "r2": (1.0 - ss_res / (ss_tot + 1e-8)).item()}


def _compute_cls_metrics(
    y_cls: torch.Tensor, y_true: torch.Tensor
) -> Dict[str, float]:
    """Return Accuracy, F1, Precision, Recall, AUC-ROC for the classification head.

    Binary target: ``y_true > DISRUPTION_THRESHOLD`` (0.3).
    Binary prediction: ``sigmoid(y_cls) >= CLS_DECISION_THRESHOLD`` (0.5).
    AUC uses the continuous sigmoid probability, not the hard label.

    Args:
        y_cls: Raw logits of shape ``(N,)`` from the classification head.
        y_true: Ground-truth severity of shape ``(N,)``.

    Returns:
        Dict with keys ``accuracy``, ``f1``, ``precision``, ``recall``, ``auc``.
    """
    y_bin: np.ndarray = (y_true.numpy() > DISRUPTION_THRESHOLD).astype(int)
    y_prob: np.ndarray = torch.sigmoid(y_cls).numpy()
    y_hat: np.ndarray = (y_prob >= CLS_DECISION_THRESHOLD).astype(int)
    auc: float = (
        float(roc_auc_score(y_bin, y_prob)) if len(np.unique(y_bin)) > 1 else 0.0
    )
    return {
        "accuracy":  float(accuracy_score(y_bin, y_hat)),
        "f1":        float(f1_score(y_bin, y_hat, zero_division=0)),
        "precision": float(precision_score(y_bin, y_hat, zero_division=0)),
        "recall":    float(recall_score(y_bin, y_hat, zero_division=0)),
        "auc":       auc,
    }


@torch.no_grad()
def _collect(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(y_pred, y_cls, y_true, x_all)`` concatenated over all batches.

    ``y_pred`` is the regression output (severity in [0, 1]).
    ``y_cls`` is the raw classification logit (pass through sigmoid for probability).
    """
    model.eval()
    preds: List[torch.Tensor] = []
    cls_logits: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    feats: List[torch.Tensor] = []
    for batch in loader:
        batch = batch.to(device)
        ea: Optional[torch.Tensor] = getattr(batch, "edge_attr", None)
        reg_out, cls_out = model(batch.x, batch.edge_index, ea)
        preds.append(reg_out.cpu())
        cls_logits.append(cls_out.cpu())
        targets.append(batch.y.cpu())
        feats.append(batch.x.cpu())
    return torch.cat(preds), torch.cat(cls_logits), torch.cat(targets), torch.cat(feats)


def _per_type_metrics(
    y_pred: torch.Tensor, y_true: torch.Tensor, x_all: torch.Tensor,
    y_cls: Optional[torch.Tensor] = None,
) -> Dict[str, Dict[str, float]]:
    """Return ``{mse, mae, r2[, f1]}`` per node layer type, inferred via argmax of one-hot slice.

    Node type is recovered by ``argmax(x[:, 5:10])``: valid post-normalisation
    because StandardScaler preserves relative ordering within each binary column.
    When ``y_cls`` is provided, F1 is added to each layer's metric dict.
    """
    layer_idx = torch.argmax(x_all[:, ONE_HOT_START: ONE_HOT_START + 5], dim=1)
    result: Dict[str, Dict[str, float]] = {}
    for li, name in enumerate(LAYER_ORDER):
        mask = layer_idx == li
        if not mask.any():
            continue
        m: Dict[str, float] = _compute_metrics(y_pred[mask], y_true[mask])
        if y_cls is not None:
            m["f1"] = _compute_cls_metrics(y_cls[mask], y_true[mask])["f1"]
        result[name] = m
    return result


@torch.no_grad()
def _extract_attention(
    gnn: DTNetGNN, data: Data, device: torch.device,
    G: Optional[nx.DiGraph] = None,
) -> Dict[str, Any]:
    """Extract per-edge GAT attention weights from both GATConv layers.

    Calls each layer with ``return_attention_weights=True``, averages attention
    over heads and layers, filters self-loops, and returns ``top_k_edges``.
    If ``G`` is provided, edges are labelled with human-readable node names.
    """
    gnn.eval()
    x = data.x.to(device)
    ei = data.edge_index.to(device)
    ea = data.edge_attr.to(device) if data.edge_attr is not None else None

    out1, (ei_ret, a1) = gnn.conv1(x, ei, edge_attr=ea, return_attention_weights=True)
    _, (_, a2) = gnn.conv2(F.relu(out1), ei, edge_attr=ea, return_attention_weights=True)

    mean_alpha: torch.Tensor = (a1.mean(-1) + a2.mean(-1)).cpu() / 2.0
    ei_cpu: torch.Tensor = ei_ret.cpu()

    keep = ei_cpu[0] != ei_cpu[1]   # filter self-loops
    fi, fa = ei_cpu[:, keep], mean_alpha[keep]

    k = min(TOP_K_EDGES, fa.shape[0])
    topk_vals, topk_pos = torch.topk(fa, k)
    node_order: Optional[List[str]] = list(G.nodes()) if G is not None else None

    top_k: List[Dict[str, Any]] = []
    for rank, (pos, val) in enumerate(zip(topk_pos.tolist(), topk_vals.tolist())):
        src, dst = int(fi[0, pos]), int(fi[1, pos])
        entry: Dict[str, Any] = {"rank": rank + 1, "src": src, "dst": dst, "attention": val}
        if node_order and src < len(node_order) and dst < len(node_order):
            entry["src_name"] = node_order[src]
            entry["dst_name"] = node_order[dst]
        top_k.append(entry)

    return {"layer1_alpha": a1.cpu(), "layer2_alpha": a2.cpu(),
            "mean_alpha": mean_alpha, "top_k_edges": top_k}


def _find_scenario_run(
    runs: List[Dict[str, Any]], scenario: str
) -> Optional[Dict[str, Any]]:
    """Return first run matching ``'single_supplier'`` or ``'multi_node'``.

    Single-supplier: 1 disrupted node with raw ``is_supplier`` flag == 1.0
    (``initial_features[idx][ONE_HOT_START]`` before normalisation).
    Multi-node: two or more initially disrupted nodes.
    """
    for run in runs:
        dis: Dict[str, float] = run["initial_disruption"]
        order: List[str] = run["node_order"]
        if scenario == "single_supplier" and len(dis) == 1:
            nid = next(iter(dis))
            if float(run["initial_features"][order.index(nid)][ONE_HOT_START]) == 1.0:
                return run
        elif scenario == "multi_node" and len(dis) > 1:
            return run
    return None


@torch.no_grad()
def _eval_scenario(
    gnn: DTNetGNN, baseline: IsolatedBaseline,
    run: Dict[str, Any], scaler: StandardScaler, device: torch.device,
    struct_feats: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Build a Data object from a raw run dict and compare both model predictions.

    ``edge_attr`` is set to None: raw run dicts do not store edge attributes
    (those require the NetworkX graph G).  The GNN uses structure-only attention.
    """
    x_raw: np.ndarray = np.array(run["initial_features"], dtype=np.float32)
    if struct_feats is not None:
        # Scaler was fitted on all 16 dims — concatenate before transforming.
        x = torch.from_numpy(
            scaler.transform(np.concatenate([x_raw, struct_feats], axis=1)).astype(np.float32)
        )
    else:
        # Scaler covers only the 10 base dims; compute structural features from
        # the run's own edge topology and append them after scaling.
        _node_order: List[str] = run["node_order"]
        _tmp_G: nx.DiGraph = nx.DiGraph()
        _tmp_G.add_nodes_from(_node_order)
        for _s, _d in run["edge_index"]:
            _tmp_G.add_edge(_node_order[_s], _node_order[_d])
        _sf: np.ndarray = _compute_structural_features(_tmp_G, _node_order)
        x = torch.from_numpy(
            np.concatenate([scaler.transform(x_raw), _sf], axis=1).astype(np.float32)
        )
    pairs: List[List[int]] = run["edge_index"]
    ei = (torch.tensor(pairs, dtype=torch.long).T.contiguous()
          if pairs else torch.zeros((2, 0), dtype=torch.long))
    y = torch.tensor(run["final_severities"], dtype=torch.float32)
    data = Data(x=x, edge_index=ei, edge_attr=None, y=y).to(device)

    gnn.eval(); baseline.eval()
    gp, _ = gnn(data.x, data.edge_index, None)
    bp, _ = baseline(data.x, data.edge_index, None)
    gp = gp.cpu(); bp = bp.cpu()
    y_cpu = y.cpu()
    order: List[str] = run["node_order"]
    return {
        "initial_disruption": run["initial_disruption"],
        "node_order": order,
        "y_true": y_cpu.tolist(),
        "gnn_pred": gp.tolist(),
        "baseline_pred": bp.tolist(),
        "gnn_metrics": _compute_metrics(gp, y_cpu),
        "baseline_metrics": _compute_metrics(bp, y_cpu),
        "disrupted_nodes_actual": [order[i] for i, s in enumerate(y_cpu.tolist()) if s > 0.0],
    }


def run_evaluation(
    test_loader: DataLoader,
    G: Optional[nx.DiGraph] = None,
    runs: Optional[List[Dict[str, Any]]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load best checkpoints and run full evaluation on the held-out test set.

    Evaluates DTNetGNN and IsolatedBaseline, computes per-node-type breakdowns,
    extracts GAT attention from the first test graph, and (when raw run dicts
    are supplied) runs single-supplier and multi-node scenario comparisons.

    Args:
        test_loader: PyG DataLoader for the held-out test split.
        G: Optional DTNet DiGraph; used to label edges in attention output.
        runs: Optional raw simulation run dicts (from data_generator or pkl).
            Required for scenario analysis; omitted if ``None``.
        device: Torch device. Defaults to CUDA if available, else CPU.

    Returns:
        Dict with keys ``gnn_test``, ``baseline_test``, ``per_type_gnn``,
        ``per_type_base``, ``attention``, ``scenario_single``,
        ``scenario_multi``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] Device: {device}")

    in_channels: int = next(iter(test_loader)).x.shape[1]
    gnn: DTNetGNN = DTNetGNN(
        in_channels=in_channels, hidden_channels=HIDDEN_CHANNELS,
        heads_1=NUM_HEADS, dropout=DROPOUT,
    ).to(device)
    gnn.load_state_dict(torch.load(GNN_SAVE_PATH, map_location=device))
    baseline: IsolatedBaseline = IsolatedBaseline(
        in_channels=in_channels, hidden_channels=HIDDEN_CHANNELS,
    ).to(device)
    baseline.load_state_dict(torch.load(BASELINE_SAVE_PATH, map_location=device))
    print(f"[evaluate] Loaded {GNN_SAVE_PATH}  |  {BASELINE_SAVE_PATH}")

    gnn_pred, gnn_cls, y_true, x_all = _collect(gnn, test_loader, device)
    base_pred, base_cls, _, _ = _collect(baseline, test_loader, device)

    gnn_test: Dict[str, float] = _compute_metrics(gnn_pred, y_true)
    baseline_test: Dict[str, float] = _compute_metrics(base_pred, y_true)
    gnn_cls_test: Dict[str, float] = _compute_cls_metrics(gnn_cls, y_true)
    base_cls_test: Dict[str, float] = _compute_cls_metrics(base_cls, y_true)
    per_type_gnn: Dict[str, Dict] = _per_type_metrics(gnn_pred, y_true, x_all, gnn_cls)
    per_type_base: Dict[str, Dict] = _per_type_metrics(base_pred, y_true, x_all, base_cls)
    attention: Dict[str, Any] = _extract_attention(
        gnn, next(iter(test_loader)).to_data_list()[0], device, G
    )

    _sep = "─" * 54
    print(f"\n[evaluate] ── Test Set Comparison {_sep}")
    print(f"[evaluate]  REGRESSION:")
    print(f"[evaluate]  {'Model':<22}  {'MSE':>10}  {'MAE':>10}  {'R²':>10}")
    print(f"[evaluate]  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*10}")
    for lbl, m in [("DTNetGNN", gnn_test), ("IsolatedBaseline", baseline_test)]:
        print(f"[evaluate]  {lbl:<22}  {m['mse']:>10.6f}  {m['mae']:>10.6f}  {m['r2']:>10.4f}")

    print(f"\n[evaluate]  CLASSIFICATION (disrupted threshold={DISRUPTION_THRESHOLD}):")
    print(f"[evaluate]  {'Model':<22}  {'Acc':>8}  {'F1':>8}  {'Prec':>8}  {'Recall':>8}  {'AUC':>8}")
    print(f"[evaluate]  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for lbl, cm in [("DTNetGNN", gnn_cls_test), ("IsolatedBaseline", base_cls_test)]:
        print(
            f"[evaluate]  {lbl:<22}"
            f"  {cm['accuracy']:>8.4f}  {cm['f1']:>8.4f}"
            f"  {cm['precision']:>8.4f}  {cm['recall']:>8.4f}  {cm['auc']:>8.4f}"
        )

    print("\n[evaluate] ── Per-Node-Type Breakdown ────────────────────────────────")
    for name in LAYER_ORDER:
        gm = per_type_gnn.get(name)
        bm = per_type_base.get(name)
        if not gm:
            continue
        g_str = (f"GNN  mse={gm['mse']:.4f} mae={gm['mae']:.4f}"
                 f" r2={gm['r2']:.3f} f1={gm.get('f1', 0):.3f}")
        b_str = (f"Base mse={bm['mse']:.4f} mae={bm['mae']:.4f}"
                 f" r2={bm['r2']:.3f} f1={bm.get('f1', 0):.3f}" if bm else "")
        print(f"[evaluate]  {name:<14}  {g_str}  |  {b_str}")

    print(f"\n[evaluate] ── Top-{TOP_K_EDGES} Attention Edges ─────────────────────────────")
    for e in attention["top_k_edges"]:
        src_lbl = e.get("src_name", str(e["src"]))
        dst_lbl = e.get("dst_name", str(e["dst"]))
        print(f"[evaluate]  #{e['rank']:2d}  {src_lbl} → {dst_lbl}  attn={e['attention']:.4f}")

    scenario_single: Optional[Dict] = None
    scenario_multi: Optional[Dict] = None

    if runs is not None:
        node_order: List[str] = runs[0]["node_order"]
        struct_feats: Optional[np.ndarray] = (
            _compute_structural_features(G, node_order) if G is not None else None
        )
        all_feats = np.vstack([
            np.concatenate([np.array(r["initial_features"], dtype=np.float32), struct_feats], axis=1)
            if struct_feats is not None
            else np.array(r["initial_features"], dtype=np.float32)
            for r in runs
        ])
        scaler: StandardScaler = StandardScaler().fit(all_feats)
        scaler.scale_ = np.where(scaler.scale_ == 0.0, 1.0, scaler.scale_)

        for key, label in [("single_supplier", "Single Supplier Failure"),
                            ("multi_node", "Multi-Node Failure")]:
            rd = _find_scenario_run(runs, key)
            if rd is None:
                print(f"[evaluate] WARNING: no '{key}' scenario run found.")
                continue
            res: Dict[str, Any] = _eval_scenario(gnn, baseline, rd, scaler, device, struct_feats)
            print(f"\n[evaluate] ── Scenario: {label} ────────────────────────────────")
            print(f"[evaluate]  Disrupted initially : {list(res['initial_disruption'].keys())}")
            print(f"[evaluate]  Cascaded to (actual): {res['disrupted_nodes_actual']}")
            for mlbl, mm in [("GNN     ", res["gnn_metrics"]), ("Baseline", res["baseline_metrics"])]:
                print(f"[evaluate]  {mlbl}  mse={mm['mse']:.4f}  mae={mm['mae']:.4f}  r2={mm['r2']:.3f}")
            if key == "single_supplier":
                scenario_single = res
            else:
                scenario_multi = res

    print("[evaluate] ────────────────────────────────────────────────────────────\n")
    return {
        "gnn_test": gnn_test,
        "baseline_test": baseline_test,
        "gnn_cls_test": gnn_cls_test,
        "base_cls_test": base_cls_test,
        "per_type_gnn": per_type_gnn,
        "per_type_base": per_type_base,
        "attention": attention,
        "scenario_single": scenario_single,
        "scenario_multi": scenario_multi,
    }


if __name__ == "__main__":
    from src.data.entity_mapping import build_entity_mappings
    from src.data.loader import load_csv
    from src.data.preprocess import preprocess
    from src.graph.builder import build_graph
    from src.graph.topology import infer_topology
    _pkl: Path = Path("data/processed/simulation_runs.pkl")
    _df_raw = load_csv("updated_data.csv")
    _df_clean, _ = preprocess(_df_raw)
    _em = build_entity_mappings(_df_raw)
    _nodes, _edges = infer_topology(_em)
    _G = build_graph(_nodes, _edges, _df_clean)
    _, _, _test = build_dataloaders(_pkl, G=_G)
    _runs: Optional[List[Dict]] = pickle.load(open(_pkl, "rb")) if _pkl.exists() else None
    run_evaluation(_test, G=_G, runs=_runs)
