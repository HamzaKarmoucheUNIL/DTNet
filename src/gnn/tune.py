"""tune.py — Hyperparameter grid search for DTNetGNN.

Runs a 3×3×3×3 grid (81 trials) over hidden_channels, num_heads,
learning_rate, and dropout.  Each trial trains for TUNE_EPOCHS epochs
and is evaluated on the validation set.  Top-10 combinations are ranked
by combined score = val_mse − val_f1 (lower is better: penalises high
regression error and low classification F1 equally).  Best config saved
to results/best_hyperparams.json.

Run with: python -m src.gnn.tune
"""

from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from src.data.entity_mapping import build_entity_mappings
from src.data.loader import load_csv
from src.data.preprocess import preprocess
from src.graph.builder import build_graph
from src.graph.topology import infer_topology
from src.gnn.dataset import PKL_PATH, build_dataloaders
from src.gnn.model import DTNetGNN
from src.gnn.train import CLS_LOSS_WEIGHT, DISRUPTION_THRESHOLD, WEIGHT_DECAY

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader  # type: ignore[no-redef]

np.random.seed(42)
torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Grid and tuning constants (COMMON_MISTAKES #5 — no hardcoding)
# ---------------------------------------------------------------------------

HIDDEN_CHANNELS_GRID: List[int]  = [32, 64, 128]
NUM_HEADS_GRID: List[int]        = [2, 4, 8]
LR_GRID: List[float]             = [0.001, 0.005, 0.01]
DROPOUT_GRID: List[float]        = [0.2, 0.3, 0.5]

TUNE_EPOCHS: int        = 50
SEED: int               = 42      # reset per trial (COMMON_MISTAKES #3)
CLS_THRESHOLD: float    = 0.5    # sigmoid output threshold for hard label
RESULTS_DIR: Path       = Path("results")
BEST_HP_PATH: Path      = RESULTS_DIR / "best_hyperparams.json"
CSV_PATH: str           = "updated_data.csv"

# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------


def _run_trial(
    hidden: int,
    heads: int,
    lr: float,
    dropout: float,
    in_channels: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Train one hyperparameter combination for TUNE_EPOCHS; return (val_mse, val_f1).

    Seed is reset at the start of every trial so results are comparable
    regardless of evaluation order (COMMON_MISTAKES #3).

    Args:
        hidden: Hidden channel dimension per attention head.
        heads: Number of attention heads in the first GATConv layer.
        lr: Adam learning rate.
        dropout: Dropout probability after each GAT layer.
        in_channels: Node feature dimension (derived from loader).
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        device: Torch device.

    Returns:
        Tuple ``(val_mse, val_f1)`` on the validation split.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model: DTNetGNN = DTNetGNN(
        in_channels=in_channels,
        hidden_channels=hidden,
        heads_1=heads,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    model.train()
    for _ in range(TUNE_EPOCHS):
        for batch in train_loader:
            batch = batch.to(device)
            ea = getattr(batch, "edge_attr", None)
            optimizer.zero_grad()
            reg_out, cls_out = model(batch.x, batch.edge_index, ea)
            cls_tgt = (batch.y > DISRUPTION_THRESHOLD).float()
            loss = (F.mse_loss(reg_out, batch.y)
                    + CLS_LOSS_WEIGHT * F.binary_cross_entropy_with_logits(cls_out, cls_tgt))
            loss.backward()
            optimizer.step()

    model.eval()
    all_reg: List[torch.Tensor] = []
    all_cls: List[torch.Tensor] = []
    all_y:   List[torch.Tensor] = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            ea = getattr(batch, "edge_attr", None)
            reg_out, cls_out = model(batch.x, batch.edge_index, ea)
            all_reg.append(reg_out.cpu())
            all_cls.append(cls_out.cpu())
            all_y.append(batch.y.cpu())

    y_pred = torch.cat(all_reg)
    y_cls  = torch.cat(all_cls)
    y_true = torch.cat(all_y)
    val_mse = float(F.mse_loss(y_pred, y_true).item())
    y_bin = (y_true.numpy() > DISRUPTION_THRESHOLD).astype(int)
    y_hat = (torch.sigmoid(y_cls).numpy() >= CLS_THRESHOLD).astype(int)
    return val_mse, float(f1_score(y_bin, y_hat, zero_division=0))


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------


def tune(pkl_path: Path = PKL_PATH) -> Dict:
    """Run the full hyperparameter grid search; return the best config dict.

    Builds graph + dataloaders once (structural features included), then
    iterates all 81 combinations, printing per-trial progress and a ranked
    top-10 table on completion.  Saves the best config to BEST_HP_PATH.

    Args:
        pkl_path: Path to ``simulation_runs.pkl``.

    Returns:
        Dict with keys ``hidden``, ``heads``, ``lr``, ``dropout``,
        ``val_mse``, ``val_f1``, ``score``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[tune] Device: {device}")

    df_raw = load_csv(CSV_PATH)
    df_clean, _ = preprocess(df_raw)
    em = build_entity_mappings(df_raw)
    nodes, edges = infer_topology(em)
    G = build_graph(nodes, edges, df_clean)
    print(f"[tune] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    train_loader, val_loader, _ = build_dataloaders(pkl_path, G=G)
    in_channels: int = next(iter(train_loader)).x.shape[1]
    combos = list(product(HIDDEN_CHANNELS_GRID, NUM_HEADS_GRID, LR_GRID, DROPOUT_GRID))
    print(f"[tune] {len(combos)} combinations × {TUNE_EPOCHS} epochs  in_ch={in_channels}\n")

    results: List[Dict] = []
    t0 = time.time()

    for idx, (hidden, heads, lr, dropout) in enumerate(combos, 1):
        val_mse, val_f1 = _run_trial(
            hidden, heads, lr, dropout, in_channels, train_loader, val_loader, device
        )
        score = round(val_mse - val_f1, 6)
        results.append({
            "hidden": hidden, "heads": heads, "lr": lr, "dropout": dropout,
            "val_mse": round(val_mse, 6), "val_f1": round(val_f1, 6), "score": score,
        })
        eta = (time.time() - t0) / idx * (len(combos) - idx)
        print(
            f"[tune] {idx:3d}/{len(combos)}"
            f"  h={hidden:3d} heads={heads} lr={lr:.3f} drop={dropout:.1f}"
            f"  mse={val_mse:.4f} f1={val_f1:.4f}  eta={eta/60:.1f}m"
        )

    results.sort(key=lambda r: r["score"])

    print("\n[tune] ── Top-10 Combinations ───────────────────────────────────────────")
    print(f"  {'Rank':>4}  {'Hidden':>6}  {'Heads':>5}  {'LR':>7}  {'Dropout':>7}  {'Val_MSE':>8}  {'Val_F1':>7}")
    print(f"  {'─'*4}  {'─'*6}  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*7}")
    for rank, r in enumerate(results[:10], 1):
        print(
            f"  {rank:>4}  {r['hidden']:>6}  {r['heads']:>5}  {r['lr']:>7.4f}"
            f"  {r['dropout']:>7.1f}  {r['val_mse']:>8.4f}  {r['val_f1']:>7.4f}"
        )

    best = results[0]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(BEST_HP_PATH, "w") as fh:
        json.dump(best, fh, indent=2)
    print(f"\n[tune] Best: hidden={best['hidden']}  heads={best['heads']}"
          f"  lr={best['lr']}  dropout={best['dropout']}  score={best['score']:.4f}")
    print(f"[tune] Saved → {BEST_HP_PATH}")
    return best


if __name__ == "__main__":
    tune()
