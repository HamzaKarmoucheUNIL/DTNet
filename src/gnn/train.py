"""train.py — Training loop for DTNetGNN and IsolatedBaseline.

Trains both models on simulation run data loaded via ``build_dataloaders``.
Applies early stopping on validation MSE and saves the best checkpoint for
each model.  Returns full training histories for downstream analysis.

Public API: ``run_training(train_loader, val_loader, device) -> (gnn_hist, base_hist)``
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.gnn.dataset import build_dataloaders
from src.gnn.model import DTNetGNN, IsolatedBaseline

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader  # type: ignore[no-redef]

np.random.seed(42)
torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LR: float = 0.001
WEIGHT_DECAY: float = 5e-4
PATIENCE: int = 20
MAX_EPOCHS: int = 200
LOG_INTERVAL: int = 10
GNN_SAVE_PATH: Path = Path("results/dtnet_gnn_best.pt")
BASELINE_SAVE_PATH: Path = Path("results/isolated_baseline_best.pt")

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch; return mean MSE loss averaged over all nodes."""
    model.train()
    total_loss: float = 0.0
    total_nodes: int = 0

    for batch in loader:
        batch = batch.to(device)
        edge_attr: torch.Tensor | None = getattr(batch, "edge_attr", None)

        optimizer.zero_grad()
        preds: torch.Tensor = model(batch.x, batch.edge_index, edge_attr)
        loss: torch.Tensor = criterion(preds, batch.y)
        loss.backward()
        optimizer.step()

        n_nodes: int = batch.y.shape[0]
        total_loss += loss.item() * n_nodes
        total_nodes += n_nodes

    return total_loss / total_nodes


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on a loader; return {mse, mae, r2} computed globally."""
    model.eval()
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    for batch in loader:
        batch = batch.to(device)
        edge_attr: torch.Tensor | None = getattr(batch, "edge_attr", None)
        preds: torch.Tensor = model(batch.x, batch.edge_index, edge_attr)
        all_preds.append(preds.cpu())
        all_targets.append(batch.y.cpu())

    y_pred: torch.Tensor = torch.cat(all_preds)    # (N_total,)
    y_true: torch.Tensor = torch.cat(all_targets)  # (N_total,)

    mse: float = criterion(y_pred, y_true).item()
    mae: float = torch.mean(torch.abs(y_pred - y_true)).item()

    ss_res: torch.Tensor = torch.sum((y_true - y_pred) ** 2)
    ss_tot: torch.Tensor = torch.sum((y_true - y_true.mean()) ** 2)
    r2: float = (1.0 - ss_res / (ss_tot + 1e-8)).item()

    return {"mse": mse, "mae": mae, "r2": r2}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    save_path: Path,
    device: torch.device,
    label: str = "model",
) -> Dict[str, List[float] | int | float]:
    """Train a single model with early stopping; save best checkpoint.

    Trains for up to MAX_EPOCHS epochs.  Stops early if validation MSE does
    not improve for PATIENCE consecutive epochs.  Saves ``model.state_dict()``
    at the epoch with the lowest validation MSE.

    Args:
        model: Initialised model to train (DTNetGNN or IsolatedBaseline).
        train_loader: DataLoader for the training split.
        val_loader: DataLoader for the validation split.
        save_path: File path to write the best model state dict (.pt).
        device: Device to move tensors and model to.
        label: Human-readable name for log lines.

    Returns:
        History dict with keys:
          ``train_loss``  (List[float]) — per-epoch training MSE,
          ``val_loss``    (List[float]) — per-epoch validation MSE,
          ``val_mae``     (List[float]) — per-epoch validation MAE,
          ``best_epoch``  (int)         — epoch index (0-based) of best model,
          ``best_val_loss`` (float)     — best validation MSE achieved.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.to(device)

    optimizer: torch.optim.Optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    criterion: nn.Module = nn.MSELoss()

    history: Dict[str, List[float] | int | float] = {
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "best_epoch": 0,
        "best_val_loss": float("inf"),
    }

    patience_counter: int = 0

    print(f"\n[train] ── Training {label} ──────────────────────────────────────")

    for epoch in range(MAX_EPOCHS):
        train_loss: float = _train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics: Dict[str, float] = _evaluate(model, val_loader, criterion, device)

        val_loss: float = val_metrics["mse"]
        val_mae: float = val_metrics["mae"]

        history["train_loss"].append(train_loss)   # type: ignore[union-attr]
        history["val_loss"].append(val_loss)        # type: ignore[union-attr]
        history["val_mae"].append(val_mae)          # type: ignore[union-attr]

        # Log every LOG_INTERVAL epochs (and always on epoch 0)
        if epoch % LOG_INTERVAL == 0:
            print(
                f"[train] {label}  epoch={epoch:03d}"
                f"  train_mse={train_loss:.6f}"
                f"  val_mse={val_loss:.6f}"
                f"  val_mae={val_mae:.6f}"
            )

        # Early stopping — track best val MSE
        if val_loss < history["best_val_loss"]:  # type: ignore[operator]
            history["best_val_loss"] = val_loss
            history["best_epoch"] = epoch
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(
                    f"[train] Early stopping at epoch {epoch} "
                    f"(no improvement for {PATIENCE} epochs)"
                )
                break

    best_ep: int = history["best_epoch"]  # type: ignore[assignment]
    print(
        f"[train] Best {label}: epoch={best_ep}"
        f"  val_mse={history['best_val_loss']:.6f}"
        f"  → saved to {save_path}"
    )
    return history


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_training(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device | None = None,
) -> Tuple[Dict, Dict]:
    """Train DTNetGNN and IsolatedBaseline; print final comparison on val set.

    Trains both models with identical hyperparameters (Adam lr=0.001,
    weight_decay=5e-4, MSE loss, early stopping patience=20, max 200 epochs).
    After training, loads each model's best checkpoint and evaluates on the
    validation set, then prints a side-by-side MSE / MAE / R² comparison.

    Args:
        train_loader: PyG DataLoader for the training split.
        val_loader: PyG DataLoader for the validation split.
        device: Torch device.  Defaults to CUDA if available, else CPU.

    Returns:
        Tuple ``(gnn_history, baseline_history)``.  Each history dict has keys
        ``train_loss``, ``val_loss``, ``val_mae``, ``best_epoch``,
        ``best_val_loss`` (see ``train_model`` for details).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # Infer in_channels from the first batch
    sample_batch = next(iter(train_loader))
    in_channels: int = sample_batch.x.shape[1]

    # ── Train DTNetGNN ───────────────────────────────────────────────────────
    gnn_model: DTNetGNN = DTNetGNN(in_channels=in_channels)
    gnn_history: Dict = train_model(
        gnn_model, train_loader, val_loader, GNN_SAVE_PATH, device, label="DTNetGNN"
    )

    # ── Train IsolatedBaseline ───────────────────────────────────────────────
    baseline_model: IsolatedBaseline = IsolatedBaseline(in_channels=in_channels)
    baseline_history: Dict = train_model(
        baseline_model, train_loader, val_loader,
        BASELINE_SAVE_PATH, device, label="IsolatedBaseline"
    )

    # ── Final comparison: load best checkpoints and re-evaluate ─────────────
    criterion: nn.Module = nn.MSELoss()

    gnn_model.load_state_dict(torch.load(GNN_SAVE_PATH, map_location=device))
    gnn_val: Dict[str, float] = _evaluate(gnn_model, val_loader, criterion, device)

    baseline_model.load_state_dict(
        torch.load(BASELINE_SAVE_PATH, map_location=device)
    )
    base_val: Dict[str, float] = _evaluate(baseline_model, val_loader, criterion, device)

    print("\n[train] ── Final Validation Comparison ──────────────────────────────")
    print(f"[train]  {'Model':<22}  {'MSE':>10}  {'MAE':>10}  {'R²':>10}")
    print(f"[train]  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*10}")
    print(
        f"[train]  {'DTNetGNN':<22}"
        f"  {gnn_val['mse']:>10.6f}"
        f"  {gnn_val['mae']:>10.6f}"
        f"  {gnn_val['r2']:>10.4f}"
    )
    print(
        f"[train]  {'IsolatedBaseline':<22}"
        f"  {base_val['mse']:>10.6f}"
        f"  {base_val['mae']:>10.6f}"
        f"  {base_val['r2']:>10.4f}"
    )
    print("[train] ────────────────────────────────────────────────────────────\n")

    return gnn_history, baseline_history


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_loader, val_loader, _ = build_dataloaders()
    run_training(train_loader, val_loader)
