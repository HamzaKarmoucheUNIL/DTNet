"""evaluate_robust.py — Robustness evaluation of DTNetGNN vs IsolatedBaseline across 5 seeds.

Full pipeline (data split → training → evaluation) per seed. Both models share
the same held-out test split (COMMON_MISTAKES #14); isolated baseline ignores
graph structure only. Saves mean±std summary to results/robustness_results.json.

Usage: python -m src.gnn.evaluate_robust
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from src.gnn.dataset import build_dataloaders
from src.gnn.evaluate import _collect
from src.gnn.model import DTNetGNN, IsolatedBaseline
from src.gnn.train import train_model

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
SEEDS: List[int] = [42, 123, 456, 789, 1024]
BINARY_THRESHOLD: float = 0.3
METRIC_NAMES: List[str] = ["mae", "rmse", "f1", "precision", "recall"]
PKL_PATH: Path = Path("data/processed/simulation_runs.pkl")
RESULTS_PATH: Path = Path("results/robustness_results.json")


def _extended_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    threshold: float = BINARY_THRESHOLD,
) -> Dict[str, float]:
    """Compute MAE, RMSE, F1, Precision, and Recall for a set of predictions.

    MAE and RMSE operate on raw severity scores; F1/Precision/Recall binarise
    predictions at ``threshold`` (disrupted = severity > threshold).

    Args:
        y_pred: Predicted disruption severities, shape (N,).
        y_true: Ground-truth disruption severities, shape (N,).
        threshold: Binary classification cut-off. Default 0.3.

    Returns:
        Dict with keys ``mae``, ``rmse``, ``f1``, ``precision``, ``recall``.
    """
    mae: float = torch.mean(torch.abs(y_pred - y_true)).item()
    rmse: float = math.sqrt(torch.mean((y_pred - y_true) ** 2).item())
    y_p = (y_pred.numpy() > threshold).astype(int)
    y_t = (y_true.numpy() > threshold).astype(int)
    return {
        "mae": mae,
        "rmse": rmse,
        "f1": float(f1_score(y_t, y_p, zero_division=0)),
        "precision": float(precision_score(y_t, y_p, zero_division=0)),
        "recall": float(recall_score(y_t, y_p, zero_division=0)),
    }


def run_one_seed(
    seed: int, pkl_path: Path, device: torch.device
) -> Dict[str, Dict[str, float]]:
    """Train DTNetGNN and IsolatedBaseline for one seed; return metrics for both.

    Sets np.random, torch, and random seeds before any split or weight init.
    Both models share the same ``test_loader`` (COMMON_MISTAKES #14): identical
    node features and targets; isolated baseline ignores edge structure only.

    Args:
        seed: Controls data split and weight initialisation.
        pkl_path: Path to simulation_runs.pkl.
        device: Torch device.

    Returns:
        ``{"networked": {mae, rmse, f1, precision, recall},
           "isolated":  {mae, rmse, f1, precision, recall}}``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_loader, val_loader, test_loader = build_dataloaders(pkl_path=pkl_path, seed=seed)
    in_ch: int = next(iter(train_loader)).x.shape[1]

    gnn: DTNetGNN = DTNetGNN(in_channels=in_ch)
    gnn_ckpt: Path = Path(f"results/robust_gnn_seed{seed}.pt")
    train_model(gnn, train_loader, val_loader, gnn_ckpt, device, label=f"GNN[s={seed}]")
    gnn.load_state_dict(torch.load(gnn_ckpt, map_location=device))

    baseline: IsolatedBaseline = IsolatedBaseline(in_channels=in_ch)
    base_ckpt: Path = Path(f"results/robust_baseline_seed{seed}.pt")
    train_model(
        baseline, train_loader, val_loader, base_ckpt, device,
        label=f"Isolated[s={seed}]",
    )
    baseline.load_state_dict(torch.load(base_ckpt, map_location=device))

    # Same test_loader for both — COMMON_MISTAKES #14
    gnn_pred, y_true, _ = _collect(gnn, test_loader, device)
    base_pred, _, _ = _collect(baseline, test_loader, device)

    return {
        "networked": _extended_metrics(gnn_pred, y_true),
        "isolated": _extended_metrics(base_pred, y_true),
    }


def summarize(all_results: List[Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
    """Aggregate per-seed metrics into mean ± std; print and return a summary dict.

    Δ improvement is positive when the networked model is better:
    MAE/RMSE → Δ = isolated_mean − networked_mean (lower error is better).
    F1/Precision/Recall → Δ = networked_mean − isolated_mean.

    Args:
        all_results: Per-seed result dicts from ``run_one_seed``.

    Returns:
        Dict keyed by metric with ``networked_mean``, ``networked_std``,
        ``isolated_mean``, ``isolated_std``, and ``delta``.
    """
    net = {m: [r["networked"][m] for r in all_results] for m in METRIC_NAMES}
    iso = {m: [r["isolated"][m] for r in all_results] for m in METRIC_NAMES}
    stats: Dict[str, Any] = {}
    for m in METRIC_NAMES:
        nm, ns = float(np.mean(net[m])), float(np.std(net[m]))
        im, is_ = float(np.mean(iso[m])), float(np.std(iso[m]))
        delta: float = (im - nm) if m in ("mae", "rmse") else (nm - im)
        stats[m] = {
            "networked_mean": nm, "networked_std": ns,
            "isolated_mean": im, "isolated_std": is_,
            "delta": delta,
        }

    cw = (12, 26, 26, 16)
    sep = "+" + "+".join("-" * w for w in cw) + "+"
    header = (
        f"| {'Metric':<{cw[0]-2}} "
        f"| {'Networked (mean±std)':<{cw[1]-2}} "
        f"| {'Isolated (mean±std)':<{cw[2]-2}} "
        f"| {'Δ improvement':<{cw[3]-2}} |"
    )
    print(f"\n[robust] {sep}\n[robust] {header}\n[robust] {sep}")
    for m in METRIC_NAMES:
        s = stats[m]
        d = s["delta"]
        row = (
            f"| {m:<{cw[0]-2}} "
            f"| {s['networked_mean']:.4f} ± {s['networked_std']:.4f}  "
            f"| {s['isolated_mean']:.4f} ± {s['isolated_std']:.4f}  "
            f"| {'+' if d >= 0 else ''}{d:.4f}{'':>{cw[3]-8}} |"
        )
        print(f"[robust] {row}")
    print(f"[robust] {sep}\n")
    return stats


def main() -> None:
    """Run robustness evaluation across all SEEDS and save results to JSON."""
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[robust] Device: {device}  |  Seeds: {SEEDS}")

    if not PKL_PATH.exists():
        raise FileNotFoundError(
            f"[robust] {PKL_PATH} not found. "
            "Run python -m src.simulation.generate_data first."
        )

    all_results: List[Dict[str, Dict[str, float]]] = []
    for i, seed in enumerate(SEEDS):
        print(f"\n[robust] ── Seed {seed} ({i + 1}/{len(SEEDS)}) ──────────────────────────")
        result = run_one_seed(seed, PKL_PATH, device)
        all_results.append(result)
        for approach, metrics in result.items():
            vals = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"[robust]   {approach:<12}: {vals}")

    stats = summarize(all_results)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output: Dict[str, Any] = {
        "seeds": SEEDS,
        "binary_threshold": BINARY_THRESHOLD,
        "per_seed": all_results,
        "summary": stats,
    }
    with open(RESULTS_PATH, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"[robust] Results saved → {RESULTS_PATH}")

if __name__ == "__main__":
    main()
