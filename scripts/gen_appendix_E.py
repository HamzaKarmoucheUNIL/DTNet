# scripts/gen_appendix_E.py
# Generates Appendix E: GNN training details (LaTeX tables + loss curve figure).
# Reads only from src/ — does NOT modify any file under src/.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)

from src.gnn.dataset import BATCH_SIZE, PKL_PATH, TRAIN_RATIO, VAL_RATIO, build_dataloaders
from src.gnn.model import EDGE_DIM, DTNetGNN, IsolatedBaseline
from src.gnn.train import (
    DROPOUT, HIDDEN_CHANNELS, LEARNING_RATE, MAX_EPOCHS,
    NUM_HEADS, PATIENCE, WEIGHT_DECAY, train_model,
)

IN_CHANNELS: int = 16
TEST_RATIO: float = round(1.0 - TRAIN_RATIO - VAL_RATIO, 2)
OUT_DIR: Path = Path("scripts/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEP = "=" * 62

# ─────────────────────────────────────────────────────────────────────────────
# A — Print extracted constants
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("A — EXTRACTED TRAINING CONSTANTS")
print(SEP)
print(f"  weight_decay         : {WEIGHT_DECAY}")
print(f"  batch_size           : {BATCH_SIZE}")
print(f"  max_epochs           : {MAX_EPOCHS}")
print(f"  early_stop_patience  : {PATIENCE}")
print(f"  train/val/test split : {TRAIN_RATIO:.0%} / {VAL_RATIO:.0%} / {TEST_RATIO:.0%}")
print(f"  grid criterion       : score = val_mse - val_f1  (lower is better)")
print()

# ─────────────────────────────────────────────────────────────────────────────
# B — Top-N trials LaTeX table
# ─────────────────────────────────────────────────────────────────────────────
BEST_HP_PATH = Path("results/best_hyperparams.json")

if BEST_HP_PATH.exists():
    with open(BEST_HP_PATH) as fh:
        best = json.load(fh)
    rows_B = [(1, best["hidden"], best["heads"], best["lr"], best["dropout"],
               f"{best['score']:.4f}")]
    note_B = (
        r"Only the single best configuration was persisted "
        r"(\texttt{results/best\_hyperparams.json}); "
        r"score $= \text{val\_MSE} - \text{val\_F1}$, lower is better."
    )
else:
    rows_B = [(1, HIDDEN_CHANNELS, NUM_HEADS, LEARNING_RATE, DROPOUT, r"\textemdash")]
    note_B = (
        r"\texttt{results/best\_hyperparams.json} not found; "
        r"configuration taken from \texttt{train.py} constants. "
        r"Selection criterion: $\text{score} = \text{val\_MSE} - \text{val\_F1}$ "
        r"(lower is better); exact grid-search score not stored."
    )

tex_B = OUT_DIR / "appE_top_trials.tex"
with open(tex_B, "w", encoding="utf-8") as fh:
    fh.write("\\small\n\\begin{table}[h]\n\\centering\n")
    fh.write(f"\\caption{{Grid-search best hyperparameter configuration. {note_B}}}\n")
    fh.write("\\label{tab:appE-trials}\n")
    fh.write("\\begin{tabular}{c r r r r r}\n\\toprule\n")
    fh.write(
        "\\textbf{Rank} & \\textbf{Hidden} & \\textbf{Heads} & "
        "\\textbf{LR} & \\textbf{Dropout} & \\textbf{Score}\\\\\n\\midrule\n"
    )
    for rank, hidden, heads, lr, dropout, score_s in rows_B:
        lr_s = f"{lr:.4f}" if isinstance(lr, float) else str(lr)
        dr_s = f"{dropout}" if not isinstance(dropout, str) else dropout
        fh.write(f"{rank} & {hidden} & {heads} & {lr_s} & {dr_s} & {score_s}\\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

print(f"[B] Written {tex_B}")

# ─────────────────────────────────────────────────────────────────────────────
# C — Trainable parameter count
# ─────────────────────────────────────────────────────────────────────────────
gnn_model = DTNetGNN(
    in_channels=IN_CHANNELS,
    hidden_channels=HIDDEN_CHANNELS,
    edge_dim=EDGE_DIM,
    heads_1=NUM_HEADS,
    dropout=DROPOUT,
)
base_model = IsolatedBaseline(
    in_channels=IN_CHANNELS,
    hidden_channels=HIDDEN_CHANNELS,
)

gnn_n  = sum(p.numel() for p in gnn_model.parameters()  if p.requires_grad)
base_n = sum(p.numel() for p in base_model.parameters() if p.requires_grad)

print("\nC — DTNetGNN per-layer parameter breakdown")
print(f"  {'Layer':<50} {'Shape':<22} Params")
print(f"  {'-'*50} {'-'*22} ------")
for name, param in gnn_model.named_parameters():
    if param.requires_grad:
        print(f"  {name:<50} {str(list(param.shape)):<22} {param.numel():>8,}")
print(f"\n  TOTAL DTNetGNN        : {gnn_n:>10,}")
print(f"  TOTAL IsolatedBaseline: {base_n:>10,}")

tex_C = OUT_DIR / "appE_params.tex"
with open(tex_C, "w", encoding="utf-8") as fh:
    fh.write("\\small\n\\begin{table}[h]\n\\centering\n")
    fh.write(
        "\\caption{Trainable parameter counts. "
        "DTNetGNN: \\texttt{in\\_channels=16}, \\texttt{hidden=128}, "
        "\\texttt{heads=(4,\\,1)}, \\texttt{edge\\_dim=3}. "
        "IsolatedBaseline: single hidden layer MLP with \\texttt{hidden=128}.}\n"
    )
    fh.write("\\label{tab:appE-params}\n")
    fh.write("\\begin{tabular}{l r}\n\\toprule\n")
    fh.write("\\textbf{Model} & \\textbf{Trainable parameters}\\\\\n\\midrule\n")
    fh.write(f"DTNetGNN & {gnn_n:,}\\\\\n")
    fh.write(f"IsolatedBaseline & {base_n:,}\\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

print(f"\n[C] Written {tex_C}")

# ─────────────────────────────────────────────────────────────────────────────
# D — Loss curves: build graph, train once with seed 42, plot
# ─────────────────────────────────────────────────────────────────────────────
print("\nD — Building graph + dataloaders (this runs preprocess + topology) …")
from src.data.entity_mapping import build_entity_mappings
from src.data.loader import load_csv
from src.data.preprocess import preprocess
from src.graph.builder import build_graph
from src.graph.topology import infer_topology

df_raw = load_csv("updated_data.csv")
df_c, _ = preprocess(df_raw)
em = build_entity_mappings(df_raw)
ns, es = infer_topology(em)
G = build_graph(ns, es, df_c)

train_loader, val_loader, _ = build_dataloaders(PKL_PATH, G=G, batch_size=BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"D — Device: {device}  (training with seed 42, patience={PATIENCE}, max={MAX_EPOCHS})")

torch.manual_seed(42)
np.random.seed(42)

gnn_fresh = DTNetGNN(
    in_channels=IN_CHANNELS,
    hidden_channels=HIDDEN_CHANNELS,
    edge_dim=EDGE_DIM,
    heads_1=NUM_HEADS,
    dropout=DROPOUT,
)

# Save to a separate checkpoint to avoid overwriting the production checkpoint
CURVE_CKPT = Path("results/dtnet_gnn_loss_curve.pt")
hist = train_model(
    gnn_fresh, train_loader, val_loader, CURVE_CKPT, device, label="DTNetGNN"
)

train_losses = hist["train_loss"]
val_losses   = hist["val_loss"]
best_ep      = int(hist["best_epoch"])   # 0-based
n_epochs     = len(train_losses)

epochs = list(range(1, n_epochs + 1))

fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")
ax.set_facecolor("white")
ax.plot(epochs, train_losses, color="#4A9EFF", linewidth=1.8, label="Train loss")
ax.plot(epochs, val_losses,   color="#E74C3C", linewidth=1.8, label="Val loss")
ax.axvline(best_ep + 1, color="#2ECC71", linewidth=1.5, linestyle="--",
           label=f"Best epoch ({best_ep + 1})")
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Combined loss (MSE + 0.5·BCE)", fontsize=11)
ax.set_title("DTNetGNN: Training and Validation Loss Curves", fontsize=12, fontweight="bold")
ax.legend(frameon=True, framealpha=0.9, fontsize=10)
ax.grid(True, color="#CCCCCC", linewidth=0.6, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()

FIG_PATH = Path("results/fig_loss_curves.png")
fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"D — Saved {FIG_PATH}  (total epochs trained={n_epochs}, best={best_ep + 1})")

# ─────────────────────────────────────────────────────────────────────────────
# Final print
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("FINAL SUMMARY — A")
print(SEP)
print(f"  weight_decay   : {WEIGHT_DECAY}")
print(f"  batch_size     : {BATCH_SIZE}")
print(f"  max_epochs     : {MAX_EPOCHS}")
print(f"  split ratios   : {TRAIN_RATIO:.0%} / {VAL_RATIO:.0%} / {TEST_RATIO:.0%}")
print(f"  grid criterion : val_mse - val_f1  (lower is better)")

for tex_path in (tex_B, tex_C):
    print(f"\n{'─'*62}")
    print(f"  {tex_path}")
    print("─" * 62)
    print(tex_path.read_text(encoding="utf-8"))
