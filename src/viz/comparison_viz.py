"""comparison_viz.py — Five thesis-ready figures comparing GNN vs IsolatedBaseline.

Saves publication-quality PNG files to results/thesis_figures/ at 200 DPI.
Dark theme (#0a0e17) throughout. Every figure is self-explanatory for the thesis.

Public API: ``generate_all_figures(eval_results, attention, scenario_result,
                                    runs, history, save_dir) -> Dict``
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from src.gnn.evaluate import LAYER_ORDER, ONE_HOT_START

np.random.seed(42)
torch.manual_seed(42)

BG = 'white'
LAYER_COLORS = {
    'supplier': '#4c7aff', 'logistics': '#9b59b6', 'plant': '#2ecc71',
    'machine': '#f39c12',  'distribution': '#e74c3c',
}
CASCADE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'cascade', ['#D6EAF8', '#F39C12', '#E74C3C']
)
DATE             = date.today().strftime('%Y%m%d')
SAVE_DIR_DEFAULT = Path('results/thesis_figures')

matplotlib.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG, 'axes.edgecolor': '#CCCCCC',
    'axes.labelcolor': '#333333', 'xtick.color': '#333333', 'ytick.color': '#333333',
    'text.color': '#333333', 'grid.color': '#EEEEEE', 'grid.alpha': 0.8,
    'legend.facecolor': '#F8F8F8', 'legend.edgecolor': '#CCCCCC',
})


def _savefig(fig: plt.Figure, save_dir: Path, stem: str) -> Path:
    """Save fig at 200 DPI, close it, and return the saved Path."""
    p = Path(save_dir) / f'dtnet_{stem}_{DATE}.png'
    fig.savefig(p, dpi=200, bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.close(fig)
    print(f'  [viz] {p.name}')
    return p


def _setup_ax(
    ax: plt.Axes, title: str, xlabel: str = '', ylabel: str = '', grid: bool = True
) -> None:
    """Apply light-theme styling to ax — title, labels, grid."""
    ax.set_facecolor(BG)
    ax.set_title(title, color='#333333', fontsize=11, pad=9)
    if xlabel: ax.set_xlabel(xlabel, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10)
    if grid:   ax.grid(True, alpha=0.6)


def _hierarchical_pos(node_order: List[str], layer_ids: np.ndarray) -> Dict[str, tuple]:
    """Return {node_id: (x, y)} with supply-chain layers stacked (supplier=top)."""
    y_map:   Dict[int, int] = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}
    buckets: List[List[str]] = [[] for _ in range(5)]
    for i, nid in enumerate(node_order):
        buckets[int(layer_ids[i])].append(nid)
    xs  = max(1.8, 12.0 / max((len(b) for b in buckets), default=1))
    pos: Dict[str, tuple] = {}
    for li, nodes in enumerate(buckets):
        n = len(nodes)
        for k, nid in enumerate(nodes):
            pos[nid] = ((k - (n - 1) / 2.0) * xs, y_map[li])
    return pos


def _draw_network_panel(
    ax: plt.Axes, VizG: nx.DiGraph, pos: Dict,
    node_order: List[str], layer_ids: np.ndarray,
    severities: np.ndarray, title: str,
) -> None:
    """Draw one supply-chain graph panel, nodes coloured by disruption severity."""
    cols: List = [CASCADE_CMAP(float(severities[i])) for i in range(len(node_order))]
    nx.draw_networkx_edges(VizG, pos, ax=ax, edge_color='#AAAACC',
                           arrows=True, arrowsize=10, alpha=0.55, width=0.8)
    nx.draw_networkx_nodes(VizG, pos, node_color=cols, node_size=380,
                           ax=ax, linewidths=1.0, edgecolors='#555555')
    nx.draw_networkx_labels(VizG, pos, ax=ax, font_size=5.5, font_color='#333333')
    ax.set_title(title, color='#333333', fontsize=11, pad=8)
    ax.set_facecolor(BG)
    ax.axis('off')


def plot_networked_vs_isolated(eval_results: Dict, save_dir: Path) -> Path:
    """Figure 1: Bar chart of MSE, MAE, R² — DTNetGNN vs IsolatedBaseline."""
    gm, bm = eval_results['gnn_test'], eval_results['baseline_test']
    specs  = [('MSE', gm['mse'], bm['mse']),
              ('MAE', gm['mae'], bm['mae']),
              ('R²',  gm['r2'],  bm['r2'])]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=BG)
    fig.suptitle('Networked GNN vs Isolated Baseline — Overall Performance on Test Set',
                 color='#333333', fontsize=14, y=1.02)

    for ax, (lbl, gv, bv) in zip(axes, specs):
        bars = ax.bar(['DTNetGNN', 'IsolatedBaseline'], [gv, bv],
                      color=['#4c7aff', '#9b59b6'], width=0.5, zorder=3,
                      edgecolor='#CCCCCC', linewidth=0.5)
        for bar, v in zip(bars, [gv, bv]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(bar.get_height()) * 0.03,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=10, color='#333333')
        if lbl in ('MSE', 'MAE') and bv > 0:
            pct = (bv - gv) / bv * 100
            ax.text(0.5, 0.94, f'GNN {pct:+.1f}% vs Baseline',
                    transform=ax.transAxes, ha='center', fontsize=8.5, color='#27AE60',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#EAF9EA', alpha=0.9))
        if lbl == 'R²':
            ax.axhline(0, color='gray', lw=0.8, alpha=0.5)
        ybot = min(0.0, gv, bv) * 1.1 if lbl == 'R²' else 0.0
        ax.set_ylim(ybot, max(gv, bv) * 1.28)
        _setup_ax(ax, lbl, ylabel=lbl, grid=False)

    plt.tight_layout()
    return _savefig(fig, save_dir, 'fig1_networked_vs_isolated')


def plot_cascade_spread(
    scenario_result: Dict, runs: List[Dict], save_dir: Path
) -> Path:
    """Figure 2: Side-by-side graph — actual cascade vs GNN-predicted cascade."""
    node_order = scenario_result['node_order']
    y_true     = np.array(scenario_result['y_true'])
    gnn_pred   = np.array(scenario_result['gnn_pred'])
    run0       = runs[0]
    layer_ids  = np.argmax(
        np.array(run0['initial_features'])[:, ONE_HOT_START: ONE_HOT_START + 5], axis=1
    )
    pos  = _hierarchical_pos(node_order, layer_ids)
    VizG = nx.DiGraph()
    VizG.add_nodes_from(node_order)
    for ep in run0['edge_index']:
        VizG.add_edge(node_order[ep[0]], node_order[ep[1]])

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=BG)
    _draw_network_panel(axes[0], VizG, pos, node_order, layer_ids, y_true,
                        'Ground Truth — Actual Cascade (Simulation)')
    _draw_network_panel(axes[1], VizG, pos, node_order, layer_ids, gnn_pred,
                        'DTNetGNN Prediction')

    for ax in axes:
        for nid in scenario_result['initial_disruption']:
            if nid in pos:
                ax.scatter(*pos[nid], s=320, marker='*', color='#ffd700',
                           zorder=10, linewidths=0)

    sm = plt.cm.ScalarMappable(cmap=CASCADE_CMAP, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.75, pad=0.02, aspect=30)
    cbar.set_label('Disruption Severity  [0 = healthy · 1 = failed]',
                   color='#333333', fontsize=10)
    cbar.ax.tick_params(colors='#333333')

    patches = [mpatches.Patch(color=LAYER_COLORS[l], label=l.capitalize())
               for l in LAYER_ORDER]
    patches.append(mpatches.Patch(color='#ffd700', label='★ Seed node'))
    axes[0].legend(handles=patches, loc='lower left', fontsize=8.5, framealpha=0.88)

    seed_str = ', '.join(list(scenario_result['initial_disruption'].keys()))
    fig.suptitle(
        f'Cascading Disruption Spread — Single Supplier Failure\nSeed: {seed_str}',
        color='#333333', fontsize=13, y=1.02,
    )
    plt.tight_layout()
    return _savefig(fig, save_dir, 'fig2_cascade_spread')


def plot_accuracy_by_node_type(eval_results: Dict, save_dir: Path) -> Path:
    """Figure 3: Grouped bar chart — MAE and R² per supply-chain layer type."""
    pg     = eval_results['per_type_gnn']
    pb     = eval_results['per_type_base']
    layers = [l for l in LAYER_ORDER if l in pg and l in pb]
    x, w   = np.arange(len(layers)), 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    fig.suptitle('Prediction Accuracy by Supply-Chain Node Type — GNN vs Isolated Baseline',
                 color='#333333', fontsize=14, y=1.02)

    for ax, metric, title in [
        (axes[0], 'mae', 'Mean Absolute Error per Node Type  (lower = better)'),
        (axes[1], 'r2',  'R²  per Node Type   (higher = better)'),
    ]:
        gv = [pg[l][metric] for l in layers]
        bv = [pb[l][metric] for l in layers]
        b1 = ax.bar(x - w / 2, gv, w, label='DTNetGNN',
                    color='#4c7aff', zorder=3, edgecolor='#CCCCCC', linewidth=0.4)
        b2 = ax.bar(x + w / 2, bv, w, label='IsolatedBaseline',
                    color='#9b59b6', zorder=3, edgecolor='#CCCCCC', linewidth=0.4)
        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + abs(h) * 0.03,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=7.5, color='#333333')
        ax.set_xticks(x)
        ax.set_xticklabels([l.capitalize() for l in layers], rotation=20, ha='right')
        if metric == 'r2':
            ax.axhline(0, color='gray', lw=0.8, alpha=0.5)
        _setup_ax(ax, title, ylabel=metric.upper())
        ax.legend(fontsize=9)

    plt.tight_layout()
    return _savefig(fig, save_dir, 'fig3_accuracy_by_node_type')


def generate_all_figures(
    eval_results: Dict[str, Any],
    attention: Dict[str, Any],
    scenario_result: Dict[str, Any],
    runs: List[Dict[str, Any]],
    history: Optional[List[Dict[str, Any]]] = None,
    save_dir: Path = SAVE_DIR_DEFAULT,
) -> Dict[str, Optional[Path]]:
    """Generate all five thesis figures, save to save_dir, return {name: Path}.

    Figures 4 & 5 (attention heatmap, propagation timeline) are provided by
    ``src.viz._attention_timeline_viz``.  Pass ``history`` for Figure 5; if
    None, that figure is skipped and its entry is None.
    """
    from src.viz._attention_timeline_viz import (  # lazy import avoids circular ref
        plot_attention_heatmap, plot_propagation_timeline,
    )
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    print(f'[viz] Generating thesis figures → {Path(save_dir).resolve()}')
    out: Dict[str, Optional[Path]] = {
        'networked_vs_isolated': plot_networked_vs_isolated(eval_results, save_dir),
        'cascade_spread':        plot_cascade_spread(scenario_result, runs, save_dir),
        'accuracy_by_node_type': plot_accuracy_by_node_type(eval_results, save_dir),
        'attention_heatmap':     plot_attention_heatmap(attention, runs, save_dir),
        'propagation_timeline':  (plot_propagation_timeline(history, runs, save_dir)
                                  if history else None),
    }
    saved = sum(p is not None for p in out.values())
    print(f'[viz] Done — {saved}/5 figures saved to results/thesis_figures/')
    return out
