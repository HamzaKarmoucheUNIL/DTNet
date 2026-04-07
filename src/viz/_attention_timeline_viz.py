"""_attention_timeline_viz.py — Figures 4 & 5: attention heatmap and timeline.

Imported by comparison_viz.generate_all_figures().  Depends on shared
constants and helpers from comparison_viz (BG, CASCADE_CMAP, LAYER_COLORS,
LAYER_ORDER, ONE_HOT_START, _savefig, _setup_ax).

Public API: plot_attention_heatmap, plot_propagation_timeline
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)

from src.viz.comparison_viz import (  # noqa: E402 — circular-safe: imported at call time
    BG, CASCADE_CMAP, LAYER_COLORS, LAYER_ORDER, ONE_HOT_START,
    _savefig, _setup_ax,
)


def plot_attention_heatmap(attention: Dict, runs: List[Dict], save_dir: Path) -> Path:
    """Figure 4: Top-K edge attention bar chart + 5×5 layer-to-layer flow matrix."""
    top_k      = attention['top_k_edges']
    run0       = runs[0]
    layer_ids  = np.argmax(np.array(run0['initial_features'])[:, ONE_HOT_START:ONE_HOT_START + 5], axis=1).astype(int)
    top_k_dict = {(e['src'], e['dst']): e['attention'] for e in top_k}

    L      = len(LAYER_ORDER)
    matrix = np.zeros((L, L))
    for ep in run0['edge_index']:
        sl, dl = int(layer_ids[ep[0]]), int(layer_ids[ep[1]])
        matrix[sl, dl] += top_k_dict.get((ep[0], ep[1]), 0.0)
    row_sums = matrix.sum(axis=1, keepdims=True)
    mat_norm = np.divide(matrix, row_sums, where=row_sums > 0)

    fig = plt.figure(figsize=(16, 6), facecolor=BG)
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.38)
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    labels = [
        f'{e.get("src_name", "node_" + str(e["src"]))} → '
        f'{e.get("dst_name", "node_" + str(e["dst"]))}'
        for e in top_k
    ]
    vals  = [e['attention'] for e in top_k]
    max_v = max(vals) if vals else 1.0
    ax1.barh(range(len(labels)), vals, height=0.7, edgecolor='white', linewidth=0.3,
             color=[plt.cm.YlOrRd(v / max_v) for v in vals])
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.invert_yaxis()
    _setup_ax(ax1, f'Top-{len(top_k)} Most-Attended Supply-Chain Edges',
              xlabel='Mean GAT Attention Weight (averaged over both heads & layers)')

    im  = ax2.imshow(mat_norm, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    tks = [l.capitalize() for l in LAYER_ORDER]
    ax2.set_xticks(range(L)); ax2.set_xticklabels(tks, rotation=30, ha='right', fontsize=9)
    ax2.set_yticks(range(L)); ax2.set_yticklabels(tks, fontsize=9)
    ax2.set_xlabel('Destination Layer', fontsize=10)
    ax2.set_ylabel('Source Layer', fontsize=10)
    ax2.set_title('Layer-to-Layer Attention Flow\n(row-normalised, top-K edges only)',
                  color='white', fontsize=10, pad=8)
    for i in range(L):
        for j in range(L):
            if mat_norm[i, j] > 0.01:
                ax2.text(j, i, f'{mat_norm[i, j]:.2f}', ha='center', va='center',
                         fontsize=8.5, color='black' if mat_norm[i, j] > 0.5 else 'white')
    cb = fig.colorbar(im, ax=ax2, shrink=0.82, pad=0.03)
    cb.set_label('Normalised Attention', color='white', fontsize=9)
    cb.ax.tick_params(colors='white')

    fig.suptitle('GAT Attention Analysis — Which Supply-Chain Connections Are Critical?',
                 color='white', fontsize=13, y=1.02)
    plt.tight_layout()
    return _savefig(fig, save_dir, 'fig4_attention_heatmap')


def plot_propagation_timeline(
    history: List[Dict[str, Any]], runs: List[Dict], save_dir: Path
) -> Optional[Path]:
    """Figure 5: Gantt-style disruption timeline + network health curve."""
    if not history:
        return None
    first_t: Dict[str, int] = {}
    final_sev: Dict[str, float] = {}
    for step in history:
        t = step['timestep']
        for nid in step.get('newly_disrupted', []):
            first_t.setdefault(nid, t)
        for nid, st in step.get('node_states', {}).items():
            final_sev[nid] = st['disruption_severity']
    if not first_t:
        return None

    T      = len(history)
    order  = sorted(first_t, key=lambda n: (first_t[n], -final_sev.get(n, 0.0)))
    run0   = runs[0]
    layer_ids = np.argmax(np.array(run0['initial_features'])[:, ONE_HOT_START:ONE_HOT_START + 5], axis=1).astype(int)
    n2layer   = {run0['node_order'][i]: LAYER_ORDER[int(layer_ids[i])] for i in range(len(run0['node_order']))}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), facecolor=BG,
                                   gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.38})
    for yi, nid in enumerate(order):
        sev = final_sev.get(nid, 0.0)
        ax1.barh(yi, T - first_t[nid], left=first_t[nid], color=CASCADE_CMAP(sev),
                 height=0.72, alpha=0.88, edgecolor='white', linewidth=0.2)
        ax1.text(first_t[nid] - 0.15, yi, nid, ha='right', va='center', fontsize=6.5, color='white')
        ax1.scatter([-0.9], [yi], color=LAYER_COLORS.get(n2layer.get(nid, ''), '#888888'),
                    s=28, zorder=5, clip_on=False)

    ax1.set_xlim(-2.5, T + 0.2); ax1.set_yticks([])
    ax1.grid(True, axis='x', alpha=0.3); ax1.set_facecolor(BG)
    ax1.set_title('Disruption Propagation Timeline', color='white', fontsize=12, pad=8)
    ax1.set_xlabel('Simulation Timestep', fontsize=10)

    sm = plt.cm.ScalarMappable(cmap=CASCADE_CMAP, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax1, pad=0.01, shrink=0.82, aspect=22)
    cb.set_label('Final Severity', color='white', fontsize=9)
    cb.ax.tick_params(colors='white')
    patches = [mpatches.Patch(color=LAYER_COLORS[l], label=l.capitalize()) for l in LAYER_ORDER]
    ax1.legend(handles=patches, loc='lower right', fontsize=8, framealpha=0.78)

    ts, hlth = [h['timestep'] for h in history], [h['network_health'] for h in history]
    ax2.fill_between(ts, hlth, alpha=0.30, color='#2ecc71')
    ax2.plot(ts, hlth, color='#2ecc71', lw=2.0)
    ax2.set_ylim(0, 1.05)
    _setup_ax(ax2, 'Network Health Over Time', xlabel='Simulation Timestep', ylabel='Avg Health Score')

    fig.suptitle('Disruption Propagation Through the Supply-Chain Network',
                 color='white', fontsize=14, y=1.01)
    return _savefig(fig, save_dir, 'fig5_propagation_timeline')
