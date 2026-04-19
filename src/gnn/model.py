"""model.py — GNN and baseline model definitions for DTNet disruption prediction.

Defines two models:
  - ``DTNetGNN``: Graph Attention Network that predicts disruption severity
    per node using both node features and graph topology/edge attributes.
  - ``IsolatedBaseline``: MLP that predicts disruption severity per node
    using only node features — no graph structure.  Used as the RQ3 baseline
    to quantify the value of modelling supply-chain interconnections.

Both models output one scalar in [0, 1] per node (severity).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv

np.random.seed(42)
torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IN_CHANNELS: int = 10       # node feature dimension (see dataset.py)
HIDDEN_CHANNELS: int = 64   # hidden / per-head output dimension
EDGE_DIM: int = 3           # edge feature dimension: [criticality_weight, flow_capacity, shared_parts_count]
GAT_HEADS_1: int = 4        # attention heads in first GATConv layer
GAT_HEADS_2: int = 1        # attention heads in second GATConv layer
DROPOUT_P: float = 0.3      # dropout probability applied after each GAT layer
MLP_HIDDEN: int = 64        # hidden size for IsolatedBaseline MLP


# ---------------------------------------------------------------------------
# DTNetGNN
# ---------------------------------------------------------------------------


class DTNetGNN(nn.Module):
    """Graph Attention Network for node-level disruption severity prediction.

    Uses two GATConv layers with edge-feature support so the model can learn
    which supply-chain connections matter most (via attention) and incorporate
    ``criticality_weight`` / ``flow_capacity`` edge attributes directly.

    Architecture
    ------------
    GATConv(in_channels → 64, heads=4)   → concat → 256-dim per node
    Dropout(0.3)
    GATConv(256 → 64, heads=1)           → 64-dim per node
    Dropout(0.3)
    Linear(64 → 1)
    Sigmoid                              → severity ∈ [0, 1] per node

    Args:
        in_channels: Number of input node features. Default 10.
        hidden_channels: Per-head hidden dimension. Default 64.
        edge_dim: Edge feature dimension. Default 2.
        heads_1: Number of attention heads for the first GAT layer. Default 4.
        heads_2: Number of attention heads for the second GAT layer. Default 1.
        dropout: Dropout probability. Default 0.3.
    """

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        hidden_channels: int = HIDDEN_CHANNELS,
        edge_dim: int = EDGE_DIM,
        heads_1: int = GAT_HEADS_1,
        heads_2: int = GAT_HEADS_2,
        dropout: float = DROPOUT_P,
    ) -> None:
        super().__init__()

        self.dropout: float = dropout

        # Layer 1: in_channels → hidden_channels per head, concat → hidden_channels * heads_1
        self.conv1: GATConv = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads_1,
            edge_dim=edge_dim,
            concat=True,    # output shape: (N, hidden_channels * heads_1)
        )

        # Layer 2: hidden_channels * heads_1 → hidden_channels, single head
        self.conv2: GATConv = GATConv(
            in_channels=hidden_channels * heads_1,
            out_channels=hidden_channels,
            heads=heads_2,
            edge_dim=edge_dim,
            concat=False,   # output shape: (N, hidden_channels)
        )

        # Final projection to a single severity score per node
        self.lin: nn.Linear = nn.Linear(hidden_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute disruption severity for every node in the batch.

        Args:
            x: Node feature matrix of shape ``(N, in_channels)``.
            edge_index: Graph connectivity of shape ``(2, E)`` (COO format).
            edge_attr: Edge feature matrix of shape ``(E, edge_dim)``.
                Pass ``None`` if edge attributes are unavailable; the GAT
                layers will fall back to structure-only attention.

        Returns:
            Tensor of shape ``(N,)`` with predicted severity in [0, 1].
        """
        # GATConv layer 1 — learns which edges matter most
        x = self.conv1(x, edge_index, edge_attr=edge_attr)  # (N, 256)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GATConv layer 2 — aggregates neighbourhood context
        x = self.conv2(x, edge_index, edge_attr=edge_attr)  # (N, 64)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Project to scalar and squash to [0, 1]
        x = self.lin(x)          # (N, 1)
        x = torch.sigmoid(x)     # (N, 1)
        return x.squeeze(-1)     # (N,)


# ---------------------------------------------------------------------------
# IsolatedBaseline  (RQ3)
# ---------------------------------------------------------------------------


class IsolatedBaseline(nn.Module):
    """MLP baseline that predicts disruption severity without graph structure.

    This model receives the same node features as ``DTNetGNN`` but ignores
    all edge information and graph topology.  It answers RQ3:

        *"What predictive power do we gain from modelling supply-chain
        interconnections, over and above local node features alone?"*

    Architecture
    ------------
    Linear(in_channels → hidden_channels)
    ReLU
    Linear(hidden_channels → 1)
    Sigmoid  → severity ∈ [0, 1] per node

    Args:
        in_channels: Number of input node features. Default 10.
        hidden_channels: Hidden layer size. Default 64.
    """

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        hidden_channels: int = MLP_HIDDEN,
    ) -> None:
        super().__init__()

        self.fc1: nn.Linear = nn.Linear(in_channels, hidden_channels)
        self.fc2: nn.Linear = nn.Linear(hidden_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,   # accepted but ignored
        edge_attr: torch.Tensor | None = None,    # accepted but ignored
    ) -> torch.Tensor:
        """Compute disruption severity for every node using features only.

        The ``edge_index`` and ``edge_attr`` arguments are accepted for API
        compatibility with ``DTNetGNN`` but are intentionally not used.

        Args:
            x: Node feature matrix of shape ``(N, in_channels)``.
            edge_index: Ignored. Kept for interface parity with DTNetGNN.
            edge_attr: Ignored. Kept for interface parity with DTNetGNN.

        Returns:
            Tensor of shape ``(N,)`` with predicted severity in [0, 1].
        """
        x = F.relu(self.fc1(x))   # (N, hidden_channels)
        x = self.fc2(x)            # (N, 1)
        x = torch.sigmoid(x)       # (N, 1)
        return x.squeeze(-1)       # (N,)
