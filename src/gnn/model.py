"""model.py — GNN and baseline model definitions for DTNet disruption prediction.

Defines two models:
  - ``DTNetGNN``: Graph Attention Network that predicts disruption severity
    per node using both node features and graph topology/edge attributes.
  - ``IsolatedBaseline``: MLP that predicts disruption severity per node
    using only node features — no graph structure.  Used as the RQ3 baseline
    to quantify the value of modelling supply-chain interconnections.

Both models have two output heads:
  - Regression head: predicted disruption severity (float in [0, 1]).
  - Classification head: raw logit for binary disrupted/not-disrupted
    (threshold 0.3; pass to BCEWithLogitsLoss during training).
Both ``forward()`` calls return ``(reg_out, cls_out)``.
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

IN_CHANNELS: int = 16       # node feature dimension: 10 twin + 6 structural (see dataset.py)
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
    reg_head: Linear(64 → 1) + Sigmoid   → severity ∈ [0, 1] per node
    cls_head: Linear(64 → 1)             → raw logit (disrupted yes/no)

    Args:
        in_channels: Number of input node features. Default 16.
        hidden_channels: Per-head hidden dimension. Default 64.
        edge_dim: Edge feature dimension. Default 3.
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

        # Dual output heads (share the same GAT backbone)
        self.reg_head: nn.Linear = nn.Linear(hidden_channels, 1)   # regression
        self.cls_head: nn.Linear = nn.Linear(hidden_channels, 1)   # classification logit

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute disruption severity and binary disruption logit for every node.

        Args:
            x: Node feature matrix of shape ``(N, in_channels)``.
            edge_index: Graph connectivity of shape ``(2, E)`` (COO format).
            edge_attr: Edge feature matrix of shape ``(E, edge_dim)``.
                Pass ``None`` if edge attributes are unavailable; the GAT
                layers will fall back to structure-only attention.

        Returns:
            Tuple ``(reg_out, cls_out)`` where:
            - ``reg_out``: shape ``(N,)``, predicted severity in [0, 1].
            - ``cls_out``: shape ``(N,)``, raw logit for binary disruption
              classification (disrupted if logit > 0, i.e. sigmoid > 0.5;
              use BCEWithLogitsLoss during training).
        """
        # GATConv layer 1 — learns which edges matter most
        x = self.conv1(x, edge_index, edge_attr=edge_attr)  # (N, 256)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GATConv layer 2 — aggregates neighbourhood context
        x = self.conv2(x, edge_index, edge_attr=edge_attr)  # (N, 64)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Dual output heads
        reg_out: torch.Tensor = torch.sigmoid(self.reg_head(x)).squeeze(-1)  # (N,)
        cls_out: torch.Tensor = self.cls_head(x).squeeze(-1)                 # (N,) raw logit
        return reg_out, cls_out


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
    reg_head: Linear(hidden_channels → 1) + Sigmoid → severity ∈ [0, 1]
    cls_head: Linear(hidden_channels → 1)           → raw logit (disrupted yes/no)

    Args:
        in_channels: Number of input node features. Default 16.
        hidden_channels: Hidden layer size. Default 64.
    """

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        hidden_channels: int = MLP_HIDDEN,
    ) -> None:
        super().__init__()

        self.fc1: nn.Linear = nn.Linear(in_channels, hidden_channels)
        self.reg_head: nn.Linear = nn.Linear(hidden_channels, 1)   # regression
        self.cls_head: nn.Linear = nn.Linear(hidden_channels, 1)   # classification logit

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,   # accepted but ignored
        edge_attr: torch.Tensor | None = None,    # accepted but ignored
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute disruption severity and binary logit for every node using features only.

        The ``edge_index`` and ``edge_attr`` arguments are accepted for API
        compatibility with ``DTNetGNN`` but are intentionally not used.

        Args:
            x: Node feature matrix of shape ``(N, in_channels)``.
            edge_index: Ignored. Kept for interface parity with DTNetGNN.
            edge_attr: Ignored. Kept for interface parity with DTNetGNN.

        Returns:
            Tuple ``(reg_out, cls_out)`` where:
            - ``reg_out``: shape ``(N,)``, predicted severity in [0, 1].
            - ``cls_out``: shape ``(N,)``, raw logit for binary disruption
              classification (use BCEWithLogitsLoss during training).
        """
        h: torch.Tensor = F.relu(self.fc1(x))                               # (N, hidden_channels)
        reg_out: torch.Tensor = torch.sigmoid(self.reg_head(h)).squeeze(-1)  # (N,)
        cls_out: torch.Tensor = self.cls_head(h).squeeze(-1)                 # (N,) raw logit
        return reg_out, cls_out
