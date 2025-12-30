"""Baseline GNN models for graph anomaly detection experiments.

This module provides lightweight, two-layer implementations of three
canonical DGL-based architectures (GCN, GAT, GraphSAGE).  All classes share
the same constructor and forward signatures so they can be instantiated and
invoked interchangeably inside training scripts.
"""

from __future__ import annotations

from typing import Optional

import dgl
import torch
from dgl.nn import GATConv, GraphConv, SAGEConv
from torch import nn
from torch.nn import functional as F


class BaseGNN(nn.Module):
    """Common utilities shared by simple message-passing GNNs."""

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def apply_activation(self, tensor: torch.Tensor) -> torch.Tensor:
        """ReLU activation followed by dropout."""
        return self.dropout(F.relu(tensor))


class GCN(BaseGNN):
    """Two-layer Graph Convolutional Network."""

    def __init__(
        self,
        in_feats: int,
        hid_feats: int,
        num_classes: int,
        dropout: float = 0.5,
        allow_zero_in_degree: bool = True,
    ) -> None:
        super().__init__(dropout)
        self.layer1 = GraphConv(
            in_feats,
            hid_feats,
            norm="both",
            weight=True,
            bias=True,
            allow_zero_in_degree=allow_zero_in_degree,
        )
        self.layer2 = GraphConv(
            hid_feats,
            num_classes,
            norm="both",
            weight=True,
            bias=True,
            allow_zero_in_degree=allow_zero_in_degree,
        )

    def forward(self, graph: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        h = self.apply_activation(self.layer1(graph, feat))
        return self.layer2(graph, h)


class GAT(BaseGNN):
    """Two-layer Graph Attention Network with configurable heads."""

    def __init__(
        self,
        in_feats: int,
        hid_feats: int,
        num_classes: int,
        dropout: float = 0.5,
        num_heads: int = 4,
        attn_drop: float = 0.2,
        feat_drop: float = 0.2,
        allow_zero_in_degree: bool = True,
    ) -> None:
        super().__init__(dropout)
        self.num_heads = num_heads
        self.layer1 = GATConv(
            in_feats,
            hid_feats,
            num_heads=num_heads,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            residual=False,
            activation=None,
            allow_zero_in_degree=allow_zero_in_degree,
        )
        self.layer2 = GATConv(
            hid_feats * num_heads,
            num_classes,
            num_heads=1,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            residual=False,
            activation=None,
            allow_zero_in_degree=allow_zero_in_degree,
        )

    def forward(self, graph: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        h = self.layer1(graph, feat)
        h = h.flatten(1)  # concatenate attention heads
        h = self.apply_activation(h)
        h = self.layer2(graph, h)
        return h.squeeze(1)


class GraphSAGE(BaseGNN):
    """Two-layer GraphSAGE with mean aggregation."""

    def __init__(
        self,
        in_feats: int,
        hid_feats: int,
        num_classes: int,
        dropout: float = 0.5,
        aggregator_type: str = "mean",
        feat_drop: float = 0.0,
    ) -> None:
        super().__init__(dropout)
        self.layer1 = SAGEConv(
            in_feats,
            hid_feats,
            aggregator_type=aggregator_type,
            feat_drop=feat_drop,
            bias=True,
        )
        self.layer2 = SAGEConv(
            hid_feats,
            num_classes,
            aggregator_type=aggregator_type,
            feat_drop=feat_drop,
            bias=True,
        )

    def forward(self, graph: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        h = self.apply_activation(self.layer1(graph, feat))
        return self.layer2(graph, h)


__all__ = ["GCN", "GAT", "GraphSAGE"]

